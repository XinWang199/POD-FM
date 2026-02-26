#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, math, pickle
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 添加 TensorBoard 导入
from tqdm import tqdm
from model.videoDit_cross_attention_podA_condition import VideoDiT_FM_XAttn
from dataset.pickle_cond_dataset_sst import PickleCondDataset
from timm.scheduler.cosine_lr import CosineLRScheduler

# ===== POD 辅助：加载 / 逐帧最小二乘重建 =====

def load_pod_npz(path):
    z = np.load(path)
    U = torch.from_numpy(z["U"]).float()        # (D, r)
    x_mean = torch.from_numpy(z["mean"]).float()# (D,)
    return U, x_mean

@torch.no_grad()
def build_snapshots_from_train(arr: np.ndarray, train_idx: np.ndarray, seq_len: int):
    """
    用训练集窗口构造 POD 快照矩阵 X （帧独立）。
    - arr: (N,H,W,1) 或 (N,H,W)；会转为 float32 并按x/max
    - train_idx: 窗口起点索引（seq_len=1 时就是帧索引）
    return:
      X: (Nsnap, D) float32, D=H*W*C
      H, W, C
    """
    if arr.ndim == 3:
        arr = arr[..., None]
    N, H, W, C = arr.shape
    assert C == 1, f"Expect C=1, got {C}"

    if seq_len == 1:
        frames = arr[train_idx]                       # (Nsnap, H, W, 1)
    else:
        parts = [arr[i0:i0 + seq_len] for i0 in train_idx]  # list of (L,H,W,1)
        frames = np.concatenate(parts, axis=0)                # (Nsnap, H, W, 1)

    X = frames.reshape(-1, H * W * C).astype(np.float32)       # (Nsnap, D)
    return torch.from_numpy(X), H, W, C

@torch.no_grad()
def compute_pod_basis_torch(X_centered: torch.Tensor, r: int, device="cuda"):
    """
    X_centered: (Nsnap, D) —— 已减均值后的零中心数据
    返回 U: (D, r) 列正交的空间基
    """
    Xc = X_centered.to(device, torch.float32)
    q = min(r + 10, min(Xc.shape) - 1)  # oversampling
    # Xc ≈ U_low @ diag(S) @ V^T；空间基取 V 的前 r 列
    U_low, S, V = torch.pca_lowrank(Xc, q=q, center=False)
    U = V[:, :r].contiguous()
    U = F.normalize(U, dim=0)
    return U  # (D,r)

@torch.no_grad()
def pod_coeffs_from_obs_singleframe_robust(
    y_obs: torch.Tensor,      # (B,C,1,H,W)  单帧
    mask1: torch.Tensor,      # (B,1,1,H,W)
    U: torch.Tensor,          # (D, r_max)  列正交/单位范数
    x_mean: torch.Tensor,     # (D,)
    lam: float = 1e-3,        # 基础λ，会做自适应放缩
    dynamic_r: bool = True,
    alpha_ratio: float = 0.8,
    prior_var: torch.Tensor | None = None,  # (r_max,) 训练集 a 的方差(或能量)；无则不加先验
    column_normalize: bool = True,          # 列归一
    eps: float = 1e-8,
):
    assert y_obs.dim()==5 and mask1.dim()==5
    B, C, T, H, W = y_obs.shape
    assert T == 1, "该函数仅适用于单帧样本(T=1)"
    D, r_max = U.shape
    assert D == C*H*W

    dev, dt = y_obs.device, y_obs.dtype
    U = U.to(dev, dt)
    x_mean = x_mean.to(dev, dt)
    if prior_var is not None:
        prior_var = prior_var.to(dev, dt)  # 方差而不是标准差

    y_flat = y_obs[:, :, 0].permute(0,2,3,1).reshape(B, D)  # (B,D)
    m2d = mask1[:,0,0]                                      # (B,H,W)

    A = torch.zeros(B, r_max, device=dev, dtype=dt)

    eye = None
    for b in range(B):
        idx = (m2d[b].reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(1)  # (K,)
        Kb = int(idx.numel())
        if Kb == 0:
            # 无观测：返回0系数（对应均值）
            continue

        r_eff = r_max if not dynamic_r else max(1, min(r_max, int(alpha_ratio*Kb)))
        US = U.index_select(0, idx)[:, :r_eff]        # (K, r_eff)
        yb = y_flat[b, idx] - x_mean[idx]             # (K,)

        # 列归一（缓解尺度差异/病态）
        if column_normalize:
            cn = US.norm(dim=0).clamp_min(eps)        # (r_eff,)
            USn = US / cn
        else:
            cn = torch.ones(r_eff, device=dev, dtype=dt)
            USn = US

        # 自适应 λ：随设计矩阵规模调节
        AtA = USn.T @ USn                              # (r_eff, r_eff)
        lam_eff = lam * (AtA.trace() / r_eff + eps)

        M = AtA + lam_eff * torch.eye(r_eff, device=dev, dtype=dt)

        # 可选：先验（Bayesian Ridge）：Σ_a = diag(prior_var)
        if prior_var is not None:
            s2 = prior_var[:r_eff].clamp_min(eps)      # 方差越大→惩罚越弱
            M = M + torch.diag(1.0 / s2)

        rhs = USn.T @ yb                               # (r_eff,)

        # Cholesky 解
        L = torch.linalg.cholesky(M)
        a_n = torch.cholesky_solve(rhs[:,None], L).squeeze(1)  # (r_eff,)
        a   = a_n / cn
        A[b, :r_eff] = a

    return A  # (B, r_max)
# ---------------- Utils ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)

def set_seed(seed:int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_ckpt(path, epoch, model, opt, scaler, ema, cfg):
    ck = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": (ema.shadow if ema is not None else None),
        "cfg": cfg,
    }
    torch.save(ck, path); print(f"[Save] {path}")

@torch.no_grad()
def sample_linear_pair(x1: torch.Tensor, eps=1e-3):
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t  = torch.rand(B, device=x1.device)*(1-2*eps)+eps
    tv = t.view(B,1,1,1,1)
    x_t = (1-tv)*x0 + tv*x1
    u_t = x1 - x0
    return x_t, u_t, t

# 添加学习率获取函数
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # 数据/划分
    ap.add_argument("--train_data_path", type=str, default='data/sst1993_2019.bhwc.f32.npy', help="输入训练data文件路径")
    ap.add_argument("--test_data_path", type=str, default='data/sst2020_2021.bhwc.f32.npy', help="输入测试data文件路径")
    ap.add_argument("--seq_len", type=int, default=1, help="=1 为 2D，>1 为窗口")
    # 观测/ROI
    ap.add_argument("--mask_ratio", type=float, default=0.00)
    ap.add_argument("--mask_ratio_min", type=float, default=0.000225)
    ap.add_argument("--mask_ratio_max", type=float, default=0.00225)
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--sample_type", type=str, default="exact_k", choices=["bernoulli","exact_k"])
    # 训练
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema", default=True, help="是否使用 EMA")
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--log_interval", type=int, default=10)
    # 模型
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp_ratio", type=float, default=1.0)
    ap.add_argument("--patch_t", type=int, default=1)
    ap.add_argument("--patch_h", type=int, default=5)
    ap.add_argument("--patch_w", type=int, default=5)
    ap.add_argument("--sensor_pos_dim", type=int, default=32)
    ap.add_argument("--sensor_cap", type=int, default=0)
    # 设备/保存
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_dir", type=str, default="runs/SST_3rank_1e-1")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--resume", type=str, default="runs/SST_3rank_1e-1/best.pt")
    # TensorBoard 配置
    ap.add_argument("--tb_log_dir", type=str, default="", help="TensorBoard日志目录（默认为out_dir/tensorboard）")
    ap.add_argument("--tb_log_every", type=int, default=50, help="每N个batch记录一次详细指标")
    # POD设置
    ap.add_argument("--pod_npz", type=str, default="runs/SST_3rank_1e-1/pod_basis_seq1_r3.npz",
                    help="离线计算好的 POD 基与均值 .npz")
    ap.add_argument("--pod_rank", type=int, default=3, help="POD 截断秩 r")
    ap.add_argument("--pod_lam", type=float, default=1e-1, help="POD LS 正则 λ")
    ap.add_argument("--pod_alpha", type=float, default=1.0, help="动态秩系数：r_eff=floor(alpha*K)，ROI 建议 0.25")
    ap.add_argument("--pod_dynamic_r", default=False, help="按样本 K 自适应截断 POD 秩")

    args = ap.parse_args()

    #set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 初始化 TensorBoard
    tb_log_dir = args.tb_log_dir if args.tb_log_dir else os.path.join(args.out_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TensorBoard] 日志保存到: {tb_log_dir}")
    print(f"[TensorBoard] 启动命令: tensorboard --logdir {tb_log_dir}")

    # 读取训练data
    with open(args.train_data_path, "rb") as f:
        arr_train = np.load(f)
    if isinstance(arr_train, np.ndarray) is False:
        raise ValueError(f"数据集 内容不是 numpy.ndarray: {type(arr_train)}")
    if arr_train.ndim == 3: arr_train = arr_train[..., None]  # (N,H,W,1)
    N, H, W, _ = arr_train.shape
    print(f"[Data] arr_train: {arr_train.shape}, dtype={arr_train.dtype}")
    train_idx = np.arange(N, dtype=np.int64)

    # 读取测试data
    with open(args.test_data_path, "rb") as f:
        arr_test = np.load(f)
    if isinstance(arr_test, np.ndarray) is False:
        raise ValueError(f"数据集 内容不是 numpy.ndarray: {type(arr_test)}")
    if arr_test.ndim == 3: arr_test = arr_test[..., None]  # (N,H,W,1)
    N_test, H_test, W_test, _ = arr_test.shape
    print(f"[Data] arr_test: {arr_test.shape}, dtype={arr_test.dtype}")
    test_idx = np.arange(N_test, dtype=np.int64)


    # 训练集统计量（仅用训练子集）
    tr_mean = float(arr_train.mean())
    tr_std  = float(arr_train.std())
    tr_max  = float(arr_train.max())
    np.save(os.path.join(args.out_dir, "train_mean_std.npy"), np.array([tr_mean, tr_std], dtype=np.float32))
    print(f"[Stats] mean={tr_mean:.6f}, std={tr_std:.6f}")
    print(f"[Stats] max={tr_max:.6f}")

    arr_train = arr_train / tr_max
    arr_test  = arr_test / tr_max

    # ========== POD基处理：优先使用已存在的POD基 ==========
    pod_basis_path = os.path.join(args.out_dir, f"pod_basis_seq{args.seq_len}_r{args.pod_rank}.npz")
    if os.path.exists(pod_basis_path):
        # 直接加载已存在的POD基
        print(f"[POD] Loading existing POD basis from: {pod_basis_path}")
        U, x_mean = load_pod_npz(pod_basis_path)
        U = U.to(device)
        x_mean = x_mean.to(device)
        print(f"[POD] Loaded POD basis: U.shape={U.shape}, x_mean.shape={x_mean.shape}")
    else:
        # 只有POD基不存在时才重新计算
        print("[POD] Computing POD basis ...")

        X, H, W, C = build_snapshots_from_train(arr_train, train_idx, seq_len=args.seq_len)
        x_mean = X.mean(dim=0)              # (D,)
        Xc = X - x_mean[None, :]

        # 计算 U
        U = compute_pod_basis_torch(Xc, r=args.pod_rank, device=device)  # (D,r)
        
        # 保存POD基
        np.savez(pod_basis_path,
                U=U.detach().cpu().numpy().astype(np.float32),
                mean=x_mean.detach().cpu().numpy().astype(np.float32))
        print(f"[POD] POD basis saved to: {pod_basis_path}")


    # 掩码比例设置
    mask_ratio = (args.mask_ratio_min, args.mask_ratio_max) if (args.mask_ratio_min>0 and args.mask_ratio_max>0) else args.mask_ratio

    # 构建 Dataset / Loader（用固定索引）
    L = args.seq_len
    train_set = PickleCondDataset(
        arr=arr_train, seq_len=L,
        mask_ratio=mask_ratio,
        noise_std=0.0,same_mask_per_frame=True,
        sample_type=args.sample_type,
        indices=train_idx
    )

    test_set = PickleCondDataset(
        arr=arr_test, seq_len=L,
        mask_ratio=mask_ratio,
        noise_std=0.0,same_mask_per_frame=True,
        sample_type=args.sample_type,
        indices=test_idx
    )

    def collate_fn(batch):
        x1    = torch.stack([b["x1"] for b in batch], dim=0)      # (B,1,T,H,W)
        mask1 = torch.stack([b["mask1"] for b in batch], dim=0)   # (B,1,T,H,W)
        y_obs = torch.stack([b["y_obs"] for b in batch], dim=0)   # (B,1,T,H,W)
        return {"x1": x1, "mask1": mask1, "y_obs": y_obs}

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # 模型
    model = VideoDiT_FM_XAttn(
        video_size=(L, H, W),
        patch_size=(args.patch_t, args.patch_h, args.patch_w),
        in_channels=1,
        hidden_size=args.hidden, depth=args.depth, num_heads=args.heads,
        mlp_ratio=args.mlp_ratio, sensor_pos_dim_each=args.sensor_pos_dim,
        pod_rank=args.pod_rank
    ).to(device)

    # 记录模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_text("model/info", f"Total params: {total_params:,}, Trainable: {trainable_params:,}", 0)
    print(f"[Model] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # ===== 在这里添加调度器 =====
    num_steps = len(train_loader) * args.epochs
    # 使用5个epoch进行预热
    warmup_steps = len(train_loader) * 5 
    
    lr_scheduler = CosineLRScheduler(
        opt,
        t_initial=num_steps,
        lr_min=2e-5,
        warmup_t=warmup_steps,
        warmup_lr_init=2e-5,
        warmup_prefix=True
    )
    print(f"[Scheduler] CosineLRScheduler with {warmup_steps} warmup steps created.")


    # 断点续训
    start_epoch = 1
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        if ema and ck.get("ema"):
            ema.shadow = ck["ema"]
            # 确保 EMA 影子参数在正确设备上
            for k in ema.shadow:
                ema.shadow[k] = ema.shadow[k].to(device)
        start_epoch = ck.get("epoch", 0) + 1
        global_step = ck.get("global_step", 0)
        # ===== 4. 加载调度器状态 =====
        if ck.get("scheduler"):
            lr_scheduler.load_state_dict(ck["scheduler"])
        print(f"[Resume] {args.resume} @ epoch {start_epoch-1}")

    cfg = {k: (v if isinstance(v,(int,float,str,bool)) else str(v)) for k,v in vars(args).items()}
    with open(os.path.join(args.out_dir, "cfg.json"), "w") as f: json.dump(cfg, f, indent=2)

    # 训练
    import time
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        run_loss = 0.0
        epoch_start_time = time.time()
        
        for it, batch in enumerate(pbar, 1):
            global_step += 1
            x1    = batch["x1"].to(device)
            mask1 = batch["mask1"].to(device)
            y_obs = batch["y_obs"].to(device)

            # === 改动1：用 (y_obs, mask1) 得到 Xpod（不反传） ===
            with torch.no_grad():
                a_hat = pod_coeffs_from_obs_singleframe_robust(
                    y_obs=y_obs, mask1=mask1,
                    U=U, x_mean=x_mean,
                    lam=args.pod_lam, dynamic_r=args.pod_dynamic_r, alpha_ratio=args.pod_alpha
                )
                # === 改动2：从 Xpod 采样直线对 (x_t, u_t, t) ===
                x_t, u_t, t = sample_linear_pair(x1)

            M_max = args.sensor_cap if args.sensor_cap>0 else None
            opt.zero_grad(set_to_none=True)
            v_hat = model(x_t, t, y_obs, mask1, M_max=M_max, a_hat=a_hat)
            loss = F.mse_loss(v_hat, u_t)
            loss.backward()
            if args.grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            # ===== 在这里更新学习率 =====
            lr_scheduler.step(global_step)

            if ema: ema.update(model)

            run_loss += loss.item()
            
            # 记录详细的训练指标
            if global_step % args.tb_log_every == 0:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                writer.add_scalar("train/learning_rate", get_lr(opt), global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                
                # 记录梯度信息
                total_grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar("train/grad_norm", total_grad_norm, global_step)

            if it % args.log_interval == 0:
                pbar.set_postfix(loss=f"{run_loss/it:.4f}", lr=f"{get_lr(opt):.2e}")

        # 记录每个 epoch 的训练指标
        avg_train_loss = run_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)

        # 验证
        @torch.no_grad()
        def evaluate(loader):
            mdl = model
            backup = None
            if ema:
                backup = {k:v.detach().clone() for k,v in mdl.state_dict().items()}
                mdl.load_state_dict({**mdl.state_dict(), **ema.shadow}, strict=False)
            mdl.eval()
            losses = []
            for batch in loader:
                x1    = batch["x1"].to(device)
                mask1 = batch["mask1"].to(device)
                y_obs = batch["y_obs"].to(device)

                # === 改动1：用 (y_obs, mask1) 得到 a_hat（不反传） ===
                with torch.no_grad():
                    a_hat = pod_coeffs_from_obs_singleframe_robust(
                        y_obs=y_obs, mask1=mask1,
                        U=U, x_mean=x_mean,
                        lam=args.pod_lam, dynamic_r=args.pod_dynamic_r, alpha_ratio=args.pod_alpha
                    )
                    # === 改动2：从 x1 采样直线对 (x_t, u_t, t) ===
                    x_t, u_t, t = sample_linear_pair(x1)
                M_max = args.sensor_cap if args.sensor_cap>0 else None
                v_hat = mdl(x_t, t, y_obs, mask1, M_max=M_max, a_hat=a_hat)
                losses.append(F.mse_loss(v_hat, u_t).item())
            if ema and backup: mdl.load_state_dict(backup, strict=True)
            return float(np.mean(losses)) if losses else math.nan
        
        if epoch % 20 == 0:
            val_loss = evaluate(test_loader)
            writer.add_scalar("val/loss", val_loss, epoch)
            print(f"[VAL(test)] epoch {epoch}: fm_loss={val_loss:.6f}")

            # 记录 EMA decay 如果使用 EMA
            if ema:
                writer.add_scalar("train/ema_decay", args.ema_decay, epoch)
            
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                ckpt = os.path.join(args.out_dir, f"best.pt")
                # 在保存检查点时包含 global_step
                save_ckpt_with_step = lambda path, epoch, model, opt, ema, cfg, step, lr_scheduler: torch.save({
                    "epoch": epoch,
                    "global_step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "ema": (ema.shadow if ema is not None else None),
                    "cfg": cfg,
                    "scheduler": lr_scheduler.state_dict(), # <--- 新增

                }, path)
                save_ckpt_with_step(ckpt, epoch, model, opt, ema, cfg, global_step, lr_scheduler)
                print(f"[Save] {ckpt}")

            if epoch == 300:
                # 在第300个 epoch 结束时保存模型
                ckpt = os.path.join(args.out_dir, f"300epoch.pt")
                save_ckpt_with_step(ckpt, epoch, model, opt, ema, cfg, global_step, lr_scheduler)
                print(f"[Save] {ckpt}")
            
            if epoch == args.epochs:
                # 在最后一个 epoch 结束时保存模型
                ckpt = os.path.join(args.out_dir, f"final.pt")
                save_ckpt_with_step(ckpt, epoch, model, opt, ema, cfg, global_step, lr_scheduler)
                print(f"[Save] {ckpt}")

    # 关闭 TensorBoard writer
    writer.close()
    print(f"[TensorBoard] 日志已保存，可使用以下命令查看:")
    print(f"tensorboard --logdir {tb_log_dir}")
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()

