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
from model.videoDit_cross_attention import VideoDiT_FM_XAttn
from dataset.pickle_cond_dataset_cylinder import PickleCondDataset
from timm.scheduler import CosineLRScheduler

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
    ap.add_argument("--pickle_path", type=str, default='data/Cy_Taira.pickle')
    ap.add_argument("--first_n", type=int, default=5000, help="取前 N 帧（<=0 则全量）")
    ap.add_argument("--seq_len", type=int, default=1, help="=1 为 2D，>1 为窗口")
    ap.add_argument("--test_ratio", type=float, default=0.8, help="测试集比例")
    ap.add_argument("--split_seed", type=int, default=0, help="固定随机种子（固定划分）")
    ap.add_argument("--split_dir", type=str, default="", help="已存在的划分目录（含 train_idx.npy/test_idx.npy）")
    # 观测/ROI
    ap.add_argument("--mask_ratio", type=float, default=0.00)
    ap.add_argument("--mask_ratio_min", type=float, default=0.00037)
    ap.add_argument("--mask_ratio_max", type=float, default=0.0015)
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--roi_type", type=str, default="polygon", choices=["none","box","circle","polygon"])
    ap.add_argument("--roi_box", type=str, default="", help="h0,h1,w0,w1")
    ap.add_argument("--roi_circle", type=str, default="", help="cy,cx,r")
    ap.add_argument("--roi_polygon", type=str, default="[(36,25),(76,25),(106,192),(6,192)]",
                    help="[(y1,x1),(y2,x2),...]，至少3个点")
    ap.add_argument("--ratio_mode", type=str, default="global", choices=["roi","global"],help="就是计算采样比例时是相对于哪个区域")
    ap.add_argument("--sample_type", type=str, default="exact_k", choices=["bernoulli","exact_k"])
    # 训练
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", default=False,help="是否使用自动混合精度")
    ap.add_argument("--ema", default=True, help="是否使用 EMA")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--log_interval", type=int, default=10)

    # 模型
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp_ratio", type=float, default=1.0)
    ap.add_argument("--patch_t", type=int, default=1)
    ap.add_argument("--patch_h", type=int, default=4)
    ap.add_argument("--patch_w", type=int, default=4)
    ap.add_argument("--sensor_pos_dim", type=int, default=32)
    ap.add_argument("--sensor_cap", type=int, default=0)
    # 设备/保存
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_dir", type=str, default="runs/Cylinder")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--resume", type=str, default="")
    # TensorBoard 配置
    ap.add_argument("--tb_log_dir", type=str, default="", help="TensorBoard日志目录（默认为out_dir/tensorboard）")
    ap.add_argument("--tb_log_every", type=int, default=50, help="每N个batch记录一次详细指标")


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

    # 读取 .pickle
    with open(args.pickle_path, "rb") as f:
        arr = pickle.load(f)
    if isinstance(arr, np.ndarray) is False:
        raise ValueError(f"pickle 内容不是 numpy.ndarray: {type(arr)}")
    if arr.ndim == 3: arr = arr[..., None]  # (N,H,W,1)
    if args.first_n > 0: arr = arr[:args.first_n]
    N, H, W, _ = arr.shape
    print(f"[Data] arr: {arr.shape}, dtype={arr.dtype}, first_n={args.first_n}")

    # 计算"样本总数"（用于划分）：seq_len=1 -> N；seq_len>1 -> N-L+1
    L = args.seq_len
    base_total = N if L == 1 else max(0, N - L + 1)
    if base_total <= 1: raise ValueError("样本数量太少，请检查 seq_len 与数据大小")

    # 生成或加载固定划分索引
    split_dir = args.split_dir or os.path.join(args.out_dir, f"split_seq{L}_seed{args.split_seed}_r{args.test_ratio}")
    Path(split_dir).mkdir(parents=True, exist_ok=True)
    train_idx_path = os.path.join(split_dir, "train_idx.npy")
    test_idx_path  = os.path.join(split_dir, "test_idx.npy")

    if os.path.isfile(train_idx_path) and os.path.isfile(test_idx_path):
        train_idx = np.load(train_idx_path); test_idx = np.load(test_idx_path)
        print(f"[Split] loaded fixed split from {split_dir} (train={len(train_idx)}, test={len(test_idx)})")
    else:
        # 随机选取20%作为训练集
        rng = np.random.default_rng(args.split_seed)
        shuffled_idx = rng.permutation(base_total).astype(np.int64)
        n_train = int(round(0.2 * base_total))
        train_idx = shuffled_idx[:n_train]
        test_idx = shuffled_idx[n_train:]
        
        np.save(train_idx_path, train_idx); np.save(test_idx_path, test_idx)
        print(f"[Split] created random split (20% as train, seed={args.split_seed}): train={len(train_idx)}, test={len(test_idx)}")

    # 记录数据集信息到 TensorBoard
    writer.add_text("dataset/info", f"Total samples: {N}, Train: {len(train_idx)}, Test: {len(test_idx)}", 0)
    writer.add_text("dataset/shape", f"Data shape: {arr.shape}", 0)

    # 训练集统计量（仅用训练子集）
    # 注意：mean/std 在原始像素上统计；如果 seq_len>1，这里按帧统计等价
    tr_arr = arr if L == 1 else arr  # 同一个数组，统计不依赖窗口起点
    tr_mean = float(tr_arr.mean()); tr_std = float(tr_arr.std())
    np.save(os.path.join(args.out_dir, "flow_mean_std.npy"), np.array([tr_mean, tr_std], dtype=np.float32))
    print(f"[Stats] mean={tr_mean:.6f}, std={tr_std:.6f}")

    # 记录数据统计到 TensorBoard
    writer.add_scalar("dataset/mean", tr_mean, 0)
    writer.add_scalar("dataset/std", tr_std, 0)

    # 解析 ROI
    roi_type = None if args.roi_type == "none" else args.roi_type
    roi_box = tuple(map(int, args.roi_box.split(","))) if (roi_type=="box" and args.roi_box) else None
    roi_circle = tuple(map(float, args.roi_circle.split(","))) if (roi_type=="circle" and args.roi_circle) else None
    roi_polygon = eval(args.roi_polygon) if (roi_type=="polygon" and args.roi_polygon) else None

    # 掩码比例设置
    mask_ratio = (args.mask_ratio_min, args.mask_ratio_max) if (args.mask_ratio_min>0 and args.mask_ratio_max>0) else args.mask_ratio

    # 构建 Dataset / Loader（用固定索引）
    train_set = PickleCondDataset(
        arr, seq_len=L, mean=tr_mean, std=tr_std,
        save_norm_path=os.path.join(args.out_dir, "flow_mean_std.npy"),
        mask_ratio=mask_ratio, noise_std=args.noise_std, same_mask_per_frame=True,
        roi_type=roi_type, roi_box=roi_box, roi_circle=roi_circle, roi_polygon=roi_polygon,
        ratio_mode=args.ratio_mode, sample_type=args.sample_type,
        indices=train_idx
    )
    test_set = PickleCondDataset(
        arr, seq_len=L, mean=tr_mean, std=tr_std,
        save_norm_path=None,
        mask_ratio=mask_ratio, noise_std=args.noise_std, same_mask_per_frame=True,
        roi_type=roi_type, roi_box=roi_box, roi_circle=roi_circle, roi_polygon=roi_polygon,
        ratio_mode=args.ratio_mode, sample_type=args.sample_type,
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
                              num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # 模型
    model = VideoDiT_FM_XAttn(
        video_size=(L, H, W),
        patch_size=(args.patch_t, args.patch_h, args.patch_w),
        in_channels=1,
        hidden_size=args.hidden, depth=args.depth, num_heads=args.heads,
        mlp_ratio=args.mlp_ratio, sensor_pos_dim_each=args.sensor_pos_dim,
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
        lr_min=1e-6,
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6,
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

    # 训练 training loop
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

            with torch.no_grad():
                x_t, u_t, t = sample_linear_pair(x1)

            M_max = args.sensor_cap if args.sensor_cap>0 else None
            opt.zero_grad(set_to_none=True)
            v_hat = model(x_t, t, y_obs, mask1, M_max=M_max)
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

                with torch.no_grad():
                    x_t, u_t, t = sample_linear_pair(x1)

                M_max = args.sensor_cap if args.sensor_cap>0 else None
                v_hat = mdl(x_t, t, y_obs, mask1, M_max=M_max)
                losses.append(F.mse_loss(v_hat, u_t).item())
            if ema and backup: mdl.load_state_dict(backup, strict=True)
            return float(np.mean(losses)) if losses else math.nan
        
        if epoch % 10 == 0:
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
                if epoch == 800:
                    # 在第800个 epoch 结束时保存模型
                    ckpt = os.path.join(args.out_dir, f"800epoch.pt")
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

