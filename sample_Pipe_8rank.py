#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, math, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== 按你的工程路径修改这行 ====
from model.videoDit_cross_attention_podA_condition import VideoDiT_FM_XAttn
from dataset.pickle_cond_dataset_pipe import PickleCondDataset

# ---------------- 数据/工具 ----------------
def load_pickle(path, first_n):
    with open(path,"rb") as f: arr = pickle.load(f)
    if isinstance(arr, np.ndarray) is False:
        raise ValueError(f"pickle 内容不是 numpy.ndarray: {type(arr)}")
    if arr.ndim == 3: arr = arr[..., None]
    if first_n>0: arr = arr[:first_n]
    return arr.astype(np.float32)

def load_mean_std(norm_path, fallback_arr=None):
    if norm_path and os.path.isfile(norm_path):
        m,s = np.load(norm_path).tolist(); return float(m), float(s)
    if fallback_arr is not None:
        return float(fallback_arr.mean()), float(fallback_arr.std())
    return 0.0,1.0

@torch.no_grad()
def compute_metrics(x_gt, x_rec, mask, eps: float = 1e-12):
    """
    x_* : (B, C, T, H, W) ；mask : (B, 1, T, H, W) ，float型 0/1
    返回：
      mse_all, mse_obs, mse_unobs,
      relL2_all, relL2_obs, relL2_unobs,
      nrmse_range, nrmse_std,
      err_stats: dict containing mean, std, max, min of absolute errors
    """
    x_gt = x_gt.float(); x_rec = x_rec.float(); mask = mask.float()
    diff  = x_rec - x_gt
    diff2 = diff ** 2
    abs_err = diff.abs()

    mse_all = diff2.mean().item()
    obs_cnt    = mask.sum().clamp(min=1).item()
    unobs_cnt  = (1.0 - mask).sum().clamp(min=1).item()
    mse_obs    = (diff2 * mask).sum().item() / obs_cnt
    mse_unobs  = (diff2 * (1.0 - mask)).sum().item() / unobs_cnt

    def rel_l2(m):
        num = torch.sqrt((diff.pow(2) * m).sum(dim=(1, 2, 3, 4)))
        den = torch.sqrt(((x_gt * m).pow(2)).sum(dim=(1, 2, 3, 4))).add_(eps)
        return (num / den).mean().item()

    ones = torch.ones_like(mask)
    rel_all   = rel_l2(ones)
    rel_obs   = rel_l2(mask)
    rel_unobs = rel_l2(1.0 - mask)

    # === NRMSE 计算 ===
    rmse_all = math.sqrt(mse_all)
    # NRMSE (range normalization): RMSE / (max - min)
    gt_range = (x_gt.max() - x_gt.min()).item()
    nrmse_range = rmse_all / (gt_range + eps)
    # NRMSE (std normalization): RMSE / std(gt)
    gt_std = x_gt.std().item()
    nrmse_std = rmse_all / (gt_std + eps)

    # === 误差统计 ===
    err_stats = {
        "mean": abs_err.mean().item(),
        "std": abs_err.std().item(),
        "max": abs_err.max().item(),
        "min": abs_err.min().item(),
        "median": abs_err.median().item(),
        "p95": torch.quantile(abs_err.flatten(), 0.95).item(),  # 95th percentile
        "p99": torch.quantile(abs_err.flatten(), 0.99).item(),  # 99th percentile
    }

    return mse_all, mse_obs, mse_unobs, rel_all, rel_obs, rel_unobs, nrmse_range, nrmse_std, err_stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_compare_scalar(
    x_gt, x_rec, mask, t_idx, out_png,
    flow_cmap="RdBu_r", err_cmap="viridis",
    show_points=True,
    rel_cbar_max=0.5,        # 相对误差颜色棒上限
    abs_cbar_max=None
):
    """
    x_gt, x_rec: (B,1,T,H,W) ; mask: (B,1,T,H,W)
    """
    xg = x_gt[0].detach().cpu().numpy()     # (1,T,H,W)
    xr = x_rec[0].detach().cpu().numpy()
    mk = mask[0,0,t_idx].detach().cpu().numpy()

    gt = xg[0, t_idx]                       # (H,W)
    rc = xr[0, t_idx]

    abs_err = np.abs(rc - gt)
    rel_err = abs_err / (np.abs(gt) + 1e-12)

    vmax = np.max(np.abs(np.stack([gt, rc], axis=0)))
    norm_flow = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    if abs_cbar_max is None:
        vmax_abs = float(np.quantile(abs_err, 0.995))
    else:
        vmax_abs = float(abs_cbar_max)
    vmax_rel = float(max(rel_cbar_max, 1e-12))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    im0 = ax[0,0].imshow(gt, cmap=flow_cmap, norm=norm_flow)
    ax[0,0].set_title("GT"); ax[0,0].axis("off")
    fig.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04)

    im1 = ax[0,1].imshow(rc, cmap=flow_cmap, norm=norm_flow)
    ax[0,1].set_title("Recon"); ax[0,1].axis("off")
    fig.colorbar(im1, ax=ax[0,1], fraction=0.046, pad=0.04)

    im2 = ax[0,2].imshow(abs_err, cmap=err_cmap, vmin=0.0, vmax=vmax_abs)
    ax[0,2].set_title("|err|"); ax[0,2].axis("off")
    c2 = fig.colorbar(im2, ax=ax[0,2], fraction=0.046, pad=0.04)
    c2.set_label("abs error", rotation=270, labelpad=10)

    im3 = ax[1,0].imshow(gt, cmap=flow_cmap, norm=norm_flow)
    if show_points:
        ys, xs = np.where(mk > 0.5)
        ax[1,0].scatter(xs, ys, s=18, marker='*', facecolors='none',
                        edgecolors='k', linewidths=0.9)
    ax[1,0].set_title("GT + sensors"); ax[1,0].axis("off")
    fig.colorbar(im3, ax=ax[1,0], fraction=0.046, pad=0.04)

    im4 = ax[1,1].imshow(rel_err, cmap=err_cmap, vmin=0.0, vmax=vmax_rel)
    ax[1,1].set_title("Rel L2 map"); ax[1,1].axis("off")
    c4 = fig.colorbar(im4, ax=ax[1,1], fraction=0.046, pad=0.04)
    c4.set_label("relative L2", rotation=270, labelpad=10)

    ax[1,2].axis("off")

    fig.suptitle(f"Frame t={t_idx}", fontsize=12)
    fig.tight_layout()
    Path(os.path.dirname(out_png) or ".").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close(fig)

# ---------------- POD 辅助（与训练脚本一致） ----------------
def load_pod_npz(path):
    z = np.load(path)
    U = torch.from_numpy(z["U"]).float()        # (D, r)
    x_mean = torch.from_numpy(z["mean"]).float()# (D,)
    return U, x_mean

@torch.no_grad()
def pod_ls_framewise(y_obs: torch.Tensor, mask1: torch.Tensor,
                     U: torch.Tensor, x_mean: torch.Tensor,
                     lam: float = 1e-3, dynamic_r: bool = True, alpha_ratio: float = 0.4):
    """
    用 (y_obs, mask1) 做逐帧最小二乘，得到 POD 重建：
      return Xpod: (B,C,T,H,W)
    """
    B,C,T,H,W = y_obs.shape
    D,r_max = U.shape
    assert D == H*W*C
    dev, dt = y_obs.device, y_obs.dtype
    U = U.to(dev, dt); x_mean = x_mean.to(dev, dt)

    Xpod = torch.zeros(B,C,T,H,W, device=dev, dtype=dt)
    y_flat = y_obs.permute(0,2,3,4,1).reshape(B,T,H*W*C)
    m2d = mask1[:,0,0]
    I_full = torch.eye(r_max, device=dev, dtype=dt)

    for b in range(B):
        idx = (m2d[b].reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(1)  # (K,)
        Kb  = int(idx.numel())
        if Kb == 0:
            Xpod[b] = x_mean.view(H,W,C).permute(2,0,1)
            continue
        r_eff = r_max if not dynamic_r else max(1, min(r_max, int(alpha_ratio*Kb)))
        US = U[idx, :r_eff]
        L  = torch.linalg.cholesky(US.T @ US + lam * I_full[:r_eff,:r_eff])
        for t in range(T):
            yb  = y_flat[b, t, idx] - x_mean[idx]
            rhs = US.T @ yb
            a   = torch.cholesky_solve(rhs[:,None], L).squeeze(1)  # (r_eff,)
            x_p = (U[:, :r_eff] @ a) + x_mean                      # (D,)
            Xpod[b, :, t] = x_p.view(H, W, C).permute(2,0,1)
    return Xpod

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
# Flow-Matching 前向积分（Heun/Euler）+ 可选硬投影
@torch.no_grad()
def sample_with_conditions_Acoeff(model, y_obs, mask1, a_hat, size, steps=200, solver="heun",
                           hard_dc=False, sensor_cap=0, device="cuda"):
    B,C,T,H,W = size
    x = torch.randn(B,C,T,H,W, device=device)
    N = steps; dt = 1.0/N
    M_max = sensor_cap if (sensor_cap and sensor_cap>0) else None

    for k in range(N):
        t = torch.full((B,), k/N, device=device)
        v = model(x, t, y_obs, mask1, M_max=M_max, a_hat=a_hat)
        if solver == "euler":
            x = x + dt*v
        else:
            x_pred = x + dt*v
            t_next = torch.full((B,), min((k+1)/N,1.0), device=device)
            v_next = model(x_pred, t_next, y_obs, mask1, M_max=M_max, a_hat=a_hat)
            x = x + 0.5*dt*(v + v_next)
        if hard_dc:
            x = x*(1-mask1) + y_obs*mask1
    return x
# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # 数据/划分
    ap.add_argument("--pickle_path", type=str, default='data/ch_2Dxysec.pickle')
    ap.add_argument("--first_n", type=int, default=10000)
    ap.add_argument("--seq_len", type=int, default=1)

    # 观测生成
    ap.add_argument("--mask_ratio", type=float, default=0.0082, help="单帧 mask 比例")
    ap.add_argument("--mask_ratio_min", type=float, default=-1)
    ap.add_argument("--mask_ratio_max", type=float, default=-1)
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--roi_type", type=str, default="none", choices=["none","box","circle","polygon"])
    ap.add_argument("--roi_box", type=str, default="", help="h0,h1,w0,w1")
    ap.add_argument("--roi_circle", type=str, default="", help="cy,cx,r")
    ap.add_argument("--roi_polygon", type=str, default="[(36,25),(76,25),(106,192),(6,192)]",
                    help="[(y1,x1),(y2,x2),...]，至少3个点")
    ap.add_argument("--ratio_mode", type=str, default="global", choices=["roi","global"])
    ap.add_argument("--sample_type", type=str, default="exact_k", choices=["bernoulli","exact_k"])
    # 模型
    ap.add_argument("--ckpt", type=str, default='runs/Pipe_8rank_1e-1/final.pt')
    ap.add_argument("--use_ema", default=True, help="是否使用 EMA 权重")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=1.0)
    ap.add_argument("--patch_t", type=int, default=1)
    ap.add_argument("--patch_h", type=int, default=2)
    ap.add_argument("--patch_w", type=int, default=2)
    ap.add_argument("--sensor_pos_dim", type=int, default=32)
    ap.add_argument("--sensor_cap", type=int, default=0)
    # POD
    ap.add_argument("--pod_npz", type=str, default='runs/Pipe_8rank_1e-1/pod_basis_seq1_r8.npz', help="POD 基与均值 npz（训练端生成）")
    ap.add_argument("--pod_lam", type=float, default=1e-1)
    ap.add_argument("--pod_dynamic_r", default=False, help="是否动态调整有效秩 r_eff")
    ap.add_argument("--pod_alpha", type=float, default=1.0)
    # 采样
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", type=str, default="euler", choices=["heun","euler"])
    ap.add_argument("--hard_dc", default=False, help="是否每步做硬一致性投影")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit_batches", type=int, default=10)
    ap.add_argument("--save_n", type=int, default=0, help="保存前 N 个样本的可视化")
    ap.add_argument("--t_view", type=int, default=0)
    # 在现有的参数后添加
    ap.add_argument("--obs_noise_std", type=float, default=0.0, 
                help="观测数据的相对噪声标准差（相对于观测点标准差的比例）")
    # 其他
    ap.add_argument("--norm_path", type=str, default="runs/Pipe_8rank_1e-1/flow_mean_std.npy", help="训练时保存的 flow_mean_std.npy")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", type=str, default='runs/Pipe_8rank_1e-1')
    args = ap.parse_args()
    
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- 数据与划分 ----
    arr = load_pickle(args.pickle_path, args.first_n)    # (N,H,W,1)
    N,H,W,_ = arr.shape
    base_total = N if args.seq_len==1 else max(0, N - args.seq_len + 1)
    test_idx = np.arange(base_total, dtype=np.int64)

    # 统计量（与训练一致）
    if args.norm_path and os.path.isfile(args.norm_path):
        mean,std = load_mean_std(args.norm_path)
    else:
        # 若未传入，尽量用训练子集估计
        mean,std = float(arr[test_idx].mean()), float(arr[test_idx].std())
        print(f"[Warning] norm_path 不存在，使用训练子集估计：mean={mean:.6f}, std={std:.6f}")
    print(f"[Norm] mean={mean:.6f}, std={std:.6f}")

    # ROI 参数解析
    roi_type = None if args.roi_type == "none" else args.roi_type
    roi_box = tuple(map(int, args.roi_box.split(","))) if (roi_type=="box" and args.roi_box) else None
    roi_circle = tuple(map(float, args.roi_circle.split(","))) if (roi_type=="circle" and args.roi_circle) else None
    mask_ratio = (args.mask_ratio_min, args.mask_ratio_max) if (args.mask_ratio_min>0 and args.mask_ratio_max>0) else args.mask_ratio

    # Dataset / Loader（只用 test_idx）
    ds = PickleCondDataset(
        arr, seq_len=args.seq_len, mean=mean, std=std,
        mask_ratio=mask_ratio, noise_std=args.noise_std, same_mask_per_frame=True,
        roi_type=roi_type, roi_box=roi_box, roi_circle=roi_circle, 
        ratio_mode=args.ratio_mode, sample_type=args.sample_type,
        indices=test_idx
    )

    def collate_fn(b):
        x1    = torch.stack([x["x1"] for x in b], 0)   # (B,1,T,H,W)
        mask1 = torch.stack([x["mask1"] for x in b], 0)
        y_obs = torch.stack([x["y_obs"] for x in b], 0)
        return {"x1":x1, "mask1":mask1, "y_obs":y_obs}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # ---- 模型与权重 ----
    model = VideoDiT_FM_XAttn(
        video_size=(args.seq_len, H, W),
        patch_size=(args.patch_t, args.patch_h, args.patch_w),
        in_channels=1,
        hidden_size=args.hidden, depth=args.depth, num_heads=args.heads,
        mlp_ratio=args.mlp_ratio, sensor_pos_dim_each=args.sensor_pos_dim,
        pod_rank=8, # <-- pod_rank
    ).to(device).eval()

    # ---- POD 基/均值（与训练脚本一致）----
    assert os.path.isfile(args.pod_npz), f"POD 基文件不存在: {args.pod_npz}"
    U_npz, x_mean_npz = load_pod_npz(args.pod_npz)
    # 对齐 device/dtype
    dtype_model = next(model.parameters()).dtype
    U_npz      = U_npz.to(device, dtype_model)
    x_mean_npz = x_mean_npz.to(device, dtype_model)
    print(f"[POD] Load from npz: U {U_npz.shape}, x_mean {x_mean_npz.shape}")

    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    if args.use_ema and ck.get("ema") is not None:
        state = model.state_dict()
        ema = ck["ema"]
        for k in ema:
            if k in state and state[k].shape == ema[k].shape:
                state[k] = ema[k]
        model.load_state_dict(state, strict=False)
        print("[Info] Using EMA weights")

    # ---- [新] 评估主循环 ----
    noise_levels_to_test = [0.0, 0.1, 0.2, 0.3]
    original_out_dir = args.out_dir

    for noise_level in noise_levels_to_test:
        # 为当前噪声水平设置特定的输出目录
        args.obs_noise_std = noise_level
        current_out_dir = os.path.join(original_out_dir, f"noise_std_{noise_level:.1f}")
        Path(current_out_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n\n{'='*25}")
        print(f"  Running Evaluation for obs_noise_std = {args.obs_noise_std:.1f}")
        print(f"  Results will be saved to: {current_out_dir}")
        print(f"{'='*25}\n")

        # ---- 评估 ----
        all_mse_all, all_mse_obs, all_mse_unobs = [], [], []
        all_rel_all, all_rel_obs, all_rel_unobs = [], [], []
        all_nrmse_range, all_nrmse_std = [], []
        all_err_mean, all_err_std, all_err_max, all_err_p95, all_err_p99 = [], [], [], [], []
        saved = 0
        limit = args.limit_batches if args.limit_batches>0 else math.inf

        st_run = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(loader, 1):
                if bi>limit: break
                x1    = batch["x1"].to(device)    # (B,1,T,H,W)
                mask1 = batch["mask1"].to(device)
                y_obs = batch["y_obs"].to(device)
        
            # [NEW] 添加相对噪声到观测数据
                if args.obs_noise_std > 0:
                    mask_bool = (mask1 > 0.5)
                    if mask_bool.any():
                        # 计算观测点的标准差
                        y_obs_std = y_obs[mask_bool].std()
                        # 生成相对噪声
                        noise = torch.randn_like(y_obs) * (args.obs_noise_std * y_obs_std)
                        # 只在观测点添加噪声
                        y_obs = y_obs + noise * mask1
                        print(f"[Info] Added relative noise: std_ratio={args.obs_noise_std:.3f}, y_obs_std={y_obs_std:.6f}")

                st = time.time()
                a_hat = pod_coeffs_from_obs_singleframe_robust(
                        y_obs=y_obs, mask1=mask1,
                        U=U_npz, x_mean=x_mean_npz,
                        lam=args.pod_lam, dynamic_r=args.pod_dynamic_r, alpha_ratio=args.pod_alpha
                    )
                x_rec = sample_with_conditions_Acoeff(
                    model, y_obs, mask1, a_hat, size=x1.shape, steps=args.steps,
                    solver=args.solver, hard_dc=args.hard_dc,
                    sensor_cap=args.sensor_cap, device=device,
                    )

                dur = time.time() - st

                mse_all, mse_obs, mse_unobs, r_all, r_obs, r_unobs, nrmse_range, nrmse_std, err_stats = compute_metrics(x1, x_rec, mask1)
                all_mse_all.append(mse_all); all_mse_obs.append(mse_obs); all_mse_unobs.append(mse_unobs)
                all_rel_all.append(r_all); all_rel_obs.append(r_obs); all_rel_unobs.append(r_unobs)
                all_nrmse_range.append(nrmse_range); all_nrmse_std.append(nrmse_std)
                all_err_mean.append(err_stats["mean"]); all_err_std.append(err_stats["std"])
                all_err_max.append(err_stats["max"]); all_err_p95.append(err_stats["p95"]); all_err_p99.append(err_stats["p99"])
                
                print(f"[Batch {bi}] time={dur:.3f}s  "
                    f"MSE_all={mse_all:.6f}  MSE_obs={mse_obs:.6f}  MSE_unobs={mse_unobs:.6f}  "
                    f"| RelL2 all/obs/unobs = {r_all:.4f}/{r_obs:.4f}/{r_unobs:.4f}")
                print(f"           NRMSE(range)={nrmse_range:.4f}  NRMSE(std)={nrmse_std:.4f}  "
                    f"| AbsErr: mean={err_stats['mean']:.4f} ± {err_stats['std']:.4f}, max={err_stats['max']:.4f}, p95={err_stats['p95']:.4f}")
                
                # 保存可视化
                while saved < args.save_n and saved < x1.size(0):
                    t_show = max(0, min(args.t_view, x1.size(2)-1))
                    out_png = os.path.join(args.out_dir, f"case{bi:03d}_{saved:02d}_t{t_show}.png")
                    plot_compare_scalar(x1[saved:saved+1], x_rec[saved:saved+1], mask1[saved:saved+1],
                                        t_idx=t_show, out_png=out_png, rel_cbar_max=0.5)
                    print(f"[Saved] {out_png}")
                    saved += 1

        def avg(x): return float(np.mean(x)) if len(x)>0 else float("nan")
        def std_dev(x): return float(np.std(x)) if len(x)>0 else float("nan")
        
        print(f"\n{'='*60}")
        print(f"[FINAL RESULTS for obs_noise_std = {args.obs_noise_std:.1f}]")
        print(f"{'='*60}")
        print(f"Config: steps={args.steps}, solver={args.solver}, hard_dc={args.hard_dc}")
        print(f"\n--- MSE Metrics ---")
        print(f"MSE_all:   {avg(all_mse_all):.6f} ± {std_dev(all_mse_all):.6f}")
        print(f"MSE_obs:   {avg(all_mse_obs):.6f} ± {std_dev(all_mse_obs):.6f}")
        print(f"MSE_unobs: {avg(all_mse_unobs):.6f} ± {std_dev(all_mse_unobs):.6f}")
        print(f"\n--- Relative L2 Metrics ---")
        print(f"RelL2_all:   {avg(all_rel_all):.4f} ± {std_dev(all_rel_all):.4f}")
        print(f"RelL2_obs:   {avg(all_rel_obs):.4f} ± {std_dev(all_rel_obs):.4f}")
        print(f"RelL2_unobs: {avg(all_rel_unobs):.4f} ± {std_dev(all_rel_unobs):.4f}")
        print(f"\n--- NRMSE Metrics ---")
        print(f"NRMSE(range): {avg(all_nrmse_range):.4f} ± {std_dev(all_nrmse_range):.4f}")
        print(f"NRMSE(std):   {avg(all_nrmse_std):.4f} ± {std_dev(all_nrmse_std):.4f}")
        print(f"\n--- Absolute Error Statistics ---")
        print(f"Mean AbsErr:  {avg(all_err_mean):.4f} ± {std_dev(all_err_mean):.4f}")
        print(f"Std  AbsErr:  {avg(all_err_std):.4f} ± {std_dev(all_err_std):.4f}")
        print(f"Max  AbsErr:  {avg(all_err_max):.4f} (worst batch: {max(all_err_max):.4f})")
        print(f"P95  AbsErr:  {avg(all_err_p95):.4f} ± {std_dev(all_err_p95):.4f}")
        print(f"P99  AbsErr:  {avg(all_err_p99):.4f} ± {std_dev(all_err_p99):.4f}")
        print(f"\nTotal time: {time.time()-st_run:.2f}s")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
