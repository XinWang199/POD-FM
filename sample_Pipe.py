#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, math, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== 按你的工程路径修改这行 ====
from model.videoDit_cross_attention import VideoDiT_FM_XAttn
from dataset.pickle_cond_dataset_pipe import PickleCondDataset

# ---------------- 数据/工具 ----------------
def load_pickle(path, first_n):
    with open(path,"rb") as f: arr = pickle.load(f)
    if isinstance(arr, np.ndarray) is False:
        raise ValueError(f"pickle 内容不是 numpy.ndarray: {type(arr)}")
    if arr.ndim == 3: arr = arr[..., None]
    if first_n>0: arr = arr[:first_n]
    return arr.astype(np.float32)

def load_split(split_dir):
    tr = np.load(os.path.join(split_dir,"train_idx.npy"))
    te = np.load(os.path.join(split_dir,"test_idx.npy"))
    return tr.astype(np.int64), te.astype(np.int64)

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

def plot_compare_scalar_new(
    x_gt, x_rec, mask, t_idx, out_png,
    flow_cmap="coolwarm", err_cmap="viridis",
    show_points=True,
    rel_cbar_max=0.5,        # 相对误差颜色棒上限
    abs_cbar_max=None
):
    """
    x_gt, x_rec: (B,1,T,H,W) ; mask: (B,1,T,H,W)
    显示：GT, Recon+sensors, |err| 三个子图
    针对128*48尺寸图像优化的布局
    """
    xg = x_gt[0].detach().cpu().numpy()     # (1,T,H,W)
    xr = x_rec[0].detach().cpu().numpy()
    mk = mask[0,0,t_idx].detach().cpu().numpy()

    gt = xg[0, t_idx]                       # (H,W)
    rc = xr[0, t_idx]

    abs_err = np.abs(rc - gt)

    # 固定颜色棒范围
    gt_recon_vmin = -3
    gt_recon_vmax = 3
    err_vmin = 0.00
    err_vmax = 1.50

    # 创建1行3列的子图
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.8), squeeze=False,
                          gridspec_kw={'wspace': 0.4})

    # 先调整布局，为颜色棒预留空间
    plt.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.05, 
                       wspace=0.4)
    
    # 使用二维索引访问子图
    ax = ax[0]  # 转换为一维数组方便访问

    # GT (原始全局场)
    im_gt = ax[0].imshow(gt, cmap=flow_cmap, vmin=gt_recon_vmin, vmax=gt_recon_vmax)
    ax[0].set_title("Ground Truth", fontsize=10)
    ax[0].axis("off")

    # Recon + sensors
    im0 = ax[1].imshow(rc, cmap=flow_cmap, vmin=gt_recon_vmin, vmax=gt_recon_vmax)
    if show_points:
        ys, xs = np.where(mk > 0.5)
        ax[1].scatter(xs, ys, s=12, marker='*', facecolors='none',
                     edgecolors='k', linewidths=0.8)
    ax[1].set_title("Recon + sensors", fontsize=10)
    ax[1].axis("off")

    # |err|
    im1 = ax[2].imshow(abs_err, cmap=err_cmap, vmin=err_vmin, vmax=err_vmax)
    ax[2].set_title("|err|", fontsize=10)
    ax[2].axis("off")
    
    # 在布局调整后手动创建颜色棒
    # 为GT创建颜色棒
    pos_gt = ax[0].get_position()
    cbar_ax_gt = fig.add_axes([
        pos_gt.x1 + 0.01,        # x: 距离子图0.01
        pos_gt.y0,               # y: 底部对齐
        0.012,                   # width: 颜色棒宽度0.012
        pos_gt.height            # height: 与子图同高
    ])
    plt.colorbar(im_gt, cax=cbar_ax_gt)
    
    # 为Recon创建颜色棒
    pos_recon = ax[1].get_position()
    cbar_ax1 = fig.add_axes([
        pos_recon.x1 + 0.01,     # x: 距离子图0.01
        pos_recon.y0,            # y: 底部对齐
        0.012,                   # width: 颜色棒宽度0.012
        pos_recon.height         # height: 与子图同高
    ])
    plt.colorbar(im0, cax=cbar_ax1)
    
    # 为Error创建颜色棒
    pos_err = ax[2].get_position()
    cbar_ax2 = fig.add_axes([
        pos_err.x1 + 0.01,       # x: 距离子图0.01
        pos_err.y0,              # y: 底部对齐
        0.012,                   # width: 颜色棒宽度0.012
        pos_err.height           # height: 与子图同高
    ])
    plt.colorbar(im1, cax=cbar_ax2)
    
    Path(os.path.dirname(out_png) or ".").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

@torch.no_grad()
def sample_with_conditions(model, y_obs, mask1, size, steps=200, solver="heun",
                           hard_dc=False, sensor_cap=0, device="cuda"):
    B,C,T,H,W = size
    x = torch.randn(B,C,T,H,W, device=device)
    N = steps; dt = 1.0/N
    M_max = sensor_cap if (sensor_cap and sensor_cap>0) else None

    for k in range(N):
        t = torch.full((B,), k/N, device=device)
        v = model(x, t, y_obs, mask1, M_max=M_max)
        if solver == "euler":
            x = x + dt*v
        else:
            x_pred = x + dt*v
            t_next = torch.full((B,), min((k+1)/N,1.0), device=device)
            v_next = model(x_pred, t_next, y_obs, mask1, M_max=M_max)
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
    ap.add_argument("--mask_ratio", type=float, default=0.0162, help="单帧 mask 比例")
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
    ap.add_argument("--ckpt", type=str, default='runs/Pipe/final.pt')
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
    ap.add_argument("--norm_path", type=str, default="runs/Pipe/flow_mean_std.npy", help="训练时保存的 flow_mean_std.npy（推荐传入）")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", type=str, default='runs/Pipe')
    ap.add_argument("--fixed_sampling_points", type=str, default=None, 
            help="fixed_sampling_points_pipe_without_center.npy预生成的固定采样点文件路径")
    args = ap.parse_args()
    
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- 数据与划分 ----
    arr = load_pickle(args.pickle_path, args.first_n)    # (N,H,W,1)
    N,H,W,_ = arr.shape
    base_total = N if args.seq_len==1 else max(0, N - args.seq_len + 1)
    test_idx = np.arange(base_total, dtype=np.int64)
    assert test_idx.max() <= base_total, f"test_idx 超界：base_total={base_total}, max={test_idx.max()}"

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
    # 在 main() 函数中修改 Dataset 创建部分
    if args.fixed_sampling_points and os.path.exists(args.fixed_sampling_points):
        print(f"[INFO] Loading fixed sampling points from: {args.fixed_sampling_points}")
        fixed_points = np.load(args.fixed_sampling_points)
        
        # 创建自定义数据集，使用固定采样点
        class FixedSamplingDataset:
            def __init__(self, arr, fixed_masks, indices, mean, std):
                self.arr = arr
                self.fixed_masks = fixed_masks
                self.indices = indices
                self.mean = mean
                self.std = std
                
            def __len__(self):
                return min(len(self.indices), len(self.fixed_masks))
                
            def __getitem__(self, idx):
                data_idx = self.indices[idx]
                
                # 原始数据：(H,W,C) -> (C,T,H,W) 其中 T=1
                x_raw = self.arr[data_idx]  # (H,W,1)
                x_norm = (x_raw - self.mean) / self.std  # 归一化（与预生成时保持一致）
                x1 = torch.from_numpy(x_norm).permute(2,0,1).unsqueeze(1).float()  # (C,T,H,W)
                
                # 使用预生成的固定mask
                mask_data = self.fixed_masks[idx]
                if mask_data.ndim == 4:  # (1,T,H,W)
                    mask1 = torch.from_numpy(mask_data).float()
                elif mask_data.ndim == 3:  # (T,H,W) 或 (1,H,W)
                    if mask_data.shape[0] == 1:  # (1,H,W)
                        mask1 = torch.from_numpy(mask_data).unsqueeze(0).float()  # (1,1,H,W)
                    else:  # (T,H,W)
                        mask1 = torch.from_numpy(mask_data).unsqueeze(0).float()  # (1,T,H,W)
                else:  # (H,W)
                    mask1 = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
                
                # 重要：y_obs应该是归一化后的数据乘以mask
                # 这样保证与预生成采样点时的数据处理流程一致
                y_obs = x1 * mask1
                
                return {"x1": x1, "mask1": mask1, "y_obs": y_obs}

        ds = FixedSamplingDataset(arr, fixed_points, test_idx, mean, std)
    else:
        # 使用原始的随机采样
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
    ).to(device).eval()


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

                        # 为噪声生成设置固定种子（基于batch索引确保可复现）
                        noise_seed = 12345 + bi * 1000  # bi是当前batch索引
                        torch.manual_seed(noise_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(noise_seed)

                        # 生成相对噪声
                        noise = torch.randn_like(y_obs) * (args.obs_noise_std * y_obs_std)
                        # 只在观测点添加噪声
                        y_obs = y_obs + noise * mask1
                        print(f"[Info] Added relative noise: std_ratio={args.obs_noise_std:.3f}, y_obs_std={y_obs_std:.6f}")

                st = time.time()

                x_rec = sample_with_conditions(
                    model, y_obs, mask1, size=x1.shape, steps=args.steps,
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
                    out_png = os.path.join(args.out_dir, f"without_center_noise_std_{args.obs_noise_std}/case{bi:03d}_{saved:02d}_t{t_show}.png")
                    plot_compare_scalar_new(x1[saved:saved+1], x_rec[saved:saved+1], mask1[saved:saved+1],
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
