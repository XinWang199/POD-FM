import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Union, List

class PickleCondDataset(Dataset):
    """
    .pickle -> Cross-Attn 条件式数据 (SST 版本)
    - 采样区域: 仅在数据的有效非零区域（如海洋）进行。
    - 无 ROI 功能。

    支持 seq_len=1(2D) 或窗口 L>1。
    若传入 indices，则以 indices 作为样本列表：
      - seq_len=1: indices 是帧索引 [0..N-1]
      - seq_len>1: indices 是窗口起点 [0..N-L]
    输出:
      x1   : (C=1, T=L, H, W)
      mask1: (1,   T=L, H, W)
      y_obs: (C=1, T=L, H, W)
    """
    def __init__(
        self,
        arr: np.ndarray,
        seq_len: int = 1,
        mask_ratio: Union[float, Tuple[float,float]] = 0.05,
        noise_std: float = 0.0,
        same_mask_per_frame: bool = True,
        sample_type: str = "bernoulli",            # "bernoulli" | "exact_k"
        rng: Optional[np.random.Generator] = None,
        indices: Optional[np.ndarray] = None,      # 固定划分的样本索引
    ):
        assert arr.ndim in (3,4), "arr 需为 (N,H,W,1) 或 (N,H,W)"
        if arr.ndim == 3: arr = arr[..., None]

        # --- [新] 基于原始数据创建有效数据区域（海洋）的掩码 ---
        # 假设陆地值为0，海洋为非0。为避免浮点误差，使用一个小的阈值。
        # 我们通过计算时间维度上的平均值来获得一个稳定的陆地-海洋分布图。
        mean_map = np.mean(arr, axis=0).squeeze()
        self.valid_data_mask = np.abs(mean_map) > 1e-6  # True for ocean/valid data
        self.valid_area = int(self.valid_data_mask.sum())
        if self.valid_area == 0:
            raise ValueError("数据中没有有效的非零区域可供采样。")
        self.arr = arr.astype(np.float32)
    
        self.N, self.H, self.W, self.C = self.arr.shape
        assert self.C == 1, f"C={self.C}, 该数据应为单通道"
        self.seq_len = int(seq_len)

        # 监测点
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.same_mask_per_frame = same_mask_per_frame
        self.sample_type = sample_type
        self.rng = rng if rng is not None else np.random.default_rng()

        # 样本索引
        if indices is None:
            base_total = self.N if self.seq_len == 1 else max(0, self.N - self.seq_len + 1)
            self.indices = np.arange(base_total, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self): return len(self.indices)

    def _sample_ratio(self) -> float:
        if isinstance(self.mask_ratio, (tuple, list)):
            lo, hi = self.mask_ratio; return float(self.rng.uniform(lo, hi))
        return float(self.mask_ratio)

    def _sample_mask2d(self, p: float) -> torch.Tensor:
        H, W = self.H, self.W
        # 采样区域现在固定为有效数据区域（例如海洋）
        sampling_area_mask = self.valid_data_mask
        
        if self.sample_type == "bernoulli":
            # p 是在有效区域内每个点被选中的概率
            r = self.rng.random((H, W))
            m = (sampling_area_mask & (r < p)).astype(np.float32)
        else: # "exact_k"
            # p 是要采样的点数占有效区域总点数的比例
            K = int(round(p * self.valid_area))
            K = max(0, min(K, self.valid_area))
            m = np.zeros((H, W), np.float32)
            if K > 0:
                ys, xs = np.where(sampling_area_mask)
                sel = self.rng.choice(len(ys), size=K, replace=False)
                m[ys[sel], xs[sel]] = 1.0
                
        return torch.from_numpy(m)

    def __getitem__(self, idx):
        i0 = int(self.indices[idx])
        L = self.seq_len
        if L == 1:
            x = self.arr[i0][None, ...]        # (1,H,W,1)
        else:
            x = self.arr[i0:i0+L]              # (L,H,W,1)

        x1 = torch.from_numpy(x.transpose(3,0,1,2)).float()   # (1,L,H,W)
        p = self._sample_ratio()
        m2d = self._sample_mask2d(p)                          # (H,W)
        mask1 = m2d.unsqueeze(0).unsqueeze(0)                 # (1,1,H,W)
        if L > 1 and self.same_mask_per_frame:
            mask1 = mask1.expand(1, L, self.H, self.W).contiguous()

        if self.noise_std > 0:
            y_obs = mask1 * (x1 + torch.randn_like(x1) * self.noise_std)
        else:
            y_obs = mask1 * x1

        return {"x1": x1, "mask1": mask1, "y_obs": y_obs}