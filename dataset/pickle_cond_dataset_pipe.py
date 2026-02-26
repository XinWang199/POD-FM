# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Union

class PickleCondDataset(Dataset):
    """
    .pickle -> Cross-Attn 条件式数据
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
        mean: Optional[float] = None,
        std: Optional[float] = None,
        save_norm_path: Optional[str] = None,
        mask_ratio: Union[float, Tuple[float,float]] = 0.05,
        noise_std: float = 0.0,
        same_mask_per_frame: bool = True,
        # ROI
        roi_type: Optional[str] = None,            # None | "box" | "circle"
        roi_box: Optional[Tuple[int,int,int,int]] = None,
        roi_circle: Optional[Tuple[float,float,float]] = None,
        ratio_mode: str = "roi",                   # "roi" | "global"
        sample_type: str = "bernoulli",            # "bernoulli" | "exact_k"
        rng: Optional[np.random.Generator] = None,
        indices: Optional[np.ndarray] = None,      # 固定划分的样本索引
    ):
        assert arr.ndim in (3,4), "arr 需为 (N,H,W,1) 或 (N,H,W)"
        if arr.ndim == 3: arr = arr[..., None]
        self.arr = arr.astype(np.float32)
        self.N, self.H, self.W, self.C = self.arr.shape
        assert self.C == 1, f"C={self.C}, 该数据应为单通道"
        self.seq_len = int(seq_len)

        # 归一化（建议传入训练集统计量）
        self.mean = float(self.arr.mean()) if mean is None else float(mean)
        self.std  = float(self.arr.std())  if std  is None else float(std)
        self.arr  = (self.arr - self.mean) / (self.std + 1e-6)
        if save_norm_path:
            Path(os.path.dirname(save_norm_path) or ".").mkdir(parents=True, exist_ok=True)
            np.save(save_norm_path, np.array([self.mean, self.std], dtype=np.float32))

        # ROI
        self.roi_mask2d = self._build_roi_mask2d(self.H, self.W, roi_type, roi_box, roi_circle)
        self.roi_area = int(self.roi_mask2d.sum())
        if roi_type is not None and self.roi_area == 0:
            raise ValueError("ROI 面积为 0，请检查 roi_* 参数")

        # 监测点
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.same_mask_per_frame = same_mask_per_frame
        self.ratio_mode = ratio_mode
        self.sample_type = sample_type
        self.rng = rng if rng is not None else np.random.default_rng()

        # 样本索引
        if indices is None:
            base_total = self.N if self.seq_len == 1 else max(0, self.N - self.seq_len + 1)
            self.indices = np.arange(base_total, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self): return len(self.indices)

    @staticmethod
    def _build_roi_mask2d(H, W, roi_type, roi_box, roi_circle):
        if roi_type is None: return np.ones((H,W), bool)
        if roi_type == "box":
            assert roi_box is not None
            h0,h1,w0,w1 = roi_box
            h0 = max(0,min(H-1,h0)); h1 = max(0,min(H,h1))
            w0 = max(0,min(W-1,w0)); w1 = max(0,min(W,w1))
            m = np.zeros((H,W), bool); m[h0:h1, w0:w1] = True; return m
        if roi_type == "circle":
            assert roi_circle is not None
            cy,cx,r = roi_circle; yy,xx = np.ogrid[:H,:W]
            return ((yy-cy)**2 + (xx-cx)**2 <= r**2)
        raise ValueError(f"未知 roi_type: {roi_type}")

    def _sample_ratio(self) -> float:
        if isinstance(self.mask_ratio, (tuple, list)):
            lo, hi = self.mask_ratio; return float(self.rng.uniform(lo, hi))
        return float(self.mask_ratio)

    def _sample_mask2d(self, p: float) -> torch.Tensor:
        H,W = self.H, self.W; roi = self.roi_mask2d
        p_in = p * (H*W)/max(1,roi.sum()) if self.ratio_mode=="global" else p
        p_in = min(1.0, p_in)
        if self.sample_type == "bernoulli":
            r = self.rng.random((H,W)); m = (roi & (r < p_in)).astype(np.float32)
        else:
            K = int(round((p*H*W) if self.ratio_mode=="global" else (p*roi.sum())))
            K = max(0, min(K, int(roi.sum())))
            m = np.zeros((H,W), np.float32)
            if K>0:
                ys,xs = np.where(roi); sel = self.rng.choice(len(ys), size=K, replace=False)
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

        if self.noise_std>0:
            y_obs = mask1 * (x1 + torch.randn_like(x1)*self.noise_std)
        else:
            y_obs = mask1 * x1

        return {"x1": x1, "mask1": mask1, "y_obs": y_obs}