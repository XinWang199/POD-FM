import torch
import torch.nn as nn
import torch.nn.functional as F

# 与原工程保持一致的导入
from model.videoDit import (
    TimestepEmbedder, FinalAdaLNLayer, modulate,
    get_3d_sincos_pos_embed, TubeletEmbed3D, get_1d_sincos
)

@torch.no_grad()
def pod_coeffs_from_obs_singleframe(
    y_obs: torch.Tensor,              # (B,C,1,H,W) 或 (B,C,T,H,W) 其中 T=1
    mask1: torch.Tensor,              # (B,1,1,H,W) 或 (B,1,T,H,W) 其中 T=1
    U: torch.Tensor,                  # (D, r_max)  列正交/已归一
    x_mean: torch.Tensor,             # (D,)
    lam: float = 1e-3,
    dynamic_r: bool = True,
    alpha_ratio: float = 0.4,
) -> torch.Tensor:

    if y_obs.dim() != 5 or mask1.dim() != 5:
        raise ValueError("y_obs 需为 (B,C,T,H,W)，mask1 需为 (B,1,T,H,W)")
    B, C, T, H, W = y_obs.shape
    if T != 1:
        raise ValueError(f"该函数面向单帧样本，检测到 T={T}，请确保 T=1 或改用多帧版本")

    D, r_max = U.shape
    assert D == H * W * C, f"U 第一维 D={D} 必须等于 H*W*C={H*W*C}"

    dev, dt = y_obs.device, y_obs.dtype
    U = U.to(dev, dt)
    x_mean = x_mean.to(dev, dt)

    # 展平单帧： (B,C,1,H,W) -> (B, D)
    y_flat = y_obs[:, :, 0].permute(0, 2, 3, 1).reshape(B, D)   # (B,D)
    # 首帧 2D mask
    m2d = mask1[:, 0, 0]                                        # (B,H,W)

    A = torch.zeros(B, r_max, device=dev, dtype=dt)
    for b in range(B):
        idx = (m2d[b].reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(1)  # (K,)
        Kb  = int(idx.numel())
        if Kb == 0:
            A[b].zero_()
            continue
        r_eff = r_max if not dynamic_r else max(1, min(r_max, int(alpha_ratio * Kb)))
        US = U.index_select(0, idx)[:, :r_eff]                 # (K, r_eff)
        L = torch.linalg.cholesky(US.T @ US + lam * torch.eye(r_eff, device=dev, dtype=dt))
        yb  = y_flat[b, idx] - x_mean[idx]                     # (K,)
        rhs = US.T @ yb                                        # (r_eff,)
        a   = torch.cholesky_solve(rhs[:, None], L).squeeze(1) # (r_eff,)
        A[b, :r_eff] = a
    return A  # (B, r_max)


# === 传感器 tokenizer，把 (y_obs, mask) -> 条件 token（保持） ===
class SensorTokenizer(nn.Module):
    """
    将稀疏观测 y_obs 与其 3D 位置编码为一串条件 token。
    - y_obs:  (B, C, T, H, W)  未观测处为 0
    - mask1:  (B, 1, T, H, W)  0/1（通道共享时传单通道即可）
    return:
      cond_tokens: (B, M_max, D)
      pad_mask:    (B, M_max)   True 表示 padding（供 MHA 的 key_padding_mask 使用）
    """
    def __init__(self, hidden_size: int, pos_dim_each: int = 32, in_ch_obs: int = 2):
        super().__init__()
        self.pos_dim_each = pos_dim_each
        self.in_ch_obs = in_ch_obs
        self.proj = nn.Linear(in_ch_obs + 3 * pos_dim_each, hidden_size)

    def _axis_pos_embed(self, L: int, dim: int, device: torch.device):
        return get_1d_sincos(dim, L, device)  # (L, dim), float32

    def forward(self, y_obs: torch.Tensor, mask1: torch.Tensor, M_max: int = None):
        B, C, T, H, W = y_obs.shape
        device = y_obs.device

        et = self._axis_pos_embed(T, self.pos_dim_each, device)  # (T,dt)
        eh = self._axis_pos_embed(H, self.pos_dim_each, device)  # (H,dh)
        ew = self._axis_pos_embed(W, self.pos_dim_each, device)  # (W,dw)

        feats, lengths = [], []
        for b in range(B):
            m = mask1[b, 0]
            idx = m.nonzero(as_tuple=False)
            Mb = idx.size(0)
            lengths.append(Mb)

            if Mb == 0:
                feats.append(torch.zeros(1, self.in_ch_obs + 3*self.pos_dim_each, device=device))
                continue

            t_idx, h_idx, w_idx = idx[:, 0], idx[:, 1], idx[:, 2]
            vals = y_obs[b, :, t_idx, h_idx, w_idx].T         # (Mb, C)
            pt = et[t_idx]                                     # (Mb, dt)
            ph = eh[h_idx]                                     # (Mb, dh)
            pw = ew[w_idx]                                     # (Mb, dw)
            feat = torch.cat([vals, pt, ph, pw], dim=1)        # (Mb, C+dt+dh+dw)
            feats.append(feat)

        if M_max is None:
            M_max = max(max(lengths), 1)

        tokens = []
        pad_mask = torch.ones(B, M_max, dtype=torch.bool, device=device)  # True=padding
        for b, feat in enumerate(feats):
            Mb = feat.size(0)
            pad = torch.zeros(max(M_max - Mb, 0), feat.size(1), device=device)
            tok = torch.cat([feat, pad], dim=0)[:M_max]  # (M_max, D_in)
            tokens.append(tok)
            pad_mask[b, :Mb] = False

        tokens = torch.stack(tokens, dim=0)              # (B, M_max, D_in)
        cond_tokens = self.proj(tokens)                  # (B, M_max, D)
        return cond_tokens, pad_mask


# === 带 Cross-Attention 的 DiT Block（保持） ===
class DiTBlockXAttn(nn.Module):
    """
    Cross-Attn -> FFN -> Self-Attn -> FFN
    - cond_vec: (B, D)  用于生成本 block 的 shift/scale/gate（零初始化）
    - cond_tokens: (B, M, D)  条件序列（传感器 token）
    - pad_mask: (B, M)  True=padding
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0):
        super().__init__()
        D = hidden_size
        self.norm_q_ca = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.norm_kv_ca = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.cross = nn.MultiheadAttention(D, num_heads, dropout=attn_drop, batch_first=True)

        self.norm_ff1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp1 = nn.Sequential(
            nn.Linear(D, int(D*mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(D*mlp_ratio), D)
        )

        self.norm_sa = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(D, num_heads, dropout=attn_drop, batch_first=True)

        self.norm_ff2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp2 = nn.Sequential(
            nn.Linear(D, int(D*mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(D*mlp_ratio), D)
        )

        # CA / FF1 / SA 各 (shift, scale, gate)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 9*D))

    def forward(self, x_tokens, cond_vec, cond_tokens, pad_mask):
        sh_ca, sc_ca, gt_ca, sh_f1, sc_f1, gt_f1, sh_sa, sc_sa, gt_sa = self.ada(cond_vec).chunk(9, dim=1)
        q = modulate(self.norm_q_ca(x_tokens), sh_ca, sc_ca)
        kv = self.norm_kv_ca(cond_tokens)
        ca = self.cross(q, kv, kv, key_padding_mask=pad_mask, need_weights=False)[0]
        x = x_tokens + gt_ca.unsqueeze(1) * ca
        x = x + gt_f1.unsqueeze(1) * self.mlp1(modulate(self.norm_ff1(x), sh_f1, sc_f1))
        qkv = modulate(self.norm_sa(x), sh_sa, sc_sa)
        sa = self.self_attn(qkv, qkv, qkv, need_weights=False)[0]
        x = x + gt_sa.unsqueeze(1) * sa
        x = x + self.mlp2(self.norm_ff2(x))
        return x


class VideoDiT_FM_XAttn(nn.Module):

    def __init__(self,
                 video_size=(1, 64, 64),
                 patch_size=(1, 4, 4),
                 in_channels=1,
                 hidden_size=384,
                 depth=8,
                 num_heads=8,
                 mlp_ratio=1.0,
                 sensor_pos_dim_each=32,
                 pod_rank=0, 
                 ):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = in_channels
        self.video_size = video_size
        self.patch_size = patch_size
        self.embed_dim = hidden_size

        # 1) tubelet + positional embedding
        self.tubelet = TubeletEmbed3D(in_channels, hidden_size, patch_size)
        Tm = video_size[0] // patch_size[0]
        Hm = video_size[1] // patch_size[1]
        Wm = video_size[2] // patch_size[2]
        pos = get_3d_sincos_pos_embed(hidden_size, Tm, Hm, Wm, device=torch.device("cpu"))
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)  # (1,L,D)
        self.grid_size = (Tm, Hm, Wm)

        # 2) 条件路径（保持 y_obs Cross-Attn）
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.sensor_tok = SensorTokenizer(hidden_size, pos_dim_each=sensor_pos_dim_each, in_ch_obs=in_channels)

        self.pod_rank = pod_rank
        if self.pod_rank > 0:
            self.a_proj = nn.Linear(self.pod_rank, hidden_size)
        else:
            self.a_proj = None

        # 3) Transformer blocks
        self.blocks = nn.ModuleList([DiTBlockXAttn(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])

        # 4) end head
        V = patch_size[0] * patch_size[1] * patch_size[2]
        self.final = FinalAdaLNLayer(hidden_size, patch_volume=V, out_channels=self.out_ch)

        self._init_ada_zero()

    def _init_ada_zero(self):
        for m in self.blocks:
            nn.init.constant_(m.ada[-1].weight, 0.0)
            nn.init.constant_(m.ada[-1].bias, 0.0)
        nn.init.constant_(self.final.adaLN[-1].weight, 0.0)
        nn.init.constant_(self.final.adaLN[-1].bias, 0.0)
        nn.init.constant_(self.final.proj.weight, 0.0)
        nn.init.constant_(self.final.proj.bias, 0.0)

    def _unpatchify3d(self, y_token, out_shape):
        B, L, VC = y_token.shape
        pt, ph, pw = self.patch_size
        T, H, W = self.video_size
        Ct = self.out_ch
        Tm, Hm, Wm = T // pt, H // ph, W // pw
        assert L == Tm * Hm * Wm and VC == Ct * pt * ph * pw
        y = y_token.view(B, Tm, Hm, Wm, Ct, pt, ph, pw)
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        y = y.view(B, Ct, Tm*pt, Hm*ph, Wm*pw)
        return y

    def forward(self,
                x: torch.Tensor,               # (B,C,T,H,W)
                t: torch.Tensor,               # (B,)
                y_obs: torch.Tensor,           # (B,C,T,H,W)
                mask1: torch.Tensor,           # (B,1,T,H,W)
                M_max: int | None = None,
                a_hat: torch.Tensor | None = None  # [NEW] 外部 POD 系数 (B,r)
                ) -> torch.Tensor:
        # (B,C,T,H,W)
        if x.shape[1] != self.in_ch and x.shape[2] == self.in_ch:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        tokens, grid = self.tubelet(x)  # (B,L,D)
        assert grid == self.grid_size, "输入视频尺寸需与初始化的 video_size 匹配"
        h = tokens + self.pos_embed.to(tokens.dtype).to(tokens.device)

        c = self.t_embedder(t)  # (B,D)

        cond_tokens, pad_mask = self.sensor_tok(y_obs, mask1, M_max=M_max)   # (B,M,D), (B,M)

        if a_hat is not None:
            if self.a_proj is None:
                raise ValueError(
                    "Model received a_hat, but it was initialized without a `pod_rank` > 0. "
                    "Please provide `pod_rank` during model creation."
                )
            a_tok = self.a_proj(a_hat.to(cond_tokens.dtype)).unsqueeze(1)    # (B,1,D)
            cond_tokens = torch.cat([cond_tokens, a_tok], dim=1)             # (B,M+1,D)
            a_mask = torch.zeros(cond_tokens.size(0), 1, dtype=torch.bool, device=cond_tokens.device)  # False=非padding
            pad_mask = torch.cat([pad_mask, a_mask], dim=1)                  # (B,M+1)

        # --- Transformer blocks ---
        for blk in self.blocks:
            h = blk(h, c, cond_tokens, pad_mask)

        # --- Head ---
        y = self.final(h, c)
        return self._unpatchify3d(y, x.shape)
