import os
import numpy as np
import torch
import torch.nn as nn
from model.videoDit import TimestepEmbedder, FinalAdaLNLayer, modulate, get_1d_sincos,get_3d_sincos_pos_embed
from model.videoDit import TubeletEmbed3D


class SensorTokenizer(nn.Module):
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
            m = mask1[b, 0]                    # (T,H,W)
            idx = m.nonzero(as_tuple=False)    # (Mb,3) 每行 (t,h,w) nonzero找到其中非零元素的位置索引
            Mb = idx.size(0)                   #获取非零元素的个数
            lengths.append(Mb)                 #记录每个样本非零元素数量

            if Mb == 0:
                # 至少保留一个占位 token，避免空序列
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


class DiTBlockXAttn(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0):
        super().__init__()
        D = hidden_size
        self.norm_q_ca = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.norm_kv_ca = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.cross = nn.MultiheadAttention(D, num_heads, dropout=attn_drop, batch_first=True)

        self.norm_ff1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp1 = nn.Sequential(nn.Linear(D, int(D*mlp_ratio)),
                                  nn.GELU(approximate="tanh"),
                                  nn.Linear(int(D*mlp_ratio), D))

        self.norm_sa = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(D, num_heads, dropout=attn_drop, batch_first=True)

        self.norm_ff2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp2 = nn.Sequential(nn.Linear(D, int(D*mlp_ratio)),
                                  nn.GELU(approximate="tanh"),
                                  nn.Linear(int(D*mlp_ratio), D))

        # 为 CA / FF1 / SA 各生成一套 (shift, scale, gate)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 9*D))

    def forward(self, x_tokens, cond_vec, cond_tokens, pad_mask):
        sh_ca, sc_ca, gt_ca, sh_f1, sc_f1, gt_f1, sh_sa, sc_sa, gt_sa = self.ada(cond_vec).chunk(9, dim=1)

        # 1) Cross-Attention: Q=视频 token, K/V=sensor token
        q = modulate(self.norm_q_ca(x_tokens), sh_ca, sc_ca)
        kv = self.norm_kv_ca(cond_tokens)
        ca = self.cross(q, kv, kv, key_padding_mask=pad_mask, need_weights=False)[0]
        x = x_tokens + gt_ca.unsqueeze(1) * ca

        # 2) FFN1
        x = x + gt_f1.unsqueeze(1) * self.mlp1(modulate(self.norm_ff1(x), sh_f1, sc_f1))

        # 3) Self-Attention
        qkv = modulate(self.norm_sa(x), sh_sa, sc_sa)
        sa = self.self_attn(qkv, qkv, qkv, need_weights=False)[0]
        x = x + gt_sa.unsqueeze(1) * sa

        # 4) FFN2
        x = x + self.mlp2(self.norm_ff2(x))
        return x


class VideoDiT_FM_XAttn(nn.Module):
    """
    条件化的时空 DiT：
      - input:
          x : (B, C, T, H, W)      # 当前状态 (训练时为 x_t；采样时为 x)
          t : (B,)                 # 连续时间 in [0,1]
          y_obs : (B, C, T, H, W)  # 观测值（未观测处为 0）
          mask1 : (B, 1, T, H, W)  # 0/1 掩码（每帧一致、通道共享时取单通道）
      - output:
          v_hat : (B, C, T, H, W)  # 速度场预测，与输入 x 同形状（用于 Flow Matching）
    """
    def __init__(self,
                 video_size=(10, 64, 64),
                 patch_size=(2, 4, 4),
                 in_channels=2,
                 hidden_size=384,
                 depth=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 sensor_pos_dim_each=32):
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

        # 2) condition：t + sensor token
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.sensor_tok = SensorTokenizer(hidden_size, pos_dim_each=sensor_pos_dim_each, in_ch_obs=in_channels)

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

    def forward(self, x, t, y_obs, mask1, M_max: int = None):
        """
        x:     (B,T,C,H,W) 或 (B,C,T,H,W)
        t:     (B,)
        y_obs: (B,C,T,H,W)   未观测处为 0
        mask1: (B,1,T,H,W)   0/1
        M_max: 传感器 token 对齐的最大长度（可不传，内部会取 batch 最大值）
        """
        # (B,C,T,H,W)
        if x.shape[1] != self.in_ch and x.shape[2] == self.in_ch:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        tokens, (Tm, Hm, Wm) = self.tubelet(x)                 # (B,L,D)
        assert (Tm, Hm, Wm) == (self.video_size[0]//self.patch_size[0],
                                self.video_size[1]//self.patch_size[1],
                                self.video_size[2]//self.patch_size[2]), \
               "输入视频尺寸需与初始化的 video_size 匹配"
        h = tokens + self.pos_embed.to(tokens.dtype).to(tokens.device)

        cond_tokens, pad_mask = self.sensor_tok(y_obs, mask1, M_max=M_max)  # (B,M,D), (B,M)
        c = self.t_embedder(t)                                              # (B,D)

        for blk in self.blocks:
            h = blk(h, c, cond_tokens, pad_mask)

        y = self.final(h, c)                          # (B,L,V*C)
        return self._unpatchify3d(y, x.shape)         # (B,C,T,H,W)

