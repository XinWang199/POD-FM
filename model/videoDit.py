import math
import torch
import torch.nn as nn
import torch.nn.functional as F
'''与原始dit基本一致，只是多加了个时间维度，而且不涉及条件插入训练，只接受传入的扩散步的t信息'''
# --- 与 DiT 一致的 modulate / t-embed / adaLN-Zero 思路 ---
def modulate(x, shift, scale):
    # 与 DiT 相同：x * (1 + scale) + shift
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """与 DiT 同构：正余弦频率嵌入 → MLP 到 hidden_size"""
    def __init__(self, hidden_size, freq_dim=256, max_period=10000):
        super().__init__()
        self.max_period = max_period
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):  # t: (B,)
        return self.mlp(self._timestep_embedding(t, self.freq_dim))

class AdaLNDiTBlock(nn.Module):
    """DiT 的 block：LayerNorm(无仿射) + MSA/MLP，经 adaLN-Zero 条件化并带门控"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # 生成 6×hidden 的 shift/scale/gate（MSA 与 MLP 各一套）
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        # 按 DiT 初始化习惯，建议在外部把这层权重置零

    def forward(self, x, cond):  # x:(B,L,D)  cond:(B,D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(6, dim=1)
        # MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            need_weights=False
        )[0]
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalAdaLNLayer(nn.Module):
    """DiT 的末层：adaLN 条件化 + 线性投影回 patch 像素"""
    def __init__(self, hidden_size, patch_volume, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj = nn.Linear(hidden_size, patch_volume * out_channels)  # V*C_out

    def forward(self, x, cond):  # x:(B,L,D)
        shift, scale = self.adaLN(cond).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.proj(x)  # (B,L,V*C_out)

# --- 3D Tubelet Embedding（Conv3d） ---
class TubeletEmbed3D(nn.Module):
    """把 (B,C,T,H,W) 打成 tubelet token: Conv3d(k=pt,ph,pw, s=pt,ph,pw) → (B,L,D)"""
    def __init__(self, in_ch, hidden_size, patch_size):
        super().__init__()
        pt, ph, pw = patch_size
        self.proj = nn.Conv3d(in_ch, hidden_size, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.patch_size = patch_size

    def forward(self, x):  # x:(B,C,T,H,W)
        x = self.proj(x)                        # (B,D,T',H',W')
        B, D, Tm, Hm, Wm = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, Tm*Hm*Wm, D)
        return x, (Tm, Hm, Wm)

# --- 3D 固定正余弦位置编码（T×H×W） ---
def get_1d_sincos(dim, length, device):
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, device=device, dtype=torch.float64)
    omega = 1. / (10000 ** (omega / (dim // 2)))
    pos = torch.arange(length, device=device, dtype=torch.float64)
    out = torch.einsum('m,d->md', pos, omega)  # (L, D/2)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (L, D)
    return emb.float()

def get_3d_sincos_pos_embed(embed_dim, t_size, h_size, w_size, device):
    # 把 D 平均分到 T/H/W 三个轴（每轴需要偶数维）：D % 6 == 0
    assert embed_dim % 6 == 0, "embed_dim 必须能被 6 整除（每轴正余弦各占 D/3）"
    dt = embed_dim // 3
    dh = embed_dim // 3
    dw = embed_dim - dt - dh
    # 每轴再平分成 sin/cos，要求偶数
    assert dt % 2 == 0 and dh % 2 == 0 and dw % 2 == 0
    emb_t = get_1d_sincos(dt, t_size, device)   # (T, dt)
    emb_h = get_1d_sincos(dh, h_size, device)   # (H, dh)
    emb_w = get_1d_sincos(dw, w_size, device)   # (W, dw)
    # 三轴外积 + 拼接 → (T,H,W,D)
    pos = (emb_t[:, None, None, :].expand(-1, h_size, w_size, -1),
           emb_h[None, :, None, :].expand(t_size, -1, w_size, -1),
           emb_w[None, None, :, :].expand(t_size, h_size, -1, -1))
    pos = torch.cat(pos, dim=-1).reshape(t_size * h_size * w_size, embed_dim)  # (L,D)
    return pos

# --- VideoDiT-FM 主体 ---
class VideoDiT_FM(nn.Module):
    """
    输入/输出: (B, C, T, H, W)
    Tubelet: patch_size=(pt,ph,pw)，token 长度 L=(T/pt)*(H/ph)*(W/pw)
    仅用 t 做条件：c = TimestepEmbedder(t)
    """
    def __init__(self,
                 video_size=(10, 64, 64),      # (T, H, W)
                 patch_size=(2, 4, 4),         # (pt, ph, pw) 要整除 video_size
                 in_channels=2,                # (u,v)
                 hidden_size=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4.0):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = in_channels
        self.video_size = video_size
        self.patch_size = patch_size
        self.embed_dim = hidden_size

        # Tubelet embedding
        self.tubelet = TubeletEmbed3D(in_channels, hidden_size, patch_size)

        # 固定 3D sin-cos 位置编码（与 DiT 一样固定，不参与训练）
        Tm = video_size[0] // patch_size[0]
        Hm = video_size[1] // patch_size[1]
        Wm = video_size[2] // patch_size[2]
        pos = get_3d_sincos_pos_embed(hidden_size, Tm, Hm, Wm, device=torch.device("cpu"))
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)  # (1,L,D)

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks (adaLN-Zero)
        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # 末层：把 token 还原成每个 tubelet 的像素值
        V = patch_size[0] * patch_size[1] * patch_size[2]
        self.final = FinalAdaLNLayer(hidden_size, patch_volume=V, out_channels=self.out_ch)

        self._init_weights_zerout()

    def _init_weights_zerout(self):
        # 与 DiT 做法一致：把 adaLN 的最后一层和输出层置零，确保初始为恒零场
        for m in self.blocks:
            nn.init.constant_(m.adaLN[-1].weight, 0.0)
            nn.init.constant_(m.adaLN[-1].bias, 0.0)
        nn.init.constant_(self.final.adaLN[-1].weight, 0.0)
        nn.init.constant_(self.final.adaLN[-1].bias, 0.0)
        nn.init.constant_(self.final.proj.weight, 0.0)
        nn.init.constant_(self.final.proj.bias, 0.0)

    def _unpatchify3d(self, y_token, out_shape):
        """
        y_token: (B, L, V*C),  V=pt*ph*pw
        out: (B, C, T, H, W)
        """
        B, L, VC = y_token.shape
        pt, ph, pw = self.patch_size
        T, H, W = self.video_size
        Ct = self.out_ch
        Tm, Hm, Wm = T // pt, H // ph, W // pw
        assert L == Tm * Hm * Wm and VC == Ct * pt * ph * pw

        y = y_token.view(B, Tm, Hm, Wm, Ct, pt, ph, pw)         # (B,T',H',W',C,pt,ph,pw)
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()      # (B,C,T',pt,H',ph,W',pw)
        y = y.view(B, Ct, Tm*pt, Hm*ph, Wm*pw)                  # (B,C,T,H,W)
        return y

    def forward(self, x, t):
        """
        x: (B, T, C, H, W) 或 (B, C, T, H, W)
        t: (B,) in [0,1]
        """
        if x.shape[1] != self.in_ch and x.shape[2] == self.in_ch:
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> (B,C,T,H,W)

        tokens, (Tm, Hm, Wm) = self.tubelet(x)        # (B,L,D)
        assert (Tm, Hm, Wm) == (self.video_size[0]//self.patch_size[0],
                                self.video_size[1]//self.patch_size[1],
                                self.video_size[2]//self.patch_size[2]), \
               "输入视频尺寸需与初始化的 video_size 匹配"

        h = tokens + self.pos_embed.to(tokens.dtype).to(tokens.device)  # 固定位置编码
        c = self.t_embedder(t)                                          # (B,D)

        for blk in self.blocks:
            h = blk(h, c)

        y = self.final(h, c)                      # (B,L,V*C)
        y = self._unpatchify3d(y, x.shape)        # (B,C,T,H,W)
        return y
