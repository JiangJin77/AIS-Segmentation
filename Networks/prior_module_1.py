"""
2026-07-27    尝试先验引导的另一种方案:Adaptive Multi-scale Prior Generator

    Initial Prior Estimator（Threshold） → 输出 prior_map
    Adaptive Prior Refinement（Multi-scale Diffusion） → 输出 refined_prior

架构：
                                NCCT (B,1,H,W)
                                     │
                                     ▼
                            Learnable Threshold  ─── 可微分软阈值 (lower/upper)
                                     │
                                     ▼
                            initial prior map (B,1,H,W)
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    │                         CT + initial prior map →(B,2,H,W)                   
                    │                                 │
                    │                                 ▼
                    │                    SpatialAdaptiveScaleAttention (B, 3, H, W) 逐像素权重
                    ▼                                 │
        Conv3×3  Conv5×5  Conv7×7                     │
            │       │       │                         │
            └───────┼───────┘                         │
                    ▼                                 │ 
        Spatial Multi-scale Fusion   ◄────────────────┘
                    │
                    ▼
            Refined Prior Map (B, 1, H, W)
                    │
                    ▼
            Prior Normalization
                    │
                    ▼
                prior map (B, 1, H, W)
                    │
                    ▼
                Concat with CT (B, 2, H, W)
                    │
                    ▼
                2D SegNet

关键设计:
  1. 多尺度卷积核 (3×3/5×5/7×7) 各自可学习，捕获不同尺度的空间扩散模式
  2. SpatialAdaptiveScaleAttention 从 CT+initial prior map 逐像素预测融合权重 —
     腔隙灶区域 → 3×3 权重高；大面积梗死区域 → 7×7 权重高
     同一张切片的不同空间位置可以用不同的扩散尺度
  3. 融合权重是逐像素自适应的，而非固定公式 exp(-α·r^β)

用法:
    prior_module = LearnablePrior()
    prior_map = prior_module(image)           # (B, 1, H, W) 仅先验图
    x_cat = torch.cat([image, prior_map], 1)  # (B, 2, H, W) 拼接后送入分割网络
"""

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Differentiable Threshold -可微分阈值分割
# ═══════════════════════════════════════════════════════════════════════════════

class DifferentiableThreshold(nn.Module):
    """可微软阈值层
    结构:
        输入 (B, 1, H, W)
        └── σ((x - lower)/τ) · σ((upper - x)/τ)

    | 软阈值函数 | 软阈值掩码 |
    | 温度 τ → 0 | 趋近硬阈值（阶梯函数），梯度集中在边界 |
    | 温度 τ → ∞ | 趋近线性函数，梯度分布在整个值域 |
    | 默认 τ=2.5 | τ ≈ 0.56×(upper-lower)，中心响应 σ(0.9)² ≈ 0.51 |

    Args:
        init_lower: lower 初始值，默认 17.5 HU
        init_upper: upper 初始值，默认 22.0 HU
        init_tau:   软阈值温度初始值，默认 2.5。通过 softplus 约束 τ>0
    """

    def __init__(self, init_lower: float = 17.5, init_upper: float = 22.0, init_tau: float = 2.5):
        super().__init__()

        self.lower = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        # upper = lower + softplus(gap_raw) 保证 upper > lower 恒成立
        init_gap = max(init_upper - init_lower, 1.0)
        self.gap_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(init_gap) - 1), dtype=torch.float32))

        # τ = softplus(tau_raw)，确保 τ > 0 且可端到端训练
        self.tau_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(init_tau) - 1), dtype=torch.float32))

    @property
    def tau(self) -> Tensor:
        return F.softplus(self.tau_raw)

    @property
    def upper(self) -> Tensor:
        return self.lower + F.softplus(self.gap_raw)

    def forward(self, x: Tensor) -> Tensor:
        """软阈值函数

        Args:
            x: 输入 CT 图像 (B, 1, H, W)，HU 值

        Returns:
            软阈值掩码 (B, 1, H, W)，值域 [0, 1]
        """
        lower, upper, tau = self.lower, self.upper, self.tau
        return torch.sigmoid((x - lower) / tau) * torch.sigmoid((upper - x) / tau)

    def get_threshold(self) -> Tuple[float, float]:
        return self.lower.item(), self.upper.item()

    def extra_repr(self) -> str:
        tau_val = self.tau.detach().cpu().item()
        return f"lower={self.lower.item():.2f}, upper={self.upper.item():.2f}, tau={tau_val:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Scale Attention — 从 CT 原图预测多尺度融合权重
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialAdaptiveScaleAttention(nn.Module):
    """Spatial Adaptive Scale Attention

    逐像素预测各卷积核的融合权重，使同一张切片的不同空间位置可以使用
    不同的扩散尺度。输入 CT + Soft Mask 拼接后经 encoder-decoder 提取
    多尺度上下文，输出与原图同分辨率的逐像素 softmax 权重。

    CT 提供解剖上下文判断病灶尺度，Soft Mask 提供空间定位。

    结构:
        (CT, Mask) concat → (B, 2, H, W)
          → Conv 3×3, 16ch, s=1 → ReLU                     (H,   W)
          → Conv 3×3, 32ch, s=2 → ReLU                     (H/2, W/2)
          → Conv 3×3, 32ch, s=1 → ReLU                     (H/2, W/2)
          → bilinear upsample ×2                            (H,   W)
          → Conv 3×3, 16ch, s=1 → ReLU
          → Conv 1×1, 3ch
          → Spatial Softmax(dim=1) → (B, 3, H, W)

    参数量 ~7K，在小样本医学数据上安全。

    Args:
        ct_channels:  CT 输入通道数，默认 1
        num_scales:   多尺度数量，默认 3 (3×3, 5×5, 7×7)
        mid_ch:       中间特征通道数，默认 16
    """

    def __init__(self, ct_channels: int = 1, num_scales: int = 3, mid_ch: int = 16):
        super().__init__()
        self.num_scales = num_scales
        in_ch = ct_channels + 1  # CT + mask

        # ── Encoder: stride=2 一次获得 2× 感受野扩展 ──
        self.enc_conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(mid_ch, mid_ch * 2, 3, stride=2, padding=1)  # H/2
        self.enc_conv3 = nn.Conv2d(mid_ch * 2, mid_ch * 2, 3, stride=1, padding=1)

        # ── Decoder: 上采样回原分辨率 ──
        self.dec_conv1 = nn.Conv2d(mid_ch * 2, mid_ch, 3, stride=1, padding=1)

        # ── 输出头 ──
        self.out_conv = nn.Conv2d(mid_ch, num_scales, 1)

        # ── 可学习温度: softmax 锐度，初始 τ=1 ──
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

        # ── 初始化 ──
        for m in [self.enc_conv1, self.enc_conv2, self.enc_conv3, self.dec_conv1]:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        # 输出层: 零初始化 → softmax 初始 ≈ (⅓, ⅓, ⅓)
        nn.init.xavier_uniform_(self.out_conv.weight, gain=0.01)
        nn.init.zeros_(self.out_conv.bias)

    @property
    def temperature(self) -> Tensor:
        """softmax 温度 τ = 1 + softplus(tau_raw)，τ ∈ (1, ∞)"""
        return 1.0 + F.softplus(self.tau_raw)

    def forward(self, ct: Tensor, mask: Tensor) -> Tensor:
        """前向传播

        Args:
            ct:   CT 原图       (B, C, H, W)
            mask: 软阈值掩码    (B, 1, H, W)

        Returns:
            逐像素融合权重 (B, num_scales, H, W)，softmax 归一化，∑_k w_k[i,j] = 1
        """
        x = torch.cat([ct, mask], dim=1)            # (B, C+1, H, W)

        # Encoder
        e1 = F.relu(self.enc_conv1(x))              # (B, mid_ch,  H,   W)
        e2 = F.relu(self.enc_conv2(e1))             # (B, mid_ch*2, H/2, W/2)
        e3 = F.relu(self.enc_conv3(e2))             # (B, mid_ch*2, H/2, W/2)

        # Decoder
        d1 = F.interpolate(e3, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = F.relu(self.dec_conv1(d1))             # (B, mid_ch, H, W)

        # Output
        logits = self.out_conv(d1)                  # (B, num_scales, H, W)
        weights = F.softmax(logits / self.temperature, dim=1)
        return weights

    def extra_repr(self) -> str:
        tau = self.temperature.detach().cpu().item()
        return f"num_scales={self.num_scales}, temperature={tau:.3f}, params={sum(p.numel() for p in self.parameters())}"


# ═══════════════════════════════════════════════════════════════════════════════
# Learnable Diffusion — 多尺度卷积 + CT 自适应融合
# ═══════════════════════════════════════════════════════════════════════════════

class LearnableDiffusion(nn.Module):
    """可学习多尺度扩散层 + 逐像素空间自适应融合

    对软阈值掩码做多尺度可学习卷积扩散（3×3, 5×5, 7×7），
    再由 SpatialAdaptiveScaleAttention 根据 CT+Mask 逐像素预测融合权重。

    结构:
        mask (B, 1, H, W)
          ├──→ Conv3×3 ──→ f₁ (B,1,H,W) ──┐
          ├──→ Conv5×5 ──→ f₂ (B,1,H,W) ──┤──→ Σ wᵢ[i,j]·fᵢ[i,j]
          ├──→ Conv7×7 ──→ f₃ (B,1,H,W) ──┘       ↑
                                                     │
        CT ──→ SpatialAdaptiveScaleAttention ──→ (B, 3, H, W) 逐像素权重
        mask ↗

    Args:
        in_channels:  输入通道数，默认 1
        out_channels: 输出通道数，默认 1
        radii:        卷积核半径，默认 (1, 2, 3) → 3×3, 5×5, 7×7
        mid_ch:       Attention 中间通道数，默认 16
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        radii: Tuple[int, ...] = (1, 2, 3),
        mid_ch: int = 16,
    ):
        super().__init__()
        self.radii = radii
        self.num_scales = len(radii)

        # ── 多尺度可学习卷积核 ──
        self.kernels = nn.ModuleList()
        for r in radii:
            k_size = 2 * r + 1
            conv = nn.Conv2d(in_channels, out_channels, k_size, padding=r, bias=False)
            nn.init.constant_(conv.weight, 1.0 / (k_size * k_size))  # 均匀核初始化
            self.kernels.append(conv)

        # ── 逐像素空间自适应注意力 ──
        self.attention = SpatialAdaptiveScaleAttention(
            ct_channels=in_channels,
            num_scales=self.num_scales,
            mid_ch=mid_ch,
        )

        # ── 先验归一化：可学习全局增益 ──
        self.prior_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, mask: Tensor, ct: Tensor) -> Tensor:
        """多尺度扩散 + 逐像素空间自适应融合 + 归一化

        Args:
            mask: 软阈值掩码 (B, 1, H, W)
            ct:   CT 原图     (B, C, H, W)

        Returns:
            先验密度图 (B, 1, H, W)
        """
        # ① 多尺度扩散
        scale_outputs = torch.stack([conv(mask) for conv in self.kernels], dim=1)  # (B, N, 1, H, W)

        # ② 逐像素空间自适应融合权重
        weights = self.attention(ct, mask)  # (B, N, H, W)

        # ③ 逐像素加权求和: Σ_k w_k[i,j] · f_k[i,j]
        prior = (scale_outputs.squeeze(2) * weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ④ 先验归一化：逐样本匹配 CT 量级 + 可学习增益
        ct_std = ct.std(dim=(2, 3), keepdim=True)              # (B, C, 1, 1)
        p_std = prior.std(dim=(2, 3), keepdim=True)            # (B, 1, 1, 1)
        scale = ct_std / (p_std + 1e-6)
        prior = prior * scale.clamp(max=100.0)                  # 量级对齐
        prior = prior * self.prior_gain                         # 可学习增益

        return prior

    def get_scale_weights(self, mask: Tensor, ct: Tensor) -> Tensor:
        """返回各尺度的逐像素平均权重，用于日志"""
        with torch.no_grad():
            w = self.attention(ct, mask)  # (B, N, H, W)
            return w.mean(dim=(2, 3))      # (B, N)

    def extra_repr(self) -> str:
        sizes = ", ".join(f"{2*r+1}×{2*r+1}" for r in self.radii)
        gain = self.prior_gain.detach().cpu().item()
        return (f"kernels=[{sizes}], prior_gain={gain:.3f}, "
                f"params={sum(p.numel() for p in self.parameters())}")


# ═══════════════════════════════════════════════════════════════════════════════
# Learnable Prior — 顶层模块
# ═══════════════════════════════════════════════════════════════════════════════

class LearnablePrior(nn.Module):
    """可学习先验引导模块

    流程:
        输入 CT (B, 1, H, W)
          → 可微分软阈值 → Soft Mask
          → 多尺度卷积 + CT 自适应融合 → Prior
          → 先验密度图 (B, 1, H, W)

    Args:
        init_lower:  lower 初始值 (HU)，默认 17.5
        init_upper:  upper 初始值 (HU)，默认 22.0
        init_tau:    软阈值温度，默认 2.5
        radii:       多尺度卷积核半径，默认 (1, 2, 3)
        mid_ch:      Attention 中间通道数，默认 16
    """

    def __init__(
        self,
        init_lower: float = 17.5,
        init_upper: float = 22.0,
        init_tau: float = 2.5,
        radii: Tuple[int, ...] = (1, 2, 3),
        mid_ch: int = 16,
    ):
        super().__init__()

        self.threshold = DifferentiableThreshold(
            init_lower=init_lower,
            init_upper=init_upper,
            init_tau=init_tau,
        )
        self.diffusion = LearnableDiffusion(
            in_channels=1,
            out_channels=1,
            radii=radii,
            mid_ch=mid_ch,
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入 CT 图像 (B, 1, H, W)

        Returns:
            先验密度图 (B, 1, H, W)
        """
        mask = self.threshold(x)          # 软阈值掩码 (B, 1, H, W)
        prior = self.diffusion(mask, x)    # 多尺度扩散 + CT 自适应融合
        return prior

    def get_threshold(self) -> Tuple[float, float]:
        return self.threshold.get_threshold()


# ═══════════════════════════════════════════════════════════════════════════════
# Prior Guidance Wrapper — 即插即用包装器
# ═══════════════════════════════════════════════════════════════════════════════

class PriorGuidanceWrapper(nn.Module):
    """先验引导包装器 — 将 LearnablePrior 包裹在任何分割网络外

    即插即用: 在通道维拼接 CT 和先验图，送入分割网络。
    原网络只需保证第一层接受 2 通道输入。

    用法:
        base_model = CCFANet(resinc=2)
        model = PriorGuidanceWrapper(base_model)
        output = model(image)           # image: (B, 1, H, W)

    Args:
        base_model:   任意分割网络，需接受 (B, 2, H, W) 输入
        prior_kwargs: 传递给 LearnablePrior 的参数
    """

    def __init__(self, base_model: nn.Module, **prior_kwargs):
        super().__init__()
        self.base_model = base_model
        self.prior_module = LearnablePrior(**prior_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播: CT → 先验图 → 拼接 → 分割网络

        Args:
            x: 输入 CT 图像 (B, 1, H, W)

        Returns:
            分割网络的输出
        """
        prior = self.prior_module(x)                  # (B, 1, H, W)，已内置归一化
        x_cat = torch.cat([x, prior], dim=1)           # (B, 2, H, W)
        return self.base_model(x_cat)

    def get_threshold(self) -> Tuple[float, float]:
        return self.prior_module.get_threshold()


# ═══════════════════════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟 CT 值 [0, 50] HU 的输入
    x = torch.rand(8, 1, 256, 256).to(device) * 50
    print(f"输入范围: [{x.min().item():.2f}, {x.max().item():.2f}]")

    # ── 测试 LearnablePrior ──
    prior_mod = LearnablePrior().to(device)
    out = prior_mod(x)
    print(f"\nLearnablePrior:")
    print(f"  输入: {x.shape} → 输出: {out.shape}")
    print(f"  lower: {prior_mod.get_threshold()[0]:.2f}, upper: {prior_mod.get_threshold()[1]:.2f}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 查看各尺度权重（空间平均）
    with torch.no_grad():
        mask = prior_mod.threshold(x)
        avg_weights = prior_mod.diffusion.get_scale_weights(mask, x)
        print(f"  阈值层输出范围: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
        print(f"  逐像素权重空间平均 (3×3, 5×5, 7×7): {avg_weights[0].cpu().tolist()}")

        # 查看空间权重的分布
        spatial_w = prior_mod.diffusion.attention(x, mask)  # (B, 3, H, W)
        print(f"  spatial权重 shape: {spatial_w.shape}")
        print(f"  spatial权重各尺度范围: "
              f"w₁[{spatial_w[:,0].min().item():.3f},{spatial_w[:,0].max().item():.3f}], "
              f"w₂[{spatial_w[:,1].min().item():.3f},{spatial_w[:,1].max().item():.3f}], "
              f"w₃[{spatial_w[:,2].min().item():.3f},{spatial_w[:,2].max().item():.3f}]")
        print(f"  prior_gain: {prior_mod.diffusion.prior_gain.item():.3f}")

    # ── 测试 PriorGuidanceWrapper ──
    dummy_seg = nn.Conv2d(2, 3, kernel_size=1).to(device)
    wrapped = PriorGuidanceWrapper(dummy_seg).to(device)
    out2 = wrapped(x)
    print(f"\nPriorGuidanceWrapper:")
    print(f"  最终输出 shape: {out2.shape}")

    # ── 梯度检查 ──
    loss = out2.sum()
    loss.backward()
    print(f"\n梯度检查:")
    print(f"  threshold.lower.grad:        {wrapped.prior_module.threshold.lower.grad.item():.6f}")
    print(f"  threshold.gap_raw.grad:      {wrapped.prior_module.threshold.gap_raw.grad.item():.6f}")
    print(f"  threshold.tau_raw.grad:      {wrapped.prior_module.threshold.tau_raw.grad.item():.6f}")
    print(f"  diffusion.prior_gain.grad:   {wrapped.prior_module.diffusion.prior_gain.grad.item():.6f}")
    for i, k in enumerate(wrapped.prior_module.diffusion.kernels):
        print(f"  kernel_{i} ({2*wrapped.prior_module.diffusion.radii[i]+1}×{2*wrapped.prior_module.diffusion.radii[i]+1}).weight.grad: "
              f"mean={k.weight.grad.mean().item():.6f}")
    # Attention 梯度
    attn = wrapped.prior_module.diffusion.attention
    out_weight_grad = attn.out_conv.weight.grad
    print(f"  attention.out_conv.weight.grad mean: {out_weight_grad.mean().item():.6f}")
    print(f"  attention.tau_raw.grad: {attn.tau_raw.grad.item():.6f}")

    # ── 参数统计 ──
    total = sum(p.numel() for p in wrapped.parameters())
    prior_params = sum(p.numel() for p in wrapped.prior_module.parameters())
    attn_params = sum(p.numel() for p in attn.parameters())
    print(f"\n参数统计:")
    print(f"  总计:           {total:,}")
    print(f"  Prior 模块:     {prior_params:,}")
    print(f"    - Threshold:  {sum(p.numel() for p in prior_mod.threshold.parameters()):,}")
    print(f"    - Diffusion:  {sum(p.numel() for p in prior_mod.diffusion.parameters()):,}")
    print(f"      · Kernels:  {sum(p.numel() for p in prior_mod.diffusion.kernels.parameters()):,}")
    print(f"      · Attention:{attn_params:,}")
