"""
可学习先验引导模块 (Learnable Prior Module)
  
架构：
  输入 (B, 1, H, W)
    └── DifferentiableThreshold   — 可微分软阈值层 (学习 lower, upper)
          └── LearnableDiffusion  — 多尺度可学习卷积扩散 (学习 3x3/5x5/7x7 核 + 尺度权重)
                └── 先验图 (B, 1, H, W)
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


class DifferentiableThreshold(nn.Module):
    """可微 软阈值层
    结构:
        输入 (B, 1, H, W)
        └── σ((x - lower)/τ) · σ((upper - x)/τ)

    | 软阈值函数 | 软阈值掩码 |
    | 温度 τ → 0 | 趋近硬阈值（阶梯函数），梯度集中在边界 |
    | 温度 τ → ∞ | 趋近线性函数，梯度分布在整个值域 |
    | 默认 τ=2.5 | τ ≈ 0.56×(upper-lower)，中心响应 σ(0.9)² ≈ 0.51（平衡梯度流与响应强度） |

    Args:
        init_lower: lower 初始值，默认 17.5
        init_upper: upper 初始值，默认 22.0
        init_tau: 软阈值温度初始值，默认 2.5。通过 softplus 约束 τ>0
    """

    def __init__(self, init_lower: float = 17.5, init_upper: float = 22.0, init_tau: float = 2.5):
        super().__init__()

        self.lower = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        # upper = lower + softplus(gap) 保证 upper > lower 恒成立
        init_gap = max(init_upper - init_lower, 1.0)    # 窗口宽度
        # softplus⁻¹(x) = ln(e^x - 1)，使 softplus(gap_raw) ≈ init_gap
        self.gap_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_gap) - 1), dtype=torch.float32))

        # τ = softplus(tau_raw)，确保 τ > 0 且可端到端训练
        self.tau_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_tau) - 1), dtype=torch.float32))

    @property
    def tau(self) -> Tensor:
        """τ = softplus(tau_raw)，确保 τ > 0"""
        return F.softplus(self.tau_raw)

    @property
    def upper(self) -> Tensor:
        """upper = lower + softplus(gap_raw)，保证 upper > lower"""
        return self.lower + F.softplus(self.gap_raw)

    def forward(self, x: Tensor) -> Tensor:
        """软阈值函数: σ((x - lower)/τ) · σ((upper - x)/τ)

        Args:
            x: 输入 CT 图像 (B, 1, H, W)

        Returns:
            软阈值掩码 (B, 1, H, W)，值域 [0, 1]
        """
        lower, upper, tau = self.lower, self.upper, self.tau
        return torch.sigmoid((x - lower) / tau) * torch.sigmoid((upper - x) / tau)

    def get_threshold(self) -> Tuple[float, float]:
        """返回当前 (lower, upper) 的值，用于日志记录"""
        return self.lower.item(), self.upper.item()

    def extra_repr(self) -> str:
        tau_val = self.tau.detach().cpu().item()
        return f"lower={self.lower.item():.2f}, upper={self.upper.item():.2f}, tau={tau_val:.3f}"


class LearnableDiffusion(nn.Module):
    """可学习多尺度扩散层
    对软阈值掩码进行多尺度卷积扩散，生成距离加权的先验密度图。
    每个尺度的卷积核和尺度权重都是可学习的，替代原方案中固定的均匀核 + e^(-α·r^β) 衰减。

    结构:
        输入 (B, 1, H, W)
            ├── 3x3 可学习卷积 → 权重 w₁
            ├── 5x5 可学习卷积 → 权重 w₂
            └── 7x7 可学习卷积 → 权重 w₃
                ↓ 加权求和
            先验密度图 (B, 1, H, W)

    初始化策略:
        卷积核初始化为均匀核 (每个元素 = 1/k²)，
        尺度权重初始化为 e^(-α·r^β)，沿用原项目的 α=0.75, β=0.25，

    Args:
        in_channels: 输入通道数，默认 1
        out_channels: 输出通道数，默认 1
        radii: 卷积核半径列表，默认 (1, 2, 3) 对应 (3x3, 5x5, 7x7)
        init_alpha: 权重衰减系数 α，用于初始化尺度权重
        init_beta: 权重衰减指数 β，用于初始化尺度权重
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        radii: Tuple[int, ...] = (1, 2, 3),
        init_alpha: float = 0.75,
        init_beta: float = 0.25,
    ):
        super().__init__()
        self.radii = radii

        # 可学习卷积核: 每个尺度独立，padding=r 保持空间尺寸不变
        kernels = []
        for r in radii:
            k_size = 2 * r + 1
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=r, bias=False)
            nn.init.constant_(conv.weight, 1.0 / (k_size * k_size))  # 均匀核初始化
            kernels.append(conv)
        self.kernels = nn.ModuleList(kernels)

        # 用 softplus 确保 alpha, beta > 0
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_alpha) - 1), dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_beta) - 1), dtype=torch.float32))

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.alpha_raw)

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.beta_raw)

    def get_scale_weights(self) -> Tensor:
        """计算各尺度的权重 w_r = exp(-alpha * r^beta)"""
        r = torch.tensor(list(self.radii), dtype=torch.float32, device=self.alpha_raw.device)
        return torch.exp(-self.alpha * (r ** self.beta))

    def forward(self, x: Tensor) -> Tensor:
        """多尺度扩散 + 加权求和
        Args:
            x: 软阈值掩码 (B, 1, H, W)

        Returns:
            先验密度图 (B, 1, H, W)
        """
        weights = self.get_scale_weights()                # (N_scales,)

        prior = None
        for kernel, w in zip(self.kernels, weights):
            feats = kernel(x)          # (B, 1, H, W)
            feats = feats * w          # 尺度加权
            prior = feats if prior is None else prior + feats   # 求和
        return prior

    def extra_repr(self) -> str:
        """返回模块的字符串表示，用于打印和日志记录"""
        radii_str = ", ".join([f"{2*r+1}x{2*r+1}" for r in self.radii])
        alpha = self.alpha.detach().cpu().item()
        beta = self.beta.detach().cpu().item()
        return f"kernels=[{radii_str}], alpha={alpha:.3f}, beta={beta:.3f}, params={sum(p.numel() for p in self.parameters())}"


class LearnablePrior(nn.Module):
    """可学习先验引导模块：组合 DifferentiableThreshold 和 LearnableDiffusion
    将 CT 图像转换为先验密度引导图。

    流程:
        输入 CT (B, 1, H, W)
        → 可微分软阈值: σ((x-lower)/τ)·σ((upper-x)/τ) 
        → 多尺度可学习卷积扩散 (学习 3x3/5x5/7x7 核 + 权重)
        → 先验密度图 (B, 1, H, W)

    Args:
        init_lower: lower 初始值，默认 17.5
        init_upper: upper 初始值，默认 22.0
        init_tau: 软阈值温度初始值，默认 2.5（可学习参数）
        radii: 多尺度卷积核半径，默认 (1, 2, 3)
        init_alpha: 尺度权重衰减系数 α，默认 0.75
        init_beta: 尺度权重衰减指数 β，默认 0.25
    """

    def __init__(
        self,
        init_lower: float = 17.5,
        init_upper: float = 22.0,
        init_tau: float = 2.5,
        radii: Tuple[int, ...] = (1, 2, 3),
        init_alpha: float = 0.75,
        init_beta: float = 0.25,
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
            init_alpha=init_alpha,
            init_beta=init_beta,
        )

    def forward(self, x: Tensor) -> Tensor:
        mask = self.threshold(x)       # 软阈值掩码 (B, 1, H, W)
        prior = self.diffusion(mask)    # 多尺度扩散 (B, 1, H, W)
        return prior

    def get_threshold(self) -> Tuple[float, float]:
        """返回当前 (lower, upper) 值"""
        return self.threshold.get_threshold()


class PriorGuidanceWrapper(nn.Module):
    """先验引导包装器 — 将 可学习先验模块 包裹在任何分割网络外

    即插即用: 在输入通道维拼接图像和先验图，送入原分割网络。
    原网络无需任何改动，只需保证第一层接受 2 通道输入。

    用法:
        base_model = CCFANet(resinc=2)  
        model = PriorGuidanceWrapper(base_model)
        output = model(image)           # image: (B, 1, H, W)

    Args:
        base_model: 任意分割网络，需接受 (B, 2, H, W) 输入
        prior_kwargs: 传递给 LearnablePrior 的参数
    """

    def __init__(
        self,
        base_model: nn.Module,
        **prior_kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.prior_module = LearnablePrior(**prior_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播: 生成先验图 → 通道量级匹配 → 拼接 → 送入分割网络

        通道量级匹配:
          先验图的值域 (~[0, 0.5]) 远小于 CT 图像 (~[0, 50])，
          若直接拼接，第一层卷积中先验通道的贡献和梯度被压制约 200 倍。
          这里对先验图做逐样本标准差缩放，使两通道量级初始化时一致，
          确保 prior_module 的可学习参数从一开始就能收到有效梯度。

        Args:
            x: 输入 CT 图像 (B, 1, H, W)

        Returns:
            分割网络的输出
        """
        prior = self.prior_module(x)                     # (B, 1, H, W)

        # 通道量级匹配: 逐样本缩放先验图至与输入图像同量级
        x_std = x.std(dim=(2, 3), keepdim=True)           # (B, 1, 1, 1)
        p_std = prior.std(dim=(2, 3), keepdim=True)
        scale = x_std / (p_std + 1e-6)
        scale = scale.clamp(max=100.0)                    # 防止极端情况爆炸
        prior = prior * scale                              # (B, 1, H, W)

        x_cat = torch.cat([x, prior], dim=1)              # (B, 2, H, W)
        return self.base_model(x_cat)

    def get_threshold(self) -> Tuple[float, float]:
        return self.prior_module.get_threshold()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用符合 CT 值范围的输入进行测试（实际 CT 值范围 ~ [-1000, 1000]，脑组织 ~ [0, 50]）
    x = torch.rand(8, 1, 256, 256).to(device) * 50  # 模拟 CT 值 [0, 50]
    print(f"输入范围: [{x.min().item():.2f}, {x.max().item():.2f}]")

    # 测试 LearnablePrior 模块
    prior = LearnablePrior().to(device)
    out = prior(x)
    print(f"\nLearnablePrior:")
    print(f"  输入: {x.shape} → 输出: {out.shape}")
    print(f"  lower: {prior.get_threshold()[0]:.2f}, upper: {prior.get_threshold()[1]:.2f}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 调试中间结果
    with torch.no_grad():
        mask = prior.threshold(x)
        print(f"  阈值层输出 shape: {mask.shape}, 范围: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
        diff_out = prior.diffusion(mask)
        print(f"  扩散层输出 shape: {diff_out.shape}, 范围: [{diff_out.min().item():.4f}, {diff_out.max().item():.4f}]")

    # 测试 PriorGuidanceWrapper
    dummy_seg = nn.Conv2d(2, 3, kernel_size=1).to(device)
    wrapped = PriorGuidanceWrapper(dummy_seg).to(device)
    prior_test = wrapped.prior_module(x)
    print(f"\nPriorGuidanceWrapper:")
    print(f"  prior_module 输出 shape: {prior_test.shape}")  # 应该为 (8, 1, 256, 256)
    out2 = wrapped(x)
    print(f"  最终输出 shape: {out2.shape}")

    # 测试参数梯度
    loss = out2.sum()
    loss.backward()
    print(f"\n梯度检查:")
    print(f"  threshold.lower.grad: {wrapped.prior_module.threshold.lower.grad}")
    print(f"  threshold.gap_raw.grad: {wrapped.prior_module.threshold.gap_raw.grad}")
    # 检查可学习参数的梯度
    weights = wrapped.prior_module.diffusion.get_scale_weights()
    for i, kernel in enumerate(wrapped.prior_module.diffusion.kernels):
        print(f"  kernel_{i}.weight.grad: {kernel.weight.grad.mean().item():.6f}")
        print(f"  scale_weight_{i}: {weights[i].item():.6f}  "
              f"(alpha_raw.grad: {wrapped.prior_module.diffusion.alpha_raw.grad.item():.6f}, "
              f"beta_raw.grad: {wrapped.prior_module.diffusion.beta_raw.grad.item():.6f})")