"""
CAPM — 对侧解剖先验匹配模块 (Contralateral Anatomical Prior Matching Module)

核心创新 — 用"局部对应搜索"替代"直接差分"：

理论基础：
  第一步（Prior Module）已给出 NCCT+Prior，网络知道"哪里值得关注"
  但不知道"这里是不是病灶"（脑沟、脑室、钙化都会被 Prior 激活）
  因此第二步利用对侧半脑验证异常是否真实存在。

CAPM 五步流程：
  Step 1 — 解剖对应（中线估计 + 仿射镜像 → Fc）
  Step 2 — 局部对应搜索（K×K 窗口中寻找余弦相似度最高的 Feature）
  Step 3 — 对应置信度生成（基于最高相似度）
  Step 4 — 病理残差（置信度 × (F − F_match)）
  Step 5 — 异常增强（Conv([F, Residual])）

架构:
  (B, C, H, W) ──► Encoder ──► (B, C, H, W) ──┬── Original ──┐
                                                  │              │
                                                  └── CAPM ──────┘
                                                        │
                                                Asymmetry Feature
                                                        │
                                                     Decoder

  医学约束: Normal Consistency Loss — 正常脑左右应高度一致
    L_consistency = ||F − F_match||² 只在非 GT 区域计算

接口与 DPCM/DCR² 完全兼容:
    capm = CAPM(channels=64)
    out = capm(x)               # (B, C, H, W) → (B, C, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# =============================================================================
# 子模块: 中线和旋转角度联合估计器
# =============================================================================
class MidlineRotationEstimator(nn.Module):
    """中线位置 + 旋转角度联合估计器

    中线: 空间激活质心 (COM) + 可学习偏移 MLP
    旋转: GlobalPool → MLP → Tanh(*π/4)

    Args:
        channels: 输入特征通道数 C
    """

    def __init__(self, channels: int):
        super().__init__()
        self.midline_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.rotation_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """估计中线和旋转角度

        Args:
            x: 特征图 (B, C, H, W)

        Returns:
            midline_norm: 归一化中线 x ∈ [-1, 1], (B,)
            angle: 旋转角度 ∈ [-π/4, π/4], (B,)
        """
        B, C, H, W = x.shape

        # 中线: 空间激活质心
        saliency = x.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        sal_mean = saliency.mean(dim=(2, 3), keepdim=True)
        sal_std = saliency.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        prob = torch.sigmoid((saliency - sal_mean) / sal_std)

        prob_h = prob.sum(dim=2)  # (B, 1, W)
        x_coords = torch.arange(W, device=x.device).float().view(1, 1, W)
        com_x = (prob_h * x_coords).sum(dim=2) / (prob_h.sum(dim=2) + 1e-8)
        com_x_norm = 2.0 * com_x / (W - 1) - 1.0  # [0, W-1] → [-1, 1]

        offset = self.midline_mlp(x)
        max_offset_norm = (W // 4) / (W - 1) * 2.0
        midline_norm = com_x_norm + offset * max_offset_norm
        midline_norm = midline_norm.clamp(-0.95, 0.95).squeeze(1)

        # 旋转角度
        angle_raw = self.rotation_mlp(x)
        angle = angle_raw * (math.pi / 4.0)

        return midline_norm, angle.squeeze(1)

    @torch.no_grad()
    def get_midline_angle(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理用: 获取中线和角度值"""
        return self.forward(x)


# =============================================================================
# 子模块: 仿射镜像 grid 构建器
# =============================================================================
def build_affine_grid(midline_norm: torch.Tensor,
                      angle: torch.Tensor,
                      H: int, W: int, device: torch.device) -> torch.Tensor:
    """构建仿射镜像变换的采样网格

    反射变换: 过点 (m, 0)、与垂直方向夹角 θ 的直线。
    当 θ=0 时退化为垂直中线镜像 [[-1, 0, 2m], [0, 1, 0]]。

    Args:
        midline_norm: 中线 x 坐标 (B,)，归一化 [-1, 1]
        angle: 旋转角度 (B,)，弧度 [-π/4, π/4]
        H: 特征图高度
        W: 特征图宽度
        device: 设备

    Returns:
        grid: (B, H, W, 2) 采样网格
    """
    B = midline_norm.shape[0]
    m = midline_norm

    cos2θ = torch.cos(2 * angle)
    sin2θ = torch.sin(2 * angle)
    cosθ = torch.cos(angle)

    θ_ref = torch.zeros(B, 2, 3, device=device)
    θ_ref[:, 0, 0] = -cos2θ
    θ_ref[:, 0, 1] = sin2θ
    θ_ref[:, 0, 2] = 2.0 * m * cosθ.pow(2)
    θ_ref[:, 1, 0] = sin2θ
    θ_ref[:, 1, 1] = cos2θ
    θ_ref[:, 1, 2] = -m * sin2θ

    grid = F.affine_grid(θ_ref, (B, 1, H, W), align_corners=True)
    return grid


# =============================================================================
# 核心模块: CAPM
# =============================================================================
class CAPM(nn.Module):
    """对侧解剖先验匹配模块 (Contralateral Anatomical Prior Matching Module)

    完整流程:
      Step 1 — 解剖对应:
        F → [MidlineRotationEstimator] → (midline, θ)
        → affine_grid → grid_sample → F_mirror (Fc)

      Step 2 — 局部对应搜索:
        (F, Fc) → 逐位置 K×K 窗口余弦相似度 → F_match (最佳对应)

      Step 3 — 对应可靠性:
        Reliability = σ(scale × (sim_max − bias))

      Step 4 — 病理残差:
        Residual = (1 − Reliability) × (F − F_match)
        可靠性高 → 正常组织 → Residual 被抑制
        可靠性低 → 可能病灶 → Residual 被保留

      Step 5 — 残差注入:
        Enhanced = F + Gate × Residual
        Gate = Sigmoid(Conv1x1([F, Residual]))
        F 始终直通，Gate 控制 Residual 注入量

    关键设计:
      - 搜索窗口大小 K×K 默认 7×7，基于脑解剖先验：DPCM 刚体配准后误差 ≤ 几 pixel
      - Reliability 基于匹配相似度（非 Attention），解剖语义明确
      - 正常组织 Reliability 高 → (1-R)→0 → Residual 被抑制
      - 病灶区 Reliability 低 → (1-R)→1 → Residual 被保留

    Args:
        channels: 输入特征通道数 C
        search_window: 局部搜索窗口大小 K，默认 7
        collect_debug: 训练时是否收集中间量供损失函数使用 (default False)
    """

    def __init__(self, channels: int, search_window: int = 7,
                 collect_debug: bool = False):
        super().__init__()
        assert search_window % 2 == 1, f"search_window must be odd, got {search_window}"
        self.channels = channels
        self.search_window = search_window
        self.kernel_size = search_window  # K
        self.r = search_window // 2  # 半径
        self.collect_debug = collect_debug
        self._last_debug: Optional[Dict] = None

        # Step 1: 中线和旋转估计器
        self.estimator = MidlineRotationEstimator(channels)

        # Step 2: 软匹配 — 投影层（降维做相似度，避免 unfold 内存爆炸）
        self.sim_dim = min(max(channels // 4, 16), 64)  # 投影维度：16~64 之间
        self.sim_proj = nn.Conv2d(channels, self.sim_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.sim_proj.weight, mode='fan_out', nonlinearity='linear')

        # Step 3: 对应可靠性（Reliability）— 基于全分布统计特征
        #   从 sim_vol 提取 6 个统计特征，过轻量 Conv1x1(6→8→1) + Sigmoid
        #   特征: [Top1, Top2, Gap, Entropy, Mean, Variance]
        self.reliability_net = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 5: 残差注入门控 — 用可学习 Gate 控制 Residual 的注入量
        #   Enhanced = F + Gate × Residual
        #   Gate = σ(Conv1x1([F, Residual])) — 每个位置独立控制
        #   优点：F 始终直通（信息无损），Gate 决定 Residual 的贡献比例
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    # -----------------------------------------------------------------
    # Step 2: 局部对应搜索（软匹配）
    # -----------------------------------------------------------------
    def _local_correspondence_search(self, F: torch.Tensor, Fc: torch.Tensor
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """软匹配局部对应搜索

        对每个 (i,j)，在 Fc 的 K×K 窗口内计算每个位置与 F[i,j] 的余弦相似度，
        用温度控制的 Softmax 对所有候选位置加权平均，得到软匹配 F_match。

        同时返回完整相似度分布 sim_vol 供 Reliability 计算使用。

        优点：
        - 避免 argmax 的硬决策，多个高相似度位置自然加权
        - 分布平坦时（无明确对应）匹配被平滑，不产生虚假对应
        - 所有操作可导，梯度流畅

        内存优化：通过 1×1 Conv 将 C 投影到 D=min(C//4, 64) 后再做 unfold，
        避免高通道数时 unfold 爆炸。

        Args:
            F: 原始特征图 (B, C, H, W)
            Fc: 对侧镜像特征图 (B, C, H, W)

        Returns:
            F_match: 软匹配特征图 (B, C, H, W)
            sim_vol: 余弦相似度分布体积 (B, K², H, W)
            sim_max: 最大余弦相似度图 (B, H, W)，用于 reliability 计算
        """
        B, C, H, W = F.shape
        K = self.kernel_size
        r = self.r

        # ---- Step 2a: 投影到低维做相似度 ----
        F_proj = self.sim_proj(F)            # (B, D, H, W)
        Fc_proj = self.sim_proj(Fc)          # (B, D, H, W)
        D = F_proj.shape[1]

        # L2 归一化
        F_proj_norm = F_proj / (F_proj.norm(dim=1, keepdim=True) + 1e-8)
        Fc_proj_norm = Fc_proj / (Fc_proj.norm(dim=1, keepdim=True) + 1e-8)

        # ---- Step 2b: unfold 提取 Fc 的 K×K 窗口 ----
        # pad 后 unfold → (B, D*K*K, H*W) → reshape
        Fc_pad = torch.nn.functional.pad(Fc_proj_norm, (r, r, r, r), mode='replicate')
        windows = torch.nn.functional.unfold(Fc_pad, kernel_size=K, stride=1)
        windows = windows.view(B, D, K * K, H, W)             # (B, D, K², H, W)

        # ---- Step 2c: 余弦相似度体积 ----
        sim_vol = (F_proj_norm.unsqueeze(2) * windows).sum(dim=1)  # (B, K², H, W)

        # ---- Step 2d: 软加权（温度控制 Softmax） ----
        tau = 0.5
        attn = torch.nn.functional.softmax(sim_vol / tau, dim=1)

        # ---- Step 2e: 从原始 Fc（非投影）采样 ----
        Fc_pad_orig = torch.nn.functional.pad(Fc, (r, r, r, r), mode='replicate')
        windows_orig = torch.nn.functional.unfold(Fc_pad_orig, kernel_size=K, stride=1)
        windows_orig = windows_orig.view(B, C, K * K, H, W)

        # 加权求和：F_match = Σ attn_k × Fc_window_k
        F_match = (windows_orig * attn.unsqueeze(1)).sum(dim=2)

        # 记录最大相似度
        sim_max = sim_vol.max(dim=1).values

        return F_match, sim_vol, sim_max

    # -----------------------------------------------------------------
    # 前向传播
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                raw_ct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CAPM 前向传播

        Args:
            x: 输入特征图 (B, C, H, W)
            raw_ct: 原始 CT 图像 (B, 1, H, W)，保留以兼容 DPCM 接口，当前未使用

        Returns:
            out: 对侧增强后的特征图 (B, C, H, W)，始终返回 tensor
            当 collect_debug=True 时中间量存储在 self._last_debug 中
        """
        B, C, H, W = x.shape
        device = x.device

        # ======== Step 1: 解剖对应 — 中线和旋转估计 + 仿射镜像 ========
        midline_norm, angle = self.estimator(x)
        affine_grid = build_affine_grid(midline_norm, angle, H, W, device)
        Fc = torch.nn.functional.grid_sample(x, affine_grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)

        # ======== Step 2: 局部对应搜索 ========
        F_match, sim_vol, sim_max = self._local_correspondence_search(x, Fc)

        # ======== Step 3: 对应可靠性 ========
        #   sim_vol 完整分布 → 提取 6 个统计特征:
        #   1. Top1: 最大相似度 — 基础匹配质量
        #   2. Top2: 第二高相似度 — 背景噪声水平
        #   3. Gap: Top1 - Top2 — 唯一性（大→唯一峰值可靠，小→多候选模糊）
        #   4. Entropy: 分布锐利度（低→尖峰，高→平坦不确定）
        #   5. Mean: 平均相似度
        #   6. Variance: 分布离散度
        top1, top2 = torch.topk(sim_vol, k=2, dim=1).values.chunk(2, dim=1)
        gap = top1 - top2                                                    # (B, 1, H, W)
        # 计算熵: -Σ(p_i * log(p_i+1e-10)) 其中 p_i 是 softmax 归一化后的概率
        p = F.softmax(sim_vol / 0.1, dim=1)
        entropy = -(p * torch.log(p + 1e-10)).sum(dim=1, keepdim=True)      # (B, 1, H, W)
        # 均值与方差
        mean = sim_vol.mean(dim=1, keepdim=True)                             # (B, 1, H, W)
        variance = sim_vol.var(dim=1, keepdim=True, unbiased=False)          # (B, 1, H, W)

        # 拼接成 6 通道特征
        reliability_feat = torch.cat([top1, top2, gap, entropy, mean, variance], dim=1)  # (B, 6, H, W)
        reliability = self.reliability_net(reliability_feat)                              # (B, 1, H, W)

        # ======== Step 4: 病理残差 ========
        residual_raw = x - F_match  # (B, C, H, W)
        residual = (1.0 - reliability) * residual_raw  # 不可靠×残差=病灶信号被保留

        # ======== Step 5: 残差注入 — Enhanced = F + Gate × Residual ========
        gate = self.fusion_gate(torch.cat([x, residual], dim=1))
        enhanced = x + gate * residual

        # 收集中间量供训练损失使用
        if self.collect_debug:
            self._last_debug = {
                'out': enhanced,
                'x': x,  # 原始特征图（直接保存，避免反推）
                'midline_norm': midline_norm,
                'angle': angle,
                'Fc': Fc,
                'F_match': F_match,
                'sim_vol': sim_vol,
                'sim_max': sim_max,
                'reliability': reliability.squeeze(1),
                'residual': residual,
            }

        return enhanced


# =============================================================================
# 损失函数: Bootstrap Reliability Loss + Normal Consistency Loss
# =============================================================================

def compute_capm_bootstrap_loss(
    model: nn.Module,
    lambda_bootstrap: float = 0.5,
) -> torch.Tensor:
    """Bootstrap Self-Supervision for Reliability Network

    问题: Consistency Loss 使用 R.detach()，R 不能从中获得梯度。
    Seg Loss 通过 gate → (1−R) → R 的路径太长太间接。
    导致 R 几乎得不到关于匹配质量的训练信号。

    解决方案: 用 sim_vol 提取 Gap = Top1 − Top2 构造唯一性伪标签。
        target = σ(scale × (gap − bias))
        L_bootstrap = MSE(R, target)

    Gap 直接衡量对应的唯一性，比 raw sim_max 好得多：
    - [0.92, 0.91, 0.90, 0.89] → gap=0.01 → 模糊对应 → target≈0.12 ✅
    - [0.90, 0.50, 0.30, 0.10] → gap=0.40 → 唯一对应 → target≈0.88 ✅
    - [0.98, 0.97, 0.96] → gap=0.01 → 模糊对应 → target≈0.12 ✅
    - [0.86, 0.85, 0.84] → gap=0.01 → 模糊对应 → target≈0.12 ✅

    Args:
        model: 包含 CAPM 子模块的模型
        lambda_bootstrap: bootstrap 损失权重

    Returns:
        scalar loss
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    n_modules = 0

    for m in model.modules():
        if isinstance(m, CAPM) and m._last_debug is not None:
            debug = m._last_debug
            reliability = debug.get('reliability')   # (B, H, W)
            sim_vol = debug.get('sim_vol')           # (B, K², H, W)
            if reliability is None or sim_vol is None:
                continue

            # 计算 Gap = Top1 - Top2 作为唯一性指标
            top1, top2 = torch.topk(sim_vol, k=2, dim=1).values.chunk(2, dim=1)
            gap = top1 - top2                        # (B, 1, H, W)

            # 伪标签: gap 大 → 唯一峰值 → 明确对应 → R 应高
            # bias=0.2 压制小 gap 噪声，scale=10 锐利化
            target = torch.sigmoid(10.0 * (gap - 0.2)).squeeze(1)  # (B, H, W)

            # MSE: R 直接向 target 学习
            loss = F.mse_loss(reliability, target)
            total_loss = total_loss + loss
            n_modules += 1

    if n_modules == 0:
        return torch.tensor(0.0, device=device)

    return lambda_bootstrap * total_loss / n_modules


def compute_capm_consistency_loss(
    model: nn.Module,
    lambda_consistency: float = 0.1,
) -> torch.Tensor:
    """计算 Normal Consistency 损失（完全去 GT 化）

    医学先验: 正常脑组织左右半脑应该高度一致。
    CAPM 自己判断哪里是"正常组织"——用 Reliability 作为权重。

    L_consistency = Reliability × ||F - F_match||²

    含义：
    - Reliability 高 → 模型认为"对应可靠，这是正常脑" → 强制 F ≈ F_match
    - Reliability 低 → 模型认为"可能是病灶" → 不做约束，允许残差存在

    全自监督，不需要 GT 标签。

    Args:
        model: 包含 CAPM 子模块的模型
        lambda_consistency: 一致性损失权重

    Returns:
        scalar loss
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    n_modules = 0

    for m in model.modules():
        if isinstance(m, CAPM) and m._last_debug is not None:
            debug = m._last_debug
            x = debug.get('x')
            F_match = debug.get('F_match')
            reliability = debug.get('reliability')  # (B, H, W)
            if x is None or F_match is None or reliability is None:
                continue

            # Reliability-aware一致性损失
            # ⚠️ 必须 detach()！不允许梯度反向流入 Reliability；
            # 否则模型会通过压低所有位置的 Reliability 来"逃避"一致性约束，
            # 导致权重塌陷（所有位置 Reliability→0，一致性损失永远为 0）。
            # 用 detach() 后 Reliability 只作为静态权重，反映匹配质量本身。
            r = reliability.detach().unsqueeze(1)
            diff = (x - F_match).pow(2)
            loss = (r * diff).sum() / (r.sum() + 1e-8)
            total_loss = total_loss + loss
            n_modules += 1

    if n_modules == 0:
        return torch.tensor(0.0, device=device)

    return lambda_consistency * total_loss / n_modules


def compute_capm_consistency_from_debug(
    debug_outputs: list[Dict[str, torch.Tensor]],
    lambda_consistency: float = 0.1,
) -> torch.Tensor:
    """从 CAPM debug 输出计算一致性损失（去 GT 化）

    用 Reliability 作为权重，不需要 GT 掩码。

    Args:
        debug_outputs: CAPM 的 debug 字典列表（每个 CAPM 实例一个）
        lambda_consistency: 一致性损失权重

    Returns:
        scalar loss
    """
    if not debug_outputs:
        return torch.tensor(0.0)

    device = debug_outputs[0].get('x', torch.tensor(0.)).device
    total_loss = torch.tensor(0.0, device=device)
    n_modules = 0

    for debug in debug_outputs:
        x = debug.get('x')           # 直接保存的原始特征
        F_match = debug.get('F_match')
        reliability = debug.get('reliability')
        if x is None or F_match is None or reliability is None:
            continue

        # Reliability作为权重，必须 detach() 以防权重塌陷
        r = reliability.detach().unsqueeze(1)
        diff = (x - F_match).pow(2)  # (B, C, H, W)
        loss = (r * diff).sum() / (r.sum() + 1e-8)
        total_loss = total_loss + loss
        n_modules += 1

    if n_modules == 0:
        return torch.tensor(0.0, device=device)

    return lambda_consistency * total_loss / n_modules


# =============================================================================
# 自测
# =============================================================================
if __name__ == '__main__':
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W, device=device)

    # Test CAPM module
    capm = CAPM(channels=C, collect_debug=True).to(device)
    total_params = sum(p.numel() for p in capm.parameters())
    trainable_params = sum(p.numel() for p in capm.parameters() if p.requires_grad)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"CAPM(channels={C}, search_window={capm.search_window})")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"{sep}")

    out = capm(x)  # returns tensor (B, C, H, W)
    debug = capm._last_debug  # stored debug info
    print(f"\nDebug outputs:")
    for key, value in debug.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    print(f"  reliability range: [{debug['reliability'].min().item():.4f}, {debug['reliability'].max().item():.4f}]")
    print(f"  sim_max range: [{debug['sim_max'].min().item():.4f}, {debug['sim_max'].max().item():.4f}]")

    # Verify shapes
    assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"
    assert debug['Fc'].shape == (B, C, H, W)
    assert debug['F_match'].shape == (B, C, H, W)
    assert debug['sim_max'].shape == (B, H, W)
    assert debug['reliability'].shape == (B, H, W)
    assert debug['residual'].shape == (B, C, H, W)
    print(f"\n  All shape assertions passed")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    grad_count = sum(1 for p in capm.parameters()
                     if p.requires_grad and p.grad is not None)
    total_requires_grad = sum(1 for p in capm.parameters() if p.requires_grad)
    print(f"\n  Gradients: {grad_count}/{total_requires_grad} params have gradients")
    assert grad_count == total_requires_grad, "Missing gradients!"

    # Verify reliability in [0, 1]
    assert debug['reliability'].min() >= 0.0
    assert debug['reliability'].max() <= 1.0
    print(f"  Reliability in [0, 1]: OK")

    print(f"\n[PASS] CAPM self-test passed")