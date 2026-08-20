"""
ACC — 解剖连续性一致性正则化 (Anatomical Continuity Consistency)

核心思想：
  将"梗死是连续 3D 病灶"的解剖先验数学化为可微分约束。
  相邻 CT 切片的分割结果应在光流 warp 后保持一致 —
  梗死不会在一层突然出现又在下一层突然消失。

  这是一个纯训练阶段正则化方法：FlowNet + 双向 Warp + 置信度门控 Dice。
  完成训练后丢弃 FlowNet，对推理阶段零影响。

架构概览:
  ┌──────────────────────────────────────────────────────────────────┐
  │                        训练阶段                                   │
  │                                                                  │
  │   CT_z ──► [2D SegNet] ──► S_z ──────────────────┐              │
  │                                                   │              │
  │   CT_{z+1} ──► [2D SegNet] ──► S_{z+1} ──► warp ─┤              │
  │                     (共享权重)                     │              │
  │                                                   ▼              │
  │   CT_z + CT_{z+1} ──► [FlowNet] ──► flow ────► Dice(S_z,         │
  │                                                  S_{z+1}_warped) │
  │                                                                  │
  │   L_total = L_seg(CT_z) + L_seg(CT_{z+1}) + λ·L_continuity      │
  └──────────────────────────────────────────────────────────────────┘

关键设计:
  - FlowNet: 轻量 UNet (~200K 参数)，预测稠密光流 (Dense Flow, 每像素独立位移)，仅训练时使用
  - L_continuity: 双向 Dice 损失 (前向 z→z+1 + 后向 z→z-1) + 边缘感知光流平滑正则化
  - 多因子置信度门控: 熵 × 熵 × 一致性，仅在高置信度且 warp 一致的像素施加约束
  - 即插即用: 网络架构零改动，可叠加 ISDAP (特征空间) + ACC (输出空间)
  - 分段训练: 阶段1 λ=0 (正常2D训练) → 阶段2 逐步增大 λ 微调
  - 纯训练正则化: 完成训练后 FlowNet 被丢弃，推理阶段无任何额外计算或存储开销

接口:
    acc = ACC(in_channels=1, num_classes=2)
    # 训练
    flow = acc.forward_flow(ct_z, ct_z1)         # FlowNet 前向
    loss = acc.compute_continuity_loss(s_z, s_z1, ct_z, ct_z1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# =============================================================================
# 子模块: 轻量 FlowNet (U-Net 架构)
# =============================================================================
class FlowNet(nn.Module):
    """轻量 U-Net 稠密光流估计网络 (Dense Optical Flow)

    输入: 相邻两张 CT 切片拼接 (B, 2*in_channels, H, W)
    输出: 稠密 2D 位移场 (B, 2, H, W)，**每个像素**独立预测 (dx, dy)

    结构: 4 层编码器 + 4 层解码器，skip connections
    参数量: ~200K (in_channels=1, base_ch=8 时约 210K)

    注意: 虽然叫 "轻量" FlowNet，输出是稠密的 (dense) —
         每个空间位置都有独立的位移向量，不是稀疏关键点位移。

    Args:
        in_channels: CT 输入通道数 (default 1, 灰阶 CT)
        base_ch: 基础通道数，控制网络容量 (default 8, 约 200K 参数)
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.input_ch = in_channels * 2  # 拼接两张切片 (2 通道)

        # 编码器输出通道数: [ch, 2ch, 4ch, 8ch]
        ch = base_ch
        e1_ch = ch
        e2_ch = ch * 2
        e3_ch = ch * 4
        e4_ch = ch * 8
        bn_ch = ch * 8  # 瓶颈通道

        # ==== 编码器 ====
        self.enc1 = self._conv_block(self.input_ch, e1_ch)     # 1/1
        self.enc2 = self._conv_block(e1_ch, e2_ch)              # 1/2
        self.enc3 = self._conv_block(e2_ch, e3_ch)              # 1/4
        self.enc4 = self._conv_block(e3_ch, e4_ch)              # 1/8

        # 瓶颈
        self.bottleneck = self._conv_block(bn_ch, bn_ch)       # 1/16

        # ==== 解码器 (input = up_out + skip_enc) ====
        # up4: bn_ch → e3_ch,   cat(e3_ch, e4_ch) → dec4
        self.up4 = nn.ConvTranspose2d(bn_ch, e3_ch, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(e3_ch + e4_ch, e3_ch)     # 1/8

        # up3: e3_ch → e2_ch,   cat(e2_ch, e3_ch) → dec3
        self.up3 = nn.ConvTranspose2d(e3_ch, e2_ch, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(e2_ch + e3_ch, e2_ch)     # 1/4

        # up2: e2_ch → e1_ch,   cat(e1_ch, e2_ch) → dec2
        self.up2 = nn.ConvTranspose2d(e2_ch, e1_ch, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(e1_ch + e2_ch, e1_ch)     # 1/2

        # up1: e1_ch → e1_ch,   cat(e1_ch, e1_ch) → dec1
        self.up1 = nn.ConvTranspose2d(e1_ch, e1_ch, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(e1_ch * 2, e1_ch)         # 1/1

        # 存储通道配置
        self.e1_ch, self.e2_ch, self.e3_ch, self.e4_ch = e1_ch, e2_ch, e3_ch, e4_ch

        # 输出头: 预测 2 通道位移场
        self.flow_head = nn.Sequential(
            nn.Conv2d(e1_ch, e1_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(e1_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(e1_ch, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),  # 归一化位移 ∈ [-1, 1]
        )

        # 可学习缩放因子 (允许网络自行决定最大位移量)
        self.flow_scale = nn.Parameter(torch.tensor(0.05))  # 初始 ~5% 图像尺寸

        self._init_weights()

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """双卷积块: Conv→BN→ReLU → Conv→BN→ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        """初始化权重: Kaiming for Conv, 零偏置"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # flow_head 最后一层零初始化 → 初始预测零位移
        if hasattr(self.flow_head[-2], 'weight'):
            nn.init.zeros_(self.flow_head[-2].weight)
            if self.flow_head[-2].bias is not None:
                nn.init.zeros_(self.flow_head[-2].bias)

    def forward(self,
                ct_z: torch.Tensor,
                ct_z_next: torch.Tensor) -> torch.Tensor:
        """估计相邻切片间的稠密光流 (Dense Optical Flow)

        每个像素独立预测一个 (dx, dy) 位移向量，覆盖整张图像，
        输出分辨率与输入相同 (B, 2, H, W)。

        Args:
            ct_z: 当前切片 (B, C, H, W)
            ct_z_next: 下一张切片 (B, C, H, W)

        Returns:
            flow: 稠密 2D 位移场 (B, 2, H, W)，归一化坐标 [-flow_scale, flow_scale]
        """
        # 拼接
        x = torch.cat([ct_z, ct_z_next], dim=1)  # (B, 2C, H, W)

        # ==== 编码 ====
        e1 = self.enc1(x)                          # (B, ch, H, W)
        e2 = self.enc2(F.max_pool2d(e1, 2))        # (B, 2ch, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))        # (B, 4ch, H/4, W/4)
        e4 = self.enc4(F.max_pool2d(e3, 2))        # (B, 8ch, H/8, W/8)

        # ==== 瓶颈 ====
        b = self.bottleneck(F.max_pool2d(e4, 2))   # (B, 8ch, H/16, W/16)

        # ==== 解码 (U-Net 经典对齐: 上采样后插值到 skip 尺寸) ====
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:],
                               mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:],
                               mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:],
                               mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:],
                               mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 输出头
        flow = self.flow_head(d1) * self.flow_scale  # (B, 2, H, W)

        return flow


# =============================================================================
# 子模块: Warp 操作 (基于 grid_sample)
# =============================================================================
def warp_with_flow(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """用稠密光流 warp 特征/分割图

    Args:
        x: 待 warp 的张量 (B, C, H, W)
        flow: 稠密 2D 位移场 (B, 2, H, W)，归一化坐标偏移

    Returns:
        warped: warp 后的张量 (B, C, H, W)
    """
    B, _, H, W = x.shape
    device = x.device

    # 构建基础归一化网格
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij',
    )
    base_grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # 光流添加偏移
    # flow: (B, 2, H, W) → (B, H, W, 2)
    flow_permuted = flow.permute(0, 2, 3, 1)
    sample_grid = base_grid + flow_permuted

    # 采样
    warped = F.grid_sample(x, sample_grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    return warped


# =============================================================================
# 损失函数: Dice Loss
# =============================================================================
def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-5,
              weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Soft Dice Loss — 适用于概率输出

    Args:
        pred: 预测概率 (B, C, H, W)，softmax 后
        target: 目标 (B, C, H, W)，one-hot 或 soft label
        weight: 逐像素权重 (B, 1, H, W)，用于置信度门控。
                为 None 时等价于标准 Dice (所有像素等权)。

    Returns:
        scalar: 1 - Dice 均值
    """
    dims = (0, 2, 3)  # batch 和 spatial 维度求和
    if weight is not None:
        intersection = (weight * pred * target).sum(dim=dims)
        denom = (weight * (pred + target)).sum(dim=dims)
    else:
        intersection = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def compute_continuity_dice(
    s_z: torch.Tensor,
    s_z_warped: torch.Tensor,
    smooth: float = 1e-5,
    skip_background: bool = True,
    confidence_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算连续性 Dice 损失 (两个方向对称)

    L_cont = Dice(S_z, S_{z+1}_warped)  [双向约束]

    ACC 的核心目标是约束**梗死区域**的跨切片连续性，背景（正常组织）
    覆盖 ~95% 图像面积，其 Dice 始终接近 1.0，包含背景会稀释前景梯度。
    因此默认 skip_background=True，仅对前景类别 (class 1+) 计算 Dice。

    Args:
        s_z: 当前切片分割 (B, C, H, W)，class 0 = 背景
        s_z_warped: warp 后的邻层分割 (B, C, H, W)
        smooth: 数值稳定项
        skip_background: 是否跳过 class 0 (默认 True)
        confidence_gate: 逐像素置信度门控 (B, 1, H, W)，为 None 时所有像素等权

    Returns:
        scalar loss
    """
    if skip_background and s_z.shape[1] > 1:
        s_z = s_z[:, 1:, :, :]           # (B, C-1, H, W)
        s_z_warped = s_z_warped[:, 1:, :, :]
    return dice_loss(s_z, s_z_warped, smooth, weight=confidence_gate)


def compute_confidence_gate(
    s_anchor: torch.Tensor,
    s_warped: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """多因子置信度门控 — 决定每个像素的连续性约束强度

    三个因子 (均逐像素, ∈ [0, 1]):
      f_conf_anchor = exp(-H_anchor / τ)      熵置信度 (分布越陡越自信)
      f_conf_warped = exp(-H_warped / τ)      邻层熵置信度
      f_agree       = 1 - ½·|anchor - warped|₁  预测一致性 (warp 后是否真的吻合)

    gate = f_conf_anchor · f_conf_warped · f_agree

    gate ≈ 1: 两端高置信 + warp 后一致 → 强连续性约束
    gate ≈ 0: 任一端不确定, 或 warp 后矛盾 (真实病灶变化/光流失败)

    为何优于旧版 maxprob 乘积:
      - 熵 (vs maxprob): 捕获分布陡峭度, 对 [0.51,0.49] 和 [0.99,0.01] 的区分度远高于 maxprob
      - 一致性因子: 即使两端都自信, 若 warp 后矛盾也放松约束,
        防止在真实跨切片变化处强行施加连续性

    Args:
        s_anchor: 锚定切片的分割概率 (B, C, H, W)
        s_warped: warp 后的邻层分割概率 (B, C, H, W), 已在调用前完成 warp
        temperature: 温度 τ, 越大 → 门控对不确定性越宽容 (default 1.0)
        eps: 数值稳定项

    Returns:
        gate: 逐像素门控 (B, 1, H, W), 每个值 ∈ (0, 1]
    """
    # f_conf: H = -Σ p_c·log(p_c), c = exp(-H/τ)
    H_anchor = -(s_anchor * (s_anchor + eps).log()).sum(dim=1, keepdim=True)
    H_warped = -(s_warped * (s_warped + eps).log()).sum(dim=1, keepdim=True)
    f_conf_anchor = (-H_anchor / temperature).exp()
    f_conf_warped = (-H_warped / temperature).exp()

    # f_agree: Total Variation distance → 相似度
    # |p-q|₁ ∈ [0, 2], f_agree = 1 - ½·|p-q|₁ ∈ [0, 1]
    f_agree = 1.0 - 0.5 * (s_anchor - s_warped).abs().sum(dim=1, keepdim=True)

    gate = f_conf_anchor * f_conf_warped * f_agree
    return gate


# =============================================================================
# 损失函数: 光流正则化
# =============================================================================
def flow_smoothness_loss(
    flow: torch.Tensor,
    image: torch.Tensor,
    edge_weight: float = 10.0,
) -> torch.Tensor:
    """边缘感知光流平滑损失 (Edge-Aware Flow Smoothness)

    对稠密光流场的空间梯度施加 L1 惩罚，按图像梯度加权：
      - 组织内部 (|∇I| 小) → 权重 ≈ 1 → 强平滑约束
      - 解剖边界 (|∇I| 大) → 权重 ≈ 0 → 允许流场突变

    这是光流文献 (PWC-Net, RAFT 等) 的标准正则化项，防止网络学习
    噪声流场来"作弊"降低 Dice loss。

    公式:
      w_y = exp(-edge_weight · |∂I/∂y|)
      w_x = exp(-edge_weight · |∂I/∂x|)
      L_smooth = mean(|∂u/∂y|·w_y + |∂u/∂x|·w_x +
                      |∂v/∂y|·w_y + |∂v/∂x|·w_x)

    Args:
        flow: 稠密光流 (B, 2, H, W)
        image: 参考图像，用于边缘检测 (B, C, H, W)
               multi-channel 时自动做 mean 降维
        edge_weight: 边缘敏感度 (默认 10.0)，越大则边缘处 smoothness
                     松弛越激进

    Returns:
        scalar smoothness loss (始终 ≥ 0)
    """
    # --- 流场空间梯度 (一阶前向差分) ---
    du_dy = flow[:, 0, 1:, :] - flow[:, 0, :-1, :]  # ∂u/∂y  (B, H-1, W)
    du_dx = flow[:, 0, :, 1:] - flow[:, 0, :, :-1]  # ∂u/∂x  (B, H, W-1)
    dv_dy = flow[:, 1, 1:, :] - flow[:, 1, :-1, :]  # ∂v/∂y  (B, H-1, W)
    dv_dx = flow[:, 1, :, 1:] - flow[:, 1, :, :-1]  # ∂v/∂x  (B, H, W-1)

    # --- 图像空间梯度 (边缘检测) ---
    if image.shape[1] > 1:
        img = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        img = image

    di_dy = img[:, 0, 1:, :] - img[:, 0, :-1, :]  # ∂I/∂y  (B, H-1, W)
    di_dx = img[:, 0, :, 1:] - img[:, 0, :, :-1]  # ∂I/∂x  (B, H, W-1)

    # --- 边缘权重: |∇I| 大 → w ≈ 0 (松约束) ---
    w_y = torch.exp(-edge_weight * di_dy.abs())
    w_x = torch.exp(-edge_weight * di_dx.abs())

    # --- 加权平滑损失 ---
    loss = (
        (du_dy.abs() * w_y).mean() +
        (du_dx.abs() * w_x).mean() +
        (dv_dy.abs() * w_y).mean() +
        (dv_dx.abs() * w_x).mean()
    )
    return loss


# =============================================================================
# 核心模块: ACC
# =============================================================================
class ACC(nn.Module):
    """解剖连续性一致性正则化 (Anatomical Continuity Consistency)

    封装 FlowNet + 置信度门控 + 双向 Dice 连续性约束的完整训练流程。
    纯训练阶段正则化方法，完成训练后 FlowNet 被丢弃，对推理零影响。

    Args:
        in_channels: CT 输入通道数 (default 1)
        num_classes: 分割类别数 (default 2)
        flow_base_ch: FlowNet 基础通道数 (default 8)
        lambda_continuity: L_continuity 权重 λ (default 0.1)
        lambda_smooth: 光流平滑正则化权重 (default 0.01)
        gate_temperature: 置信度门控温度 τ, 越大对不确定性越宽容 (default 1.0)
        collect_debug: 训练时是否收集调试信息 (default False)
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 flow_base_ch: int = 8,
                 lambda_continuity: float = 0.1,
                 lambda_smooth: float = 0.01,
                 gate_temperature: float = 1.0,
                 collect_debug: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lambda_continuity = lambda_continuity
        self.lambda_smooth = lambda_smooth
        self.gate_temperature = gate_temperature
        self.collect_debug = collect_debug
        self._last_debug: Optional[Dict] = None

        # FlowNet: 仅训练时使用
        self.flownet = FlowNet(in_channels=in_channels, base_ch=flow_base_ch)

    def forward_flow(self,
                     ct_z: torch.Tensor,
                     ct_z_next: torch.Tensor) -> torch.Tensor:
        """前向稠密光流估计 (z → z+1)

        每个像素独立输出 (dx, dy)，覆盖全图。

        Args:
            ct_z: 当前切片 (B, C, H, W)
            ct_z_next: 下一张切片 (B, C, H, W)

        Returns:
            flow: 稠密 2D 位移场 (B, 2, H, W)
        """
        return self.flownet(ct_z, ct_z_next)

    def compute_continuity_loss(
        self,
        s_z: torch.Tensor,
        s_z_next: torch.Tensor,
        ct_z: Optional[torch.Tensor] = None,
        ct_z_next: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        flow_rev: Optional[torch.Tensor] = None,
        confidence_gated: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算双向连续性损失

        双向: 前向 flow (z→z+1) warp S_{z+1} 与 S_z 比较
              + 后向 flow (z+1→z) warp S_z 与 S_{z+1} 比较

        光流平滑正则化 (需提供 ct_z/ct_z_next):
          边缘感知 TV smoothness，防止网络学习噪声流场。
          L_smooth = mean(|∇flow| · exp(-|∇I|))
          组织内部强平滑，解剖边界处允许流场突变。

        置信度门控 (confidence_gated=True, 默认启用):
          仅在**两端都高置信度**的像素处施加连续性约束。

        三种调用模式 (优先级从高到低):
          1. 传入 (flow, flow_rev) → 使用预计算的双向光流
             若同时传入 (ct_z, ct_z_next) → 额外计算平滑正则化
          2. 传入 flow (无 flow_rev) → 后向用 -flow 近似 (兼容旧接口)
          3. 传入 (ct_z, ct_z_next) → 内部计算双向光流 + 平滑正则化

        Args:
            s_z: 切片 z 的分割概率 (B, C, H, W)
            s_z_next: 切片 z+1 的分割概率 (B, C, H, W)
            ct_z: 切片 z 的 CT 图像 (可选，需与 ct_z_next 一起提供)
            ct_z_next: 切片 z+1 的 CT 图像
            flow: 预计算的前向光流 z→z+1 (若已计算)
            flow_rev: 预计算的后向光流 z+1→z (可选，需与 flow 一起提供;
                      若不提供则用 -flow 近似)
            confidence_gated: 是否启用置信度门控 (默认 True)

        Returns:
            loss_cont: 总连续性损失 (scalar)
            extra: 包含 flow_fwd, flow_rev, loss_fwd, loss_rev 的字典
        """
        if flow is not None and flow_rev is not None:
            # 模式 1 (推荐): 使用预计算的双向光流
            #   若同时提供 CT → 额外计算平滑正则化
            flow_fwd = flow
            # flow_rev 直接使用传入值
        elif flow is not None:
            # 模式 2: 仅前向流 → 后向用 -flow 近似 (兼容旧接口)
            flow_fwd = flow
            flow_rev = -flow
        elif ct_z is not None and ct_z_next is not None:
            # 模式 3: 从 CT 内部计算双向光流
            flow_fwd = self.forward_flow(ct_z, ct_z_next)
            flow_rev = self.forward_flow(ct_z_next, ct_z)
        else:
            raise ValueError(
                "必须提供 (ct_z, ct_z_next) 或 flow，"
                "建议同时提供 flow + flow_rev + ct 以获得平滑正则化"
            )

        # 前向 warp
        s_z_next_warped = warp_with_flow(s_z_next, flow_fwd)

        # 后向 warp
        s_z_warped = warp_with_flow(s_z, flow_rev)

        # 置信度门控: 多因子 (熵 + 一致性), 替换旧版 maxprob 乘积
        if confidence_gated:
            # 前向: anchor=S_z, warped=S_{z+1}_{warped by flow_fwd}
            gate_fwd = compute_confidence_gate(
                s_z, s_z_next_warped, temperature=self.gate_temperature)

            # 后向: anchor=S_{z+1}, warped=S_z_{warped by flow_rev}
            gate_rev = compute_confidence_gate(
                s_z_next, s_z_warped, temperature=self.gate_temperature)
        else:
            gate_fwd = None
            gate_rev = None

        loss_fwd = compute_continuity_dice(s_z, s_z_next_warped,
                                           confidence_gate=gate_fwd)
        loss_rev = compute_continuity_dice(s_z_next, s_z_warped,
                                           confidence_gate=gate_rev)

        # 光流平滑正则化 (边缘感知, 需 CT 图像)
        if ct_z is not None:
            loss_smooth_fwd = flow_smoothness_loss(flow_fwd, ct_z)
            loss_smooth_rev = flow_smoothness_loss(flow_rev, ct_z_next)
            loss_smooth = (loss_smooth_fwd + loss_smooth_rev) / 2.0
        else:
            loss_smooth = torch.tensor(0.0, device=s_z.device)

        loss_cont = self.lambda_continuity * (loss_fwd + loss_rev) / 2.0 \
                    + self.lambda_smooth * loss_smooth

        extra = {
            'flow_fwd': flow_fwd,
            'flow_rev': flow_rev,
            'loss_fwd': loss_fwd,
            'loss_rev': loss_rev,
            'loss_smooth': loss_smooth,
            's_z_next_warped': s_z_next_warped,
            's_z_warped': s_z_warped,
        }
        if confidence_gated:
            extra['gate_fwd'] = gate_fwd
            extra['gate_rev'] = gate_rev

        if self.collect_debug:
            self._last_debug = {
                **extra,
                's_z': s_z,
                's_z_next': s_z_next,
            }

        return loss_cont, extra

    def forward(self,
                ct_z: torch.Tensor,
                ct_z_next: torch.Tensor,
                s_z: Optional[torch.Tensor] = None,
                s_z_next: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """统一前向接口

        Args:
            ct_z: 切片 z (B, C, H, W)
            ct_z_next: 切片 z+1 (B, C, H, W)
            s_z: 切片 z 的分割概率 (训练时提供)
            s_z_next: 切片 z+1 的分割概率 (训练时提供)

        Returns:
            dict with keys: flow_fwd, flow_rev, loss_cont (提供 s 时), extra
        """
        flow_fwd = self.forward_flow(ct_z, ct_z_next)
        flow_rev = self.forward_flow(ct_z_next, ct_z)

        result = {'flow_fwd': flow_fwd, 'flow_rev': flow_rev}

        if s_z is not None and s_z_next is not None:
            loss_cont, extra = self.compute_continuity_loss(
                s_z, s_z_next,
                ct_z=ct_z, ct_z_next=ct_z_next,
                flow=flow_fwd, flow_rev=flow_rev)
            result['loss_cont'] = loss_cont
            result.update(extra)

        return result


# =============================================================================
# 损失函数: 从模型收集 ACC 连续性损失
# =============================================================================

def compute_acc_continuity_loss(
    model: nn.Module,
    s_z: torch.Tensor,
    s_z_next: torch.Tensor,
    ct_z: Optional[torch.Tensor] = None,
    ct_z_next: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """从包含 ACC 子模块的模型中收集连续性损失

    遍历模型中所有 ACC 模块，累加连续性损失。

    Args:
        model: 包含 ACC 子模块的模型
        s_z: 切片 z 的分割概率
        s_z_next: 切片 z+1 的分割概率
        ct_z: 切片 z 的 CT (可选)
        ct_z_next: 切片 z+1 的 CT (可选)

    Returns:
        total_loss: 连续性子模块的标量损失和
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)

    for m in model.modules():
        if isinstance(m, ACC):
            loss_cont, _ = m.compute_continuity_loss(
                s_z, s_z_next, ct_z, ct_z_next)
            total_loss = total_loss + loss_cont

    return total_loss


def compute_acc_total_loss(
    model: nn.Module,
    s_z: torch.Tensor,
    s_z_next: torch.Tensor,
    ct_z: torch.Tensor,
    ct_z_next: torch.Tensor,
    lambda_seg: float = 1.0,
    lambda_cont: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """计算 ACC 总损失

    L_total = λ_seg · [L_seg(S_z) + L_seg(S_{z+1})] + λ_cont · L_continuity

    Args:
        model: 包含 ACC 子模块的模型
        s_z, s_z_next: 分割概率
        ct_z, ct_z_next: CT 图像
        lambda_seg: 分割损失权重
        lambda_cont: 连续性损失权重

    Returns:
        dict: {'loss_total', 'loss_cont', 'loss_seg_sum'}
    """
    device = s_z.device
    total_cont = torch.tensor(0.0, device=device)

    for m in model.modules():
        if isinstance(m, ACC):
            m.lambda_continuity = lambda_cont
            loss_cont, _ = m.compute_continuity_loss(
                s_z, s_z_next, ct_z, ct_z_next)
            total_cont = total_cont + loss_cont

    return {
        'loss_cont': total_cont,
        'loss_total': total_cont,  # L_seg 由外部计算后加上
    }


# =============================================================================
# 自测
# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sep = "=" * 60

    # ===== Test FlowNet =====
    print(f"\n{sep}")
    print("Test 1: FlowNet")
    print(sep)

    B, C, H, W = 2, 1, 128, 128
    ct_z = torch.randn(B, C, H, W, device=device)
    ct_z1 = torch.randn(B, C, H, W, device=device)

    flownet = FlowNet(in_channels=1).to(device)  # base_ch=8 默认 ~200K
    total_params = sum(p.numel() for p in flownet.parameters())
    trainable_params = sum(p.numel() for p in flownet.parameters()
                           if p.requires_grad)
    print(f"FlowNet(in_channels=1, base_ch=8)")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    flow = flownet(ct_z, ct_z1)
    print(f"  Input:  {ct_z.shape} + {ct_z1.shape}")
    print(f"  Output: {flow.shape}")
    assert flow.shape == (B, 2, H, W), f"Expected {(B, 2, H, W)}, got {flow.shape}"

    # 初始输出应接近零（因为最后一层零初始化）
    flow_abs_mean = flow.abs().mean().item()
    print(f"  Initial |flow| mean: {flow_abs_mean:.6f} (should be small)")
    assert flow_abs_mean < 0.01, f"Flow too large: {flow_abs_mean}"

    # 梯度流测试
    loss = flow.sum()
    loss.backward()
    grad_count = sum(1 for p in flownet.parameters()
                     if p.requires_grad and p.grad is not None)
    total_requires_grad = sum(1 for p in flownet.parameters() if p.requires_grad)
    print(f"  Gradients: {grad_count}/{total_requires_grad} params have gradients")
    assert grad_count == total_requires_grad, f"Missing gradients!"

    print(f"[PASS] FlowNet test passed")

    # ===== Test Warp =====
    print(f"\n{sep}")
    print("Test 2: warp_with_flow")
    print(sep)

    # 全零 flow → warp 应等于恒等
    zero_flow = torch.zeros(B, 2, H, W, device=device)
    s = torch.rand(B, 2, H, W, device=device)
    s_warped_zero = warp_with_flow(s, zero_flow)
    diff = (s - s_warped_zero).abs().max().item()
    print(f"  Zero flow warp max diff: {diff:.8f} (should be ≈ 0)")
    assert diff < 1e-4, f"Zero flow should give identity warp, got diff={diff}"

    # 非零 flow → 应与输入不同
    nonzero_flow = torch.randn(B, 2, H, W, device=device) * 0.1
    s_warped_nonzero = warp_with_flow(s, nonzero_flow)
    diff_nonzero = (s - s_warped_nonzero).abs().mean().item()
    print(f"  Nonzero flow warp mean diff: {diff_nonzero:.6f} (should be > 0)")
    assert diff_nonzero > 1e-6, "Nonzero flow should change the warped output"

    print(f"[PASS] Warp test passed")

    # ===== Test Dice Loss =====
    print(f"\n{sep}")
    print("Test 3: Dice Loss")
    print(sep)

    pred = torch.rand(B, 2, H, W, device=device).softmax(dim=1)
    target = pred.clone()  # 完全一致
    loss_identical = compute_continuity_dice(pred, target)
    print(f"  Dice loss (identical): {loss_identical.item():.6f} (should be ≈ 0)")

    # 随机 target
    target_random = torch.rand(B, 2, H, W, device=device).softmax(dim=1)
    loss_random = compute_continuity_dice(pred, target_random)
    print(f"  Dice loss (random): {loss_random.item():.6f} (should be > 0)")
    assert loss_random > 0, "Random targets should yield positive Dice loss"

    print(f"[PASS] Dice loss test passed")

    # ===== Test Flow Smoothness =====
    print(f"\n{sep}")
    print("Test 3b: Flow Smoothness Loss")
    print(sep)

    # 构造一个干净边缘的图像
    ct_edge = torch.zeros(B, 1, H, W, device=device)
    ct_edge[:, :, :, :W//2] = 0.3   # 左半: 软组织 (低强度)
    ct_edge[:, :, :, W//2:] = 0.9   # 右半: 骨骼 (高强度)
    # 中间有垂直边缘 |∇I| 大

    # 情景 1: 跨边缘有噪声流场 → high smoothness loss
    noisy_flow = torch.randn(B, 2, H, W, device=device) * 0.1
    loss_noisy = flow_smoothness_loss(noisy_flow, ct_edge)
    print(f"  Noisy flow on edged image: {loss_noisy.item():.6f}")

    # 情景 2: 完全平滑的流场 → low smoothness loss
    smooth_flow = torch.zeros(B, 2, H, W, device=device)
    loss_smooth_zero = flow_smoothness_loss(smooth_flow, ct_edge)
    print(f"  Zero flow (perfectly smooth): {loss_smooth_zero.item():.8f}")
    assert loss_smooth_zero < 1e-6, \
        f"Zero flow should have near-zero smoothness loss, got {loss_smooth_zero.item():.8f}"

    # 情景 3: 流场仅在图像边缘处有突变 → 边缘感知降低惩罚
    # 构造: 流场在边缘处跳变，其他地方平滑
    edge_jump_flow = torch.zeros(B, 2, H, W, device=device)
    edge_jump_flow[:, 0, :, W//2:] = 0.05  # 只在右半有水平位移
    loss_edge_jump = flow_smoothness_loss(edge_jump_flow, ct_edge)

    # 同样流场在无边缘图像上 → penalty 应更大 (没有边缘来 justify 跳变)
    ct_flat = torch.zeros(B, 1, H, W, device=device)  # 全图均匀
    loss_no_edge = flow_smoothness_loss(edge_jump_flow, ct_flat)

    print(f"  Edge jump flow on edged image: {loss_edge_jump.item():.6f}")
    print(f"  Same flow on flat image:       {loss_no_edge.item():.6f}")
    print(f"  Edge-aware reduction: {(1 - loss_edge_jump.item()/loss_no_edge.item())*100:.1f}%")
    assert loss_edge_jump < loss_no_edge, \
        "Edge-aware should reduce penalty for jumps at image edges"
    assert loss_noisy > loss_smooth_zero, \
        "Noisy flow should have higher smoothness loss than smooth flow"

    # 情景 4: 梯度流测试
    flow_grad = torch.randn(B, 2, H, W, device=device, requires_grad=True)
    loss_grad = flow_smoothness_loss(flow_grad, ct_edge)
    loss_grad.backward()
    assert flow_grad.grad is not None and flow_grad.grad.abs().sum() > 0, \
        "Gradients should flow through smoothness loss"
    print(f"  Gradient flow: OK")

    print(f"[PASS] Flow smoothness test passed")

    # ===== Test ACC Module =====
    print(f"\n{sep}")
    print("Test 4: ACC Module")
    print(sep)

    acc = ACC(in_channels=1, num_classes=2, collect_debug=True).to(device)
    total_params = sum(p.numel() for p in acc.parameters())
    print(f"ACC(in_channels=1, num_classes=2)")
    print(f"  Total params: {total_params:,}")
    print(f"  λ_continuity: {acc.lambda_continuity}")

    s_z = torch.rand(B, 2, H, W, device=device).softmax(dim=1)
    s_z1 = torch.rand(B, 2, H, W, device=device).softmax(dim=1)

    # 测试连续性损失
    loss_cont, extra = acc.compute_continuity_loss(s_z, s_z1, ct_z, ct_z1)
    print(f"  L_continuity: {loss_cont.item():.6f}")
    print(f"  flow_fwd: {extra['flow_fwd'].shape}")
    print(f"  flow_rev: {extra['flow_rev'].shape}")
    print(f"  loss_fwd: {extra['loss_fwd'].item():.6f}")
    print(f"  loss_rev: {extra['loss_rev'].item():.6f}")
    print(f"  loss_smooth: {extra['loss_smooth'].item():.6f}")
    assert not torch.isnan(loss_cont), "NaN in continuity loss!"

    # 测试统一接口
    result = acc(ct_z, ct_z1, s_z, s_z1)
    print(f"  forward(train) keys: {list(result.keys())}")
    assert 'loss_cont' in result
    assert 'flow_fwd' in result

    # 梯度流测试
    loss_total = result['loss_cont']
    loss_total.backward()
    grad_count = sum(1 for p in acc.parameters()
                     if p.requires_grad and p.grad is not None)
    total_requires_grad = sum(1 for p in acc.parameters() if p.requires_grad)
    print(f"  Gradients: {grad_count}/{total_requires_grad} params have gradients")
    assert grad_count == total_requires_grad, f"Missing gradients!"

    print(f"[PASS] ACC module test passed")

    # ===== Test Confidence-Gated Continuity Loss =====
    print(f"\n{sep}")
    print("Test 4b: Multi-Factor Confidence Gate (Entropy + Agreement)")
    print(sep)

    # 构造场景: 左半两端自信且一致, 右半一端自信一端不自信且不一致
    # 左: [0.01,0.99] × [0.01,0.99],  H≈0.056, conf≈0.946, agree=1.0 → gate≈0.894
    # 右: [0.01,0.99] × [0.45,0.55],  H≈0.688, conf≈0.503, agree≈0.56 → gate≈0.267
    H_half = H // 2
    s_anchor = torch.zeros(B, 2, H, W, device=device)
    s_anchor[:, 0, :, :] = 0.01
    s_anchor[:, 1, :, :] = 0.99                     # 全图高置信前景

    s_neighbor = torch.zeros(B, 2, H, W, device=device)
    s_neighbor[:, 0, :, :] = 0.45                   # 默认: 低置信
    s_neighbor[:, 1, :, :] = 0.55
    s_neighbor[:, 0, :, :H_half] = 0.01             # 左半: 高置信且与 anchor 一致
    s_neighbor[:, 1, :, :H_half] = 0.99

    zero_flow = torch.zeros(B, 2, H, W, device=device)

    # 门控 on vs off
    loss_gated, extra_gated = acc.compute_continuity_loss(
        s_anchor, s_neighbor, flow=zero_flow, confidence_gated=True)
    loss_ungated, _ = acc.compute_continuity_loss(
        s_anchor, s_neighbor, flow=zero_flow, confidence_gated=False)

    print(f"  Left: both confident + agree | Right: confident x uncertain + disagree")
    print(f"    Gated:   {loss_gated.item():.6f}  (右半因 disagreement 被大幅压缩)")
    print(f"    Ungated: {loss_ungated.item():.6f}  (右半 mismatch 等权计入)")
    reduction_pct = (1 - loss_gated.item() / loss_ungated.item()) * 100
    print(f"    Reduction: {reduction_pct:.1f}%")

    assert loss_gated.item() < loss_ungated.item(), \
        "Gate should reduce loss when mismatch region has low confidence"
    # 三因子门控 (含一致性子因子) 比旧 maxprob 压缩更狠
    assert reduction_pct > 20, \
        f"Multi-factor gate should reduce loss by >20%, got {reduction_pct:.1f}%"

    # --- 验证三个因子的数值 ---
    gate_fwd_raw = compute_confidence_gate(s_anchor, s_neighbor, temperature=1.0)
    gate_left_raw = gate_fwd_raw[:, :, :, :H_half].mean().item()
    gate_right_raw = gate_fwd_raw[:, :, :, H_half:].mean().item()
    print(f"    Gate mean left:  {gate_left_raw:.4f}  (~0.89: confxconfxagree=0.95x0.95x1.0)")
    print(f"    Gate mean right: {gate_right_raw:.4f}  (~0.27: confxconfxagree=0.95x0.50x0.56)")
    print(f"    Left/Right ratio: {gate_left_raw/gate_right_raw:.1f}x")
    assert gate_left_raw > 0.85, f"Left gate should be >0.85, got {gate_left_raw:.3f}"
    assert gate_right_raw < 0.35, f"Right gate should be <0.35, got {gate_right_raw:.3f}"
    assert gate_left_raw > gate_right_raw * 3.0, \
        f"Multi-factor gate should strongly distinguish, ratio={gate_left_raw/gate_right_raw:.1f}"

    # --- 验证一致性因子独立作用 ---
    # 构造: 两端都自信, 但 warp 后矛盾 (模拟真实病灶跨层变化)
    s_conflict = torch.zeros(B, 2, H, W, device=device)
    s_conflict[:, 0, :, :] = 0.99   # 自信地预测为背景
    s_conflict[:, 1, :, :] = 0.01

    gate_agree = compute_confidence_gate(s_anchor, s_conflict, temperature=1.0)
    gate_agree_mean = gate_agree.mean().item()
    print(f"    Both confident but disagree: gate={gate_agree_mean:.4f} (should be near 0)")
    assert gate_agree_mean < 0.05, \
        f"Agreement factor should kill gate when predictions contradict, got {gate_agree_mean:.3f}"

    # --- 验证 entropy 信息量 > maxprob ---
    s_50 = torch.zeros(1, 2, 1, 1, device=device); s_50[:, 0, :, :] = 0.5; s_50[:, 1, :, :] = 0.5
    s_60 = torch.zeros(1, 2, 1, 1, device=device); s_60[:, 0, :, :] = 0.4; s_60[:, 1, :, :] = 0.6
    s_99 = torch.zeros(1, 2, 1, 1, device=device); s_99[:, 0, :, :] = 0.01; s_99[:, 1, :, :] = 0.99

    g_50 = compute_confidence_gate(s_50, s_50, temperature=1.0).item()
    g_60 = compute_confidence_gate(s_60, s_60, temperature=1.0).item()
    g_99 = compute_confidence_gate(s_99, s_99, temperature=1.0).item()
    print(f"    Gate([.5,.5] vs self): {g_50:.4f}  (low but non-zero)")
    print(f"    Gate([.6,.4] vs self): {g_60:.4f}  (slightly higher)")
    print(f"    Gate([.99,.01] vs self): {g_99:.4f}  (near 1)")
    assert g_50 < g_60 < g_99, \
        "Gate should monotonically increase with prediction sharpness"

    # 梯度流
    loss_test, _ = acc.compute_continuity_loss(
        s_z, s_z1, ct_z, ct_z1, confidence_gated=True)
    loss_test.backward()
    print(f"  Gradients flow through gated loss: OK")

    print(f"[PASS] Multi-factor confidence gate test passed")

    # ===== Test compute_acc_continuity_loss =====
    print(f"\n{sep}")
    print("Test 5: Global loss collector")
    print(sep)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.acc = ACC(in_channels=1, num_classes=2)

    dummy = DummyModel().to(device)
    loss_collected = compute_acc_continuity_loss(dummy, s_z, s_z1, ct_z, ct_z1)
    print(f"  Collected L_cont: {loss_collected.item():.6f}")
    assert loss_collected.item() > 0, "Should collect nonzero loss"

    print(f"[PASS] Global loss collector test passed")

    # ===== Test different input sizes =====
    print(f"\n{sep}")
    print("Test 6: Different input sizes")
    print(sep)

    # 16 的倍数（之前覆盖的）和非 16 倍数（验证对齐修复）
    for size in [(64, 64), (127, 127), (224, 224), (255, 255), (256, 256)]:
        H_t, W_t = size
        ct_test = torch.randn(1, 1, H_t, W_t, device=device)
        ct_test2 = torch.randn(1, 1, H_t, W_t, device=device)
        flow_test = flownet(ct_test, ct_test2)
        assert flow_test.shape == (1, 2, H_t, W_t), \
            f"Expected (1,2,{H_t},{W_t}), got {flow_test.shape}"
        print(f"  Input ({H_t},{W_t}) -> Flow {flow_test.shape}  [OK]")

    print(f"[PASS] Input sizes test passed (incl. non-16-multiple)")

    print(f"\n{'=' * 60}")
    print(f"All ACC tests passed successfully!")
    print(f"{'=' * 60}")
