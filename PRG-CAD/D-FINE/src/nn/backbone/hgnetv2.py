"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import logging
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d
from ...core import register

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']

from einops import rearrange

from .swin_transformer import SwinBasicLayer


class BDA(nn.Module):
    def __init__(self, c_in, c_out, WH, wz, depth):  # c_in输入通道，在yaml文件里传进来，在task里处理一下
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()

        self.swin_layer = SwinBasicLayer(dim=c_in, input_resolution=[WH, WH], depth=depth, num_heads=8, window_size=wz,
                                         mlp_ratio=2.)
        # self.cv1 = Conv(c_in, c_out)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c_in, c_in * 2, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(c_in * 2, c_out, 1, 1, 0, bias=True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        identity0 = x  # [batchsize,  c, H, W]
        # x: (B, H * W, C)
        # C, H, W = x.shape[1], x.shape[2], x.shape[3]
        # x = x.permute(0, 2, 3, 1)  # 形状变为 [batchsize, H, W,  c]
        # x = x.reshape(-1, H * W, C)  # 形状变为 [batchsize,  H * W, C]
        x = self.swin_layer(x)
        # x = x.reshape(-1, H, W, C)  #
        # x = x.permute(0, 3, 1, 2)  # 形状变为 [batchsize, C,H, W]
        identity1 = x
        x = identity0 - x
        # 这里直接卷积然后concat会不会更好
        pooled = self.pool(x)
        # pooled_flat = pooled.view(pooled.size(0), -1)
        out = self.act2(self.fc2(self.act1(self.fc1(pooled))))
        return identity1 + identity0 * out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))



class CRA(nn.Module):  # 先下采样 负责宽高向量提供的支路最后不再concat,直接加,1x1的DW不要，仅保留3x3和5x5
    def __init__(self, c, c_out, h, w, depth=6):
        super().__init__()
        assert h == w, "Height and width must be equal"
        self.c = c
        self.h = self.w = h // 2
        h = w = h // 2
        self.depth = depth
        self.conv_wh = LightConv(c, c_out, 3)
        # self.conv_end1=DWConv(2*c_out,c_out,1)
        # self.conv_end= DWConv(c_out,c_out,3,2)
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Shared components
        self.dw_conv = DWConv(c, c, 3)  # 共享DW卷积
        self.dw_conv1 = DWConv(c, c, 3)  # 共享DW卷积
        self.dw_conv2 = DWConv(c, c, 3)  # 共享DW卷积

        # 路径1组件
        self.fc_a = nn.Sequential(nn.Linear(w * 2, 256), nn.ReLU())
        self.fc_a1to6 = nn.ModuleList([nn.Linear(256, w) for _ in range(4)])
        self.multiscale1 = nn.ModuleList([
            # DWConv(2*c, 2*c, 1 ),
            DWConv(2 * c, 2 * c, 3),
            DWConv(2 * c, 2 * c, 5)
        ])

        # 路径2组件
        self.fc_a_prime = nn.Sequential(nn.Linear(h * 2, 256), nn.ReLU())
        self.fc_a1to6_prime = nn.ModuleList([nn.Linear(256, h) for _ in range(4)])
        self.multiscale2 = nn.ModuleList([
            # DWConv(2*c, 2*c, 1),
            DWConv(2 * c, 2 * c, 3),
            DWConv(2 * c, 2 * c, 5)
        ])
        # 注意力生成器
        self.ch_atten_hw = ChannelAttention(c_out)
        self.ch_atten_dw = ChannelAttention(c_out)
        self.ch_atten_hd = ChannelAttention(c_out)
        # self.conv_final = Conv(c_out, c_out, k=1)

    def forward(self, x):
        c = x.shape[1]
        x = self.conv_wh(x)  # 经过wh卷积
        x = self.downSample(x)
        bs_depth, c_out, h, w = x.shape
        if self.training:
            batch_size = bs_depth // self.depth
            depth = self.depth
        else:
            batch_size = 1
            depth = bs_depth
        # ------------------- 路径1处理 -------------------
        # 变形为 [batch_size, h, c, depth, w]
        x_ = x.view(batch_size, depth, c_out, h, w)

        x1 = x_.permute(0, 3, 2, 1, 4)
        x1 = x1.contiguous().view(-1, c_out, depth, w)  # [batch_size*h, c, depth, w]

        B = x1
        # 拆分处理
        A1, A2 = torch.split(x1, c, dim=1)
        A1 = self.dw_conv(A1)  # [batch_size*h, c/2, depth, w]
        A2 = self.dw_conv1(A2)  # [batch_size*h, c/2, depth, w]

        # 池化修正
        A_pool = A1.view(batch_size, w, c, h, depth).mean(dim=(1, 2, 4))  # [batch_size, w]
        A2_pool = A2.view(batch_size, w, c, h, depth).mean(dim=(1, 2, 4))  # [batch_size, w]

        # 全连接处理
        a = self.fc_a(torch.concat((A_pool, A2_pool), dim=1))
        a_list = [fc(a) for fc in self.fc_a1to6]  # 6个[batch_size, w]

        # 多尺度处理
        B_list = [conv(B) for conv in self.multiscale1]
        B_list = [b.view(batch_size, h, c * 2, depth, w)
                      .permute(0, 3, 2, 1, 4)
                      .contiguous().view(bs_depth, c * 2, h, w)
                  for b in B_list]

        # ------------------- 路径2处理 -------------------
        # 变形为 [batch_size, w, c, h, depth]
        x2 = x_.permute(0, 4, 2, 3, 1)
        x2 = x2.contiguous().view(-1, c_out, h, depth)  # [batch_size*w, c, h, depth]

        # 拆分处理
        # A_prime, B_prime = torch.split(x2, c // 2, dim=1)## [batch_size*w, c/2, h, depth]
        B_prime = x2
        A_prime, A2_prime = torch.split(x2, c, dim=1)
        A1_prime = self.dw_conv(A_prime)  # [batch_size*w, c/2, h, depth]
        A2_prime = self.dw_conv2(A2_prime)  # [batch_size*w, c/2, h, depth]

        # 池化修正
        A_pool_prime = A1_prime.view(batch_size, w, c, h, depth).mean(dim=(1, 2, 4))  # [batch_size , h]
        A2_pool_prime = A2_prime.view(batch_size, w, c, h, depth).mean(dim=(1, 2, 4))  # [batch_size , h]

        # 全连接处理
        a_prime = self.fc_a_prime(torch.concat((A_pool_prime, A2_pool_prime), dim=1))
        a_prime_list = [fc(a_prime) for fc in self.fc_a1to6_prime]  # 6个[batch_size, h]

        # 多尺度处理
        B_prime_list = [conv(B_prime) for conv in self.multiscale2]
        B_prime_list = [b.view(batch_size, w, 2 * c, h, depth)
                            .permute(0, 4, 2, 3, 1)
                            .contiguous().view(bs_depth, 2 * c, h, w)
                        for b in B_prime_list]

        # ------------------- 注意力机制 -------------------
        W_list = []
        for a_vec, a_p_vec in zip(a_list, a_prime_list):
            # 外积生成注意力矩阵 [batch_size, h, w]
            W = torch.einsum('bw,bh->bhw', a_vec, a_p_vec)
            W_list.append(W)

        # 扩展注意力矩阵到完整维度 [batch_size*depth, 1, h, w]
        def expand_attention(W, depth):
            return W.unsqueeze(1).repeat(1, depth, 1, 1).view(-1, 1, h, w)

        # 应用注意力
        X1 = sum(expand_attention(W, depth) * B for W, B in zip(W_list[:3], B_list))
        X2 = sum(expand_attention(W, depth) * B for W, B in zip(W_list[3:], B_prime_list))

        # 特征拼接
        A1_final = torch.concat((A1, A2), dim=1).view(batch_size, h, 2 * c, depth, w) \
            .permute(0, 3, 2, 1, 4).contiguous().view(bs_depth, 2 * c, h, w)
        A1_prime_final = torch.concat((A1_prime, A2_prime), dim=1).view(batch_size, w, 2 * c, h, depth) \
            .permute(0, 4, 2, 3, 1).contiguous().view(bs_depth, 2 * c, h, w)

        X1_concat = A1_final + X1
        X2_concat = A1_prime_final + X2

        X1_concat = self.ch_atten_hd(X1_concat)
        X2_concat = self.ch_atten_dw(X2_concat)
        X3 = self.ch_atten_hw(x)
        output = X1_concat + X2_concat + X3
        # return self.conv_final(x_dw + x_hd + x_hw)
        return output


class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class MyStage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0., stage=2
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                int(in_chs * 0.75),
                int(in_chs * 0.75),
                kernel_size=3,
                stride=2,
                groups=int(in_chs * 0.75),
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    int(in_chs * 1.25) if i == 0 else out_chs,
                    mid_chs,
                    int(out_chs),
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)
        if stage == 2:
            self.my_path = nn.Sequential(
                BDA(int(in_chs * 0.25), int(in_chs * 0.25), 64, 8, 2),
                # CRA(int(in_chs * 0.25), int(out_chs * 0.25), kernel_size=3, depth=6),
                CRA(int(in_chs * 0.25), int(out_chs * 0.25), h=64, w=64 ,depth=6),
                BDA(int(out_chs * 0.25), int(out_chs * 0.25), 32, 4, 2)
            )
        else:
            # self.my_path = CRA(int(in_chs * 0.25), int(out_chs * 0.25),  kernel_size=3, depth=6)
            self.my_path = CRA(int(in_chs * 0.25), int(out_chs * 0.25), h=32, w=32 , depth=6)
        # 计算拆分点
        self.split_point = int(in_chs * 0.25)

    def forward(self, x):

        # 从通道维度拆分张量
        x1 = x[:, :self.split_point, :, :]  # 前 0.25 的通道
        x1 = self.my_path(x1)
        x2 = x[:, self.split_point:, :, :]  # 后 0.75 的通道
        x2 = self.downsample(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.blocks(x)
        return x


@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=False,
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = \
                stage_config[
                    k]
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            """
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))
            """
            if i < 2:
                self.stages.append(
                    HG_Stage(
                        in_channels,
                        mid_channels,
                        out_channels,
                        block_num,
                        layer_num,
                        downsample,
                        light_block,
                        kernel_size,
                        use_lab))
            elif i == 2:
                self.stages.append(MyStage(in_channels,
                                           mid_channels,
                                           out_channels,
                                           block_num,
                                           layer_num,
                                           downsample,
                                           light_block,
                                           kernel_size,
                                           use_lab))
            else:
                self.stages.append(MyStage(in_channels,
                                           mid_channels,
                                           out_channels,
                                           block_num,
                                           layer_num,
                                           downsample,
                                           light_block,
                                           kernel_size,
                                           use_lab, stage=3))
            # """
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if False:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    if torch.distributed.get_rank() == 0:
                        print(
                            GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(
                            GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu',
                                                                   model_dir=local_model_dir)
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                self.load_state_dict(state)

            except (Exception, KeyboardInterrupt) as e:
                if torch.distributed.get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                  + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
