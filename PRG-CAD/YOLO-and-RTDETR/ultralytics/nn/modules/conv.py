# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv", "Conv3D", "Upsample",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv", "MixDonwSample"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MixDonwSample1(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        self.conv2D = Conv(in_ch, out_ch, k=k, s=s)
        # 将3*3*3改为1*1*3试试
        self.conv3D = Conv3D(in_ch, out_ch, k=k, s=s, depth=6)
        # self.conv3D = Conv3D(in_ch, out_ch, k= (k, 1, 1), s=s, depth=6)

        # 通道注意力
        self.ch_atten3D = ChannelAttention(out_ch)
        self.ch_atten2D = ChannelAttention(out_ch)

        # self.sp_atten3D=SpatialAttention()
        # self.sp_atten2D=SpatialAttention()

    def forward(self, x):
        return self.ch_atten3D(self.conv3D(x)) + self.ch_atten2D(self.conv2D(x))
        # return self.sp_atten3D(self.conv3D(x)) + self.sp_atten2D(self.conv2D(x))


class Conv1D(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=(3, 1, 1), s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.conv = nn.Conv3d(c1, c2, k, (1, s, s), autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.depth = depth

    def forward(self, x):
        """Apply convolution, batch normalization and activation to iput tensor."""
        x = self.act(self.bn(self.conv(x)))
        return x


class Conv_triplet1(nn.Module):
    # class MixDonwSample(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.depth = depth

        self.conv1D_hw = Conv1D(c1, c2, k=(k, 1, 1), s=1)
        self.conv1D_wd = Conv1D(c1, c2, k=(k, 1, 1), s=1)
        self.conv1D_dh = Conv1D(c1, c2, k=(k, 1, 1), s=1)
        self.ch_atten_hw = ChannelAttention(c2)
        self.ch_atten_wd = ChannelAttention(c2)
        self.ch_atten_dh = ChannelAttention(c2)
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to iput tensor."""
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        ### x = x.reshape(-1, self.depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        x = x.reshape(1, -1, C, H, W)  # 测试和验证用
        identity = x
        # 深度高度方向
        x_dh = identity.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize,  c, depth,H, W]
        x_dh = self.conv1D_dh(x_dh)
        x_dh = x_dh.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize, depth, c, H, W]
        C, H, W = x_dh.shape[2], x_dh.shape[3], x_dh.shape[4]
        x_dh = x_dh.reshape(-1, C, H, W)  # 形状变为 [batchsize*12, c, H, W]
        # 宽高方向
        x_hw = identity.permute(0, 2, 3, 4, 1)  # 形状变为 [batchsize,  c,H, W, depth]
        x_hw = self.conv1D_hw(x_hw)
        x_hw = x_hw.permute(0, 4, 1, 2, 3)  # 形状变为 [batchsize, depth, c, H, W]
        C, H, W = x_hw.shape[2], x_hw.shape[3], x_hw.shape[4]
        x_hw = x_hw.reshape(-1, C, H, W)  # 形状变为 [batchsize*12, c, H, W]
        # 宽度深度方向
        x_wd = identity.permute(0, 2, 4, 1, 3)  # 形状变为 [batchsize,  c, W, depth,H]
        x_wd = self.conv1D_wd(x_wd)
        x_wd = x_wd.permute(0, 3, 1, 4, 2)  # 形状变为 [batchsize, depth, c, H, W]
        C, H, W = x_wd.shape[2], x_wd.shape[3], x_wd.shape[4]
        x_wd = x_wd.reshape(-1, C, H, W)  # 形状变为 [batchsize*12, c, H, W]

        return self.downSample(self.ch_atten_hw(x_hw) + self.ch_atten_wd(x_wd) + self.ch_atten_dh(x_dh))


class Conv1D_2(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, (k, 1), s, (1, 0), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()
        # 普通卷积层，用于生成偏移量（offset）
        self.offset_conv = nn.Conv2d(
            in_channels,  # 输入通道数
            2 * kernel_size * kernel_size,  # 输出通道数为 2 * kernel_size^2（每个位置有 x 和 y 两个方向的偏移）
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )
        # 可变形卷积层
        self.deform_conv = ops.DeformConv2d(
            in_channels,  # 输入通道数
            out_channels,  # 输出通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)
        # 应用可变形卷积
        out = self.deform_conv(x, offset)
        return self.act(self.bn(out))


import torch
import torch.nn as nn


class DepthwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k=3,stride=1, padding=(1, 1, 1)):
        super().__init__()
        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=k,  # 核形状 [depth, height, width]
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键：每个输入通道独立卷积
            bias=False
        )
        # 逐点卷积 (Pointwise Convolution)
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,  # 1x1x1核用于通道融合
            bias=False
        )
        # 标准化与激活
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
class MixDonwSample2(nn.Module):
# class Conv_triplet(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.depth = depth
        self.c_in=c1
        self.c_out=c2
        self.conv_hw = Conv(self.c_in, self.c_out, k=3)

        self.conv_wd =  DepthwiseConv3d(self.c_in, self.c_out, k=(3,1,3),padding=(1, 0, 1))
        self.conv_dh =  DepthwiseConv3d(self.c_in, self.c_out, k=(3,3,1),padding=(1, 1, 0))

        self.ch_atten_hw = ChannelAttention(c2)
        self.ch_atten_wd = ChannelAttention(c2)
        self.ch_atten_dh = ChannelAttention(c2)
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to iput tensor."""
        # # x:[batchsize*depth, c, H, W]
        # C, H, W = x.shape[1], x.shape[2], x.shape[3]
        # # depth = x.shape[0]# 测试和验证用
        # depth =self.depth
        # # 高方向
        # x_hw = self.conv_hw(x)
        # # 宽方向
        # x_wd = x.reshape(-1, depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # 形状变为 [batchsize, H, c, depth, W]
        # x_wd = x_wd.reshape(-1, C, depth, W)  # 形状变为 [ batchsize*H , c, depth, W]
        # x_wd = self.conv_wd(x_wd)
        # x_wd = x_wd.reshape(-1,H, self.c_out, depth, W)  # 形状变为 [ batchsize, H , c, depth, W]
        # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # [ batchsize, depth, c, H , W]
        # x_wd = x_wd.reshape(-1,self.c_out, H , W)
        # # 深度高度方向
        # x_dh = x.reshape(-1, depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize, w, c, H, depth]
        # x_dh = x_dh.reshape(-1, C, H, depth)  # 形状变为 [batchsize*W,  c,H, depth]
        # x_dh = self.conv_dh(x_dh)
        # x_dh = x_dh.reshape(-1, W,self.c_out, H, depth)  # 形状变为 [batchsize,  W,  c,H, depth]
        # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize,  depth,c, H，W]
        # x_dh = x_dh.reshape(-1, self.c_out, H , W)

        x_hw = self.conv_hw(x)  # 输出形状: [batchsize*depth, self.c_out, H, W]

        B_times_D, C, H, W = x.shape
        depth = self.depth
        # # depth = x.shape[0]# 测试和验证用
        B = B_times_D // depth
        # 高方向
        # 转换为5D张量 [B, D, C, H, W] -> [B, C, D, H, W]
        x_3d = x.view(B, self.depth, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # 显式保证内存连续

        # 3D卷积操作
        x_wd = self.conv_wd(x_3d)  #
        x_dh = self.conv_dh(x_3d)  # 输出形状 [B, C_out, D, H, W]

        # 恢复原始形状
        x_wd = x_wd.permute(0, 2, 1, 3, 4).contiguous().view(B * self.depth, -1, H, W)   # [B, D, C_out, H, W]
         # [B*D, C_out, H, W]
        x_dh = x_dh.permute(0, 2, 1, 3, 4).contiguous().view(B * self.depth, -1, H, W)   # [B, D, C_out, H, W]
        # [B*D, C_out, H, W]

        return self.downSample(self.ch_atten_hw(x_hw) + self.ch_atten_wd(x_wd) + self.ch_atten_dh(x_dh))

class MixDonwSample(nn.Module):
# class Conv_triplet(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.depth = depth
        self.c_in=c1
        self.c_out=c2
        self.conv_hw = Conv(self.c_in, self.c_out, k=3)

        self.conv_wd =  DepthwiseConv3d(self.c_out, self.c_out, k=(3,1,3),padding=(1, 0, 1))
        self.conv_dh =  DepthwiseConv3d(self.c_out, self.c_out, k=(3,3,1),padding=(1, 1, 0))

        self.ch_atten_hw = ChannelAttention(c2)
        self.ch_atten_wd = ChannelAttention(c2)
        self.ch_atten_dh = ChannelAttention(c2)
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to iput tensor."""
        # # x:[batchsize*depth, c, H, W]
        # C, H, W = x.shape[1], x.shape[2], x.shape[3]
        # # depth = x.shape[0]# 测试和验证用
        # depth =self.depth
        # # 高方向
        # x_hw = self.conv_hw(x)
        # # 宽方向
        # x_wd = x.reshape(-1, depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # 形状变为 [batchsize, H, c, depth, W]
        # x_wd = x_wd.reshape(-1, C, depth, W)  # 形状变为 [ batchsize*H , c, depth, W]
        # x_wd = self.conv_wd(x_wd)
        # x_wd = x_wd.reshape(-1,H, self.c_out, depth, W)  # 形状变为 [ batchsize, H , c, depth, W]
        # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # [ batchsize, depth, c, H , W]
        # x_wd = x_wd.reshape(-1,self.c_out, H , W)
        # # 深度高度方向
        # x_dh = x.reshape(-1, depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize, w, c, H, depth]
        # x_dh = x_dh.reshape(-1, C, H, depth)  # 形状变为 [batchsize*W,  c,H, depth]
        # x_dh = self.conv_dh(x_dh)
        # x_dh = x_dh.reshape(-1, W,self.c_out, H, depth)  # 形状变为 [batchsize,  W,  c,H, depth]
        # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize,  depth,c, H，W]
        # x_dh = x_dh.reshape(-1, self.c_out, H , W)

        x_hw = self.conv_hw(x)  # 输出形状: [batchsize*depth, self.c_out, H, W]
        x_hw=self.downSample(x_hw)
        B_times_D, C, H, W = x_hw.shape
        depth = self.depth
        # depth = x_hw.shape[0]# 测试和验证用
        B = B_times_D // depth
        # 高方向
        # 转换为5D张量 [B, D, C, H, W] -> [B, C, D, H, W]
        x_3d= x_hw.view(B, self.depth, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # 显式保证内存连续

        # 3D卷积操作
        x_wd = self.conv_wd(x_3d)  #
        x_dh = self.conv_dh(x_3d)  # 输出形状 [B, C_out, D, H, W]

        # 恢复原始形状
        x_wd = x_wd.permute(0, 2, 1, 3, 4).contiguous().view(B * self.depth, -1, H, W)   # [B, D, C_out, H, W]
         # [B*D, C_out, H, W]
        x_dh = x_dh.permute(0, 2, 1, 3, 4).contiguous().view(B * self.depth, -1, H, W)   # [B, D, C_out, H, W]
        # [B*D, C_out, H, W]

        return self.ch_atten_wd(x_wd) + self.ch_atten_dh(x_dh)+self.ch_atten_hw(x_hw)


class Conv3D(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.conv = nn.Conv3d(c1, c2, k, (1, s, s), autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.depth = depth


    def forward(self, x):
        """Apply convolution, batch normalization and activation to iput tensor."""
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        if self.training:
            x = x.reshape(-1, self.depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
        else:
            x = x.reshape(1, -1, C, H, W)  # 测试和验证用
        x = x.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize,  c, depth,H, W]
        x = self.act(self.bn(self.conv(x)))
        x = x.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize, 12, c, H, W]
        C, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.reshape(-1, C, H, W)  # 形状变为 [batchsize*12, c, H, W]
        #
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Conv3D_total3D(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, depth=6):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 步长在深度上是1，在高度和宽度上都是2
        self.conv = nn.Conv3d(c1, c2, k, (1, s, s), autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.depth = depth

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        if len(x.shape) == 4:
            C, H, W = x.shape[1], x.shape[2], x.shape[3]
            # x = x.reshape(-1, self.depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
            x = x.reshape(1, -1, C, H, W)  # 测试和验证用
            x = x.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize,  c, depth,H, W]

        x = self.act(self.bn(self.conv(x)))
        # x = x.permute(0, 2, 1, 3, 4)  # 形状变为 [batchsize, 12, c, H, W]
        # C, H, W = x.shape[2], x.shape[3], x.shape[4]
        # x = x.reshape(-1, C, H, W)  # 形状变为 [batchsize*12, c, H, W]
        #
        return x


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


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


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


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Upsample(nn.Module):  # 上采样，仅变宽高
    """Concatenate a list of tensors along dimension."""

    def __init__(self, *args):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.upsacle = nn.Upsample(scale_factor=tuple([1, 2, 2]))

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return self.upsacle(x)
