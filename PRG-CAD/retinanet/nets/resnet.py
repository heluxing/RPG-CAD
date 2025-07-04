import math
import random

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange

from nets.swin_transformer import SwinBasicLayer

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

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


class CRAbabckup(nn.Module):  # 两分支并联后与wh并联加通道注意力相加
    def __init__(self, in_channels, out_channels, kernel_size=3, depth=6, windowing=True):
        super().__init__()
        self.depth = depth
        self.windowing = windowing
        # 第一支路：H-W平面处理
        self.conv_hw = Conv(in_channels, out_channels, kernel_size)
        # 第二支路：H-D平面处理（高度-深度方向）
        self.conv_hd = DWConv(out_channels, out_channels, k=kernel_size)
        # 第三支路：W-D平面处理（宽度-深度方向）
        self.conv_dw = DWConv(out_channels, out_channels, k=kernel_size)
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # self.wpm1 = WindowPredictionModel(out_channels)
        # self.wpm2 = WindowPredictionModel(out_channels)
        # self.wpm3 = WindowPredictionModel(out_channels)

        # 注意力生成器
        self.ch_atten_hw = ChannelAttention(out_channels)
        self.ch_atten_dw = ChannelAttention(out_channels)
        self.ch_atten_hd = ChannelAttention(out_channels)
        self.conv_final = Conv(out_channels, out_channels, k=1)

    def forward(self, x):
        # --- 分支1：常规H-W处理 ---
        x_hw = self.conv_hw(x)  # [B*D, C_out, H, W]
        x_hw = self.downSample(x_hw)
        batchsize_depth, C, H, W = x_hw.shape
        depth = self.depth  # 后面改成随机深度
        # 宽方向
        if self.training:
            # 复用 x_dw1
            x_dw1 = rearrange(x_hw, '(b d) c h w -> b d c h w', d=depth)  # [batchsize, depth, c, H, W]
            # 宽方向处理
            x_dw = rearrange(x_dw1, 'b d c h w -> (b h) c d w')  # [batchsize*H, c, depth, W]
            x_dw = self.conv_dw(x_dw)  # [batchsize*H, self.c_out, depth, W]
            x_dw = rearrange(x_dw, '(b h) c_out d w -> (b d) c_out h w', h=H)  # [batchsize, depth, self.c_out, H, W]

            # 深度高度方向处理
            x_hd = rearrange(x_dw1, 'b d c h w -> (b w) c h d')  # [batchsize*W, c, H, depth]
            x_hd = self.conv_hd(x_hd)  # [batchsize*W, self.c_out, H, depth]
            x_hd = rearrange(x_hd, '(b w) c_out h d -> (b d) c_out h w', w=W)  # [batchsize, depth, self.c_out, H, W]
        else:
            x_dw1 = rearrange(x_hw, '(b d) c h w -> b d c h w', b=1)  # [batchsize, depth, c, H, W]
            # 宽方向处理
            x_dw = rearrange(x_dw1, 'b d c h w -> (b h) c d w')  # [batchsize*H, c, depth, W]
            x_dw = self.conv_dw(x_dw)  # [batchsize*H, self.c_out, depth, W]
            x_dw = rearrange(x_dw, '(b h) c_out d w ->(b d) c_out h w', h=H)  # [batchsize, depth, self.c_out, H, W]

            # 深度高度方向处理
            x_hd = rearrange(x_dw1, 'b d c h w -> (b w) c h d')  # [batchsize*W, c, H, depth]
            x_hd = self.conv_hd(x_hd)  # [batchsize*W, self.c_out, H, depth]
            x_hd = rearrange(x_hd, '(b w) c_out h d -> (b d) c_out h w', w=W)  # [batchsize, depth, self.c_out, H, W]

        x_hd = self.ch_atten_hd(x_hd)
        x_dw = self.ch_atten_dw(x_dw)
        x_hw = self.ch_atten_hw(x_hw)
        return self.conv_final(x_dw + x_hd + x_hw)
        # if self.windowing:
        #     window_width_wh, window_level_wh = self.wpm1(x_hw)
        #     x_hw = x_hw + apply_windowing_per_channel(x_hw, window_width_wh, window_level_wh)
        #     window_width_dw, window_level_dw = self.wpm2(x_dw)
        #     x_dw = x_dw + apply_windowing_per_channel(x_dw, window_width_dw, window_level_dw)
        #     window_width_hd, window_level_hd = self.wpm3(x_hd)
        #     x_hd = x_hd + apply_windowing_per_channel(x_hd, window_width_hd, window_level_hd)

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
        c1 = c_out // 2

        self.dw_conv = DWConv(c1, c1, 3)  # 共享DW卷积
        self.dw_conv1 = DWConv(c1, c1, 3)  # 共享DW卷积
        self.dw_conv2 = DWConv(c1, c1, 3)  # 共享DW卷积

        # 路径1组件
        self.fc_a = nn.Sequential(nn.Linear(w * 2, 256), nn.ReLU())
        self.fc_a1to6 = nn.ModuleList([nn.Linear(256, w) for _ in range(4)])
        self.multiscale1 = nn.ModuleList([
            # DWConv(c_out, c_out, 1 ),
            DWConv(c_out, c_out, 3),
            DWConv(c_out, c_out, 5)
        ])

        # 路径2组件
        self.fc_a_prime = nn.Sequential(nn.Linear(h * 2, 256), nn.ReLU())
        self.fc_a1to6_prime = nn.ModuleList([nn.Linear(256, h) for _ in range(4)])
        self.multiscale2 = nn.ModuleList([
            # DWConv(c_out,c_out, 1),
            DWConv(c_out, c_out, 3),
            DWConv(c_out, c_out, 5)
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
        c = x1.shape[1] // 2
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
        B_list = [b.view(batch_size, h, c_out, depth, w)
                      .permute(0, 3, 2, 1, 4)
                      .contiguous().view(bs_depth, c_out, h, w)
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
        B_prime_list = [b.view(batch_size, w, c_out, h, depth)
                            .permute(0, 4, 2, 3, 1)
                            .contiguous().view(bs_depth, c_out, h, w)
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
        A1_final = torch.concat((A1, A2), dim=1).view(batch_size, h, c_out, depth, w) \
            .permute(0, 3, 2, 1, 4).contiguous().view(bs_depth, c_out, h, w)
        A1_prime_final = torch.concat((A1_prime, A2_prime), dim=1).view(batch_size, w, c_out, h, depth) \
            .permute(0, 4, 2, 3, 1).contiguous().view(bs_depth, c_out, h, w)

        X1_concat = A1_final + X1
        X2_concat = A1_prime_final + X2

        X1_concat = self.ch_atten_hd(X1_concat)
        X2_concat = self.ch_atten_dw(X2_concat)
        X3 = self.ch_atten_hw(x)
        output = X1_concat + X2_concat + X3
        # return self.conv_final(x_dw + x_hd + x_hw)
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,flag=3)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,flag=4)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,flag=0):
        downsample = None
        """
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        """
        if stride != 1 or self.inplanes != planes * block.expansion:
            if flag==0:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif flag==3:
                downsample =nn.Sequential(
                    BDA(self.inplanes, self.inplanes,64, 8, 2),
                    # CRA(self.inplanes, planes * block.expansion, kernel_size=3, depth=6),
                    CRA(self.inplanes, planes * block.expansion,  h=64, w=64 ,depth=6),
                    BDA(planes * block.expansion,planes * block.expansion,32, 4, 2)
                )
            else:
                # downsample =CRA(self.inplanes, planes * block.expansion, kernel_size=3, depth=6)
                downsample =CRA(self.inplanes, planes * block.expansion,  h=32, w=32 ,depth=6)
        # """
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='model_data'), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='model_data'), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='model_data'), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='model_data'), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='model_data'), strict=False)
    return model
