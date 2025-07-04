import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .conv import ChannelAttention

class SpatialAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        # 第一支路：H-W平面处理
        self.conv_hw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=kernel_size // 2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

        # 第二支路：H-D平面处理（高度-深度方向）
        self.conv_dh = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1),
                      padding=(kernel_size // 2, 0), groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

        # 第三支路：W-D平面处理（宽度-深度方向）
        self.conv_wd = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, kernel_size),
                      padding=(0, kernel_size // 2), groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
        # 先卷积，后通过平均池化实现下采样
        self.downSample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 注意力生成器
        self.ch_atten_hw = ChannelAttention(out_channels)
        self.ch_atten_wd = ChannelAttention(out_channels)
        self.ch_atten_dh = ChannelAttention(out_channels)

    def forward(self, x):
        # --- 分支1：常规H-W处理 ---
        x_hw = self.conv_hw(x)  # [B*D, C_out, H, W]
        x_hw = self.downSample(x_hw)
        batchsize_depth, C, H, W = x_hw.shape
        depth = self.depth
        # 宽方向

        if self.training:
            # 复用 x_wd1
            x_wd1 = rearrange(x_hw, '(b d) c h w -> b d c h w', d=depth)  # [batchsize, depth, c, H, W]

            # 宽方向处理
            x_wd = rearrange(x_wd1, 'b d c h w -> (b h) c d w')  # [batchsize*H, c, depth, W]
            x_wd = self.conv_wd(x_wd)  # [batchsize*H, self.c_out, depth, W]
            x_wd = rearrange(x_wd, '(b h) c_out d w -> b d c_out h w', h=H)  # [batchsize, depth, self.c_out, H, W]
            x_wd = rearrange(x_wd, 'b d c_out h w -> (b d) c_out h w')  # [batchsize*depth, self.c_out, H, W]

            # 深度高度方向处理
            x_dh = rearrange(x_wd1, 'b d c h w -> (b w) c h d')  # [batchsize*W, c, H, depth]
            x_dh = self.conv_dh(x_dh)  # [batchsize*W, self.c_out, H, depth]
            x_dh = rearrange(x_dh, '(b w) c_out h d -> b d c_out h w', w=W)  # [batchsize, depth, self.c_out, H, W]
            x_dh = rearrange(x_dh, 'b d c_out h w -> (b d) c_out h w')  # [batchsize*depth, self.c_out, H, W]
            # x_wd1 = x_hw.reshape(-1, self.depth, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
            # x_wd = x_wd1.permute(0, 3, 2, 1, 4)  # 形状变为 [batchsize, H, c, depth, W]
            # x_wd = x_wd.reshape(-1, C, depth, W)  # 形状变为 [ batchsize*H , c, depth, W]
            # x_wd = self.conv_wd(x_wd)
            # x_wd = x_wd.reshape(-1,H, C, depth, W)  # 形状变为 [ batchsize, H , c, depth, W]
            # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # [ batchsize, depth, c, H , W]
            # x_wd = x_wd.reshape(-1, C, H , W)
            # # 深度高度方向
            # x_dh = x_wd1.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize, w, c, H, depth]
            # x_dh = x_dh.reshape(-1, C, H, depth)  # 形状变为 [batchsize*W,  c,H, depth]
            # x_dh = self.conv_dh(x_dh)
            # x_dh = x_dh.reshape(-1, W,C, H, depth)  # 形状变为 [batchsize,  W,  c,H, depth]
            # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize,  depth,c, H，W]
            # x_dh = x_dh.reshape(-1, C, H , W)
        else:
            # 测试阶段,batchsize为1
            # x_wd1 = x_hw.reshape(1, -1, C, H, W)  # 形状变为 [batchsize, depth, c, H, W]
            # x_wd = x_wd1.permute(0, 3, 2, 1, 4)  # 形状变为 [batchsize, H, c, depth, W]
            # x_wd = x_wd.reshape(H, C, -1, W)  # 形状变为 [ batchsize*H , c, depth, W]
            # x_wd = self.conv_wd(x_wd)
            # x_wd = x_wd.reshape(1,H, C, -1, W)  # 形状变为 [ batchsize, H , c, depth, W]
            # x_wd = x_wd.permute(0, 3, 2, 1, 4)  # [ batchsize, depth, c, H , W]
            # x_wd = x_wd.reshape(-1, C, H , W)
            # # 深度高度方向
            # x_dh = x_wd1.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize, w, c, H, depth]
            # x_dh = x_dh.reshape(W, C, H, -1)  # 形状变为 [batchsize*W,  c,H, depth]
            # x_dh = self.conv_dh(x_dh)
            # x_dh = x_dh.reshape(1, W,C, H, -1)  # 形状变为 [batchsize,  W,  c,H, depth]
            # x_dh = x_dh.permute(0, 4, 2, 3, 1)  # 形状变为 [batchsize,  depth,c, H，W]
            # x_dh = x_dh.reshape(-1, C, H , W)
            # 复用 x_wd1
            x_wd1 = rearrange(x_hw, '(b d) c h w -> b d c h w', b=1)  # [batchsize, depth, c, H, W]

            # 宽方向处理
            x_wd = rearrange(x_wd1, 'b d c h w -> (b h) c d w')  # [batchsize*H, c, depth, W]
            x_wd = self.conv_wd(x_wd)  # [batchsize*H, self.c_out, depth, W]
            x_wd = rearrange(x_wd, '(b h) c_out d w -> b d c_out h w', h=H)  # [batchsize, depth, self.c_out, H, W]
            x_wd = rearrange(x_wd, 'b d c_out h w -> (b d) c_out h w')  # [batchsize*depth, self.c_out, H, W]

            # 深度高度方向处理
            x_dh = rearrange(x_wd1, 'b d c h w -> (b w) c h d')  # [batchsize*W, c, H, depth]
            x_dh = self.conv_dh(x_dh)  # [batchsize*W, self.c_out, H, depth]
            x_dh = rearrange(x_dh, '(b w) c_out h d -> b d c_out h w', w=W)  # [batchsize, depth, self.c_out, H, W]
            x_dh = rearrange(x_dh, 'b d c_out h w -> (b d) c_out h w')  # [batchsize*depth, self.c_out, H, W]

        return self.ch_atten_wd(x_wd) + self.ch_atten_dh(x_dh)+self.ch_atten_hw(x_hw)