import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.conv import ChannelAttention, DWConv, LightConv


class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128 * 2, output_dim=128):
        super(MLP, self).__init__()
        # 定义 MLP 层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 激活函数
        x = F.relu(self.fc2(x))  # 激活函数
        x = self.fc3(x)  # 输出层
        return x


class WindowPredictionModel(nn.Module):
    def __init__(self, in_ch):
        super(WindowPredictionModel, self).__init__()
        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出形状: [b, c, 1, 1]
        # MLP 层
        self.mlp = MLP(input_dim=in_ch, hidden_dim=in_ch * 4, output_dim=in_ch * 2)  # 输出 16 个值
        self.in_ch = in_ch

    def forward(self, x):
        # 输入形状: [b, c, h, w], c=16
        # 全局平均池化
        x = self.global_pool(x)  # 输出形状: [b, c, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [b, c], c=16
        # 通过 MLP
        x = self.mlp(x)  # 输出形状: [b, 16]
        # 拆分窗宽和窗位
        window_width = x[:, :self.in_ch]  # 前 8 个值为窗宽
        window_level = x[:, self.in_ch:]  # 后 8 个值为窗位
        return window_width, window_level  # 形状: [b, self.in_ch]


def apply_windowing_per_channel(input_tensor, window_width, window_level):
    """
    对输入数据的每个通道分别应用开窗操作。

    参数:
        input_tensor (torch.Tensor): 输入数据，形状为 [b, c, h, w]，其中 c=8。
        window_width (torch.Tensor): 窗宽，形状为 [b, 8]。
        window_level (torch.Tensor): 窗位，形状为 [b, 8]。

    返回:
        torch.Tensor: 开窗后的数据，形状为 [b, c, h, w]。
    """
    b, c, h, w = input_tensor.shape
    # assert c == 8, "输入数据的通道数必须为 8"

    # 将窗宽和窗位扩展到与输入数据相同的形状
    window_width = window_width.view(b, c, 1, 1)  # [b, 8, 1, 1]
    window_level = window_level.view(b, c, 1, 1)  # [b, 8, 1, 1]

    # 计算开窗后的数据
    output = (input_tensor - (window_level - 0.5 * window_width)) / window_width
    # 将值限制在 [0, 1] 范围内
    output = torch.clamp(output, 0, 1)
    return output  # 形状: [b, c, h, w]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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


