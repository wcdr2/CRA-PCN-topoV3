import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class wConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv1d, self).__init__()
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.kernel_size = _single(kernel_size)
        self.groups = groups
        self.dilation = _single(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', self.alfa)

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv1d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv2d, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv3d, self).__init__()
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.einsum('i,j,k->ijk', self.alfa, self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


if __name__ == '__main__':
    # Test wConv1d
    print("Testing wConv1d...")
    den = [0.5, 0.75]
    block = wConv1d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32).to('cuda')
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
    print()

    # Test wConv2d
    print("Testing wConv2d...")
    den = [0.5, 0.75]
    block = wConv2d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32, 32).to('cuda')
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
    print()

    # Test wConv3d
    print("Testing wConv3d...")
    den = [0.5, 0.75]
    block = wConv3d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32, 32, 32).to('cuda')
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())


"""
====================================================================
🧩 Weighted Convolution (wConv1d / wConv2d / wConv3d) 模块说明
====================================================================

这些模块是带权卷积操作的扩展版本，分别处理 1D、2D、3D 数据。通过在常规卷积核上引入加权矩阵（Φ），
实现对输入信号的区域性加权建模，可增强网络对不同区域特征的敏感度。

====================================================================
✅ 模块创新点
====================================================================

1. 空间权重调制（Weighted Kernel Modulation）:
   - 使用 den 参数生成一个对称的权重序列 `alfa`。
   - 构造出权重模板 `Phi`，并与卷积核逐元素相乘，实现位置敏感的卷积计算。

2. 多维通用性（1D/2D/3D）统一设计:
   - 分别采用 `torch.outer` 和 `torch.einsum` 构造 Phi，实现从一维到三维的结构扩展。
   - 接口风格统一，便于在不同类型的时序、图像或体积数据中调用。

3. 动态可调的空间响应（Flexible Spatial Bias）:
   - 通过传入不同的 `den` 权重密度序列，可自由调控卷积核的响应区域和重心。

4. 与标准卷积兼容（Plug-and-play）:
   - 保留了标准卷积的输入输出形式、groups、stride、padding 等参数。
   - 可无缝替代 nn.Conv 系列模块。

====================================================================
🚀 应用场景举例
====================================================================

1. 时间序列建模（wConv1d）:
   - 可用于语音识别、金融预测中的局部趋势建模。

2. 图像识别与分割（wConv2d）:
   - 在目标检测、图像分割中用于加强空间区域的特征提取。

3. 医疗影像处理（wConv3d）:
   - 对 CT/MRI 等体数据中的重要体素区域赋予更高权重，提高诊断性能。

4. 遥感与多光谱图像:
   - 加强多维图像中的信息融合，如光谱权重建模。

5. 时空建模与图结构学习:
   - 在 ST-GNN 等模型中作为时空局部卷积模块使用。

"""
