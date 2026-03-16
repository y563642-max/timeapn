import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """
    用于移除序列末尾多余的padding，以实现因果卷积输出长度与输入长度一致。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: shape (N, C, L_in)
        Returns:
            shape (N, C, L_out) where L_out = L_in - chomp_size
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN的基本残差模块。
    包含两层空洞因果卷积和ReLU激活、Dropout。
    如果输入输出通道数不同，会使用一个1x1卷积来匹配维度以进行残差连接。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # 移除padding，确保因果性且长度不变
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding) # 移除padding
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上述层打包成一个序列
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接的下采样（如果输入输出通道数不同）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # 权重初始化
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: shape (N, C_in, L)
        Returns:
            shape (N, C_out, L)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    """
    时间卷积网络 (TCN)。
    由多个TemporalBlock堆叠而成，每个block的dilation因子指数级增长。
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: 输入特征的维度 (通道数)。
            num_channels: 一个列表，包含每个TemporalBlock的隐藏层输出通道数。
                          列表的长度决定了TCN的层数(block数)。
                          例如: [25, 25, 25] 表示3个block，每个block输出25个通道。
            kernel_size: 卷积核大小。
            dropout: Dropout比率。
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # 空洞因子指数增长: 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1] # 第一个block的输入通道是num_inputs
            out_channels = num_channels[i]
            # 计算因果卷积所需的padding量
            # padding = (kernel_size - 1) * dilation_size ensures causal, Chomp1d removes it
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状应为 (批量大小, 输入通道数, 序列长度)
               例如: (N, C_in, L_in)
        Returns:
            输出张量，形状为 (批量大小, 最后一个block的输出通道数, 序列长度)
            例如: (N, C_out_last_block, L_in)
        """
        return self.network(x)