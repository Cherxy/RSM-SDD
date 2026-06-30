from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 参考论文：https://arxiv.org/pdf/2306.15988
# 标题：Asymptotic Feature Pyramid Network for Labeling Pixels and Regions（2024-1区）
# ai缝合大王

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    # 如果未提供pad参数，则依据卷积核尺寸自动计算填充量
    if not pad:
        # 当卷积核尺寸为奇数时，确保两侧填充值均等
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        # 若用户给定pad，则直接采用该数值
        pad = pad

    # 利用nn.Sequential构建一个有序层序列
    return nn.Sequential(OrderedDict([
        # 添加卷积操作
        ("conv", nn.Conv2d(in_channels=filter_in, out_channels=filter_out,
                           kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        # 添加批标准化操作
        ("bn", nn.BatchNorm2d(num_features=filter_out)),
        # 应用ReLU激活函数（原地操作）
        ("relu", nn.ReLU(inplace=True)),
    ]))

class BasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        # 调用父类 nn.Module 的构造方法
        super(BasicBlock, self).__init__()
        # 初始化第一层卷积
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=3, padding=1)
        # 设置第一层批归一化
        self.bn1 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)
        # 初始化ReLU激活模块
        self.relu = nn.ReLU(inplace=True)
        # 构造第二层卷积
        self.conv2 = nn.Conv2d(in_channels=filter_out, out_channels=filter_out, kernel_size=3, padding=1)
        # 配置第二层批归一化
        self.bn2 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)

    def forward(self, x):
        # 保存输入特征以便后续执行残差相加
        residual = x
        # 第一阶段：卷积 -> 批归一化 -> ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二阶段：卷积 -> 批归一化
        out = self.conv2(out)
        out = self.bn2(out)
        # 将原始输入添加回输出，形成残差连接
        out += residual
        # 最后进行ReLU激活，输出最终结果
        out = self.relu(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
        # 构建上采样模块：先用1x1卷积调整通道，再采用双线性插值进行放大
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),  # 采用1x1卷积调整通道
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')  # 双线性插值上采样
        )

    def forward(self, x):
        # 利用上采样模块处理输入数据
        x = self.upsample(x)
        return x

class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()
        # 构建下采样模块，通过步长为2的卷积实现2倍降采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)  # 使用2x2卷积，步长为2
        )

    def forward(self, x):
        # 利用下采样模块降低特征图分辨率
        x = self.downsample(x)
        return x

class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()
        # 构建下采样模块，通过步长为4的卷积实现4倍降采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)  # 采用4x4卷积实现步长为4的下采样
        )

    def forward(self, x):
        # 使用下采样模块对输入进行处理
        x = self.downsample(x)
        return x

class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()
        # 构建下采样模块，通过步长为8的卷积实现8倍降采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)  # 利用8x8卷积，步长设为8
        )

    def forward(self, x):
        # 经由下采样模块降低输入分辨率
        x = self.downsample(x)
        return x

class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化两个通道压缩卷积层，用于特征降维
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建权重融合层，用以整合压缩后的特征
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        # 构造融合后处理卷积层，进一步细化融合特征
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        # 分别对两个输入特征图进行通道压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        # 拼接压缩后的特征图
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        # 利用融合层计算各级别特征的权重分布
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 根据计算得到的权重，对两个特征图进行加权合并
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]
        # 经过卷积层进一步整合融合后的特征
        out = self.conv(fused_out_reduced)
        return out

class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化三个降维卷积层，对应三个输入特征图
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建融合权重层，将三个降维后的特征图整合在一起
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        # 构造后续处理卷积层，对加权融合后的特征进行优化
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        # 分别对三个输入进行通道压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        # 拼接所有压缩后的特征图
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # 计算三个特征图的权重分布
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 按权重比例加权合并三个输入特征图
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
        # 经过卷积操作进一步融合特征
        out = self.conv(fused_out_reduced)
        return out

class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化四个降维卷积层，分别处理四个输入特征图
        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建融合权重计算层，用以整合四个降维特征
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        # 构造融合后特征处理卷积层
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        # 分别对四个输入特征图进行降维处理
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        # 拼接所有降维后的特征图
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # 计算各层特征的权重
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 根据权重对四个特征图进行加权合并
        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]
        # 用卷积层进一步整合融合结果
        out = self.conv(fused_out_reduced)
        return out

import torch.nn as nn

class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        # 调用基类 nn.Module 的初始化函数
        super(BlockBody, self).__init__()
        # 设置四个1x1卷积模块，用于调整各层特征图的通道数量
        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        # 构建2倍下采样和上采样模块
        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        # 初始化两个ASFF_2模块用于特征融合
        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        # 构建两个残差块序列
        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        # 配置多个降采样和上采样操作模块
        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        # 初始化三个ASFF_3模块用于多尺度特征融合
        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        # 构造三个残差块序列
        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        # 配置更多的降采样和上采样模块以匹配不同尺度
        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        # 初始化四个ASFF_4模块用于更高层次的特征融合
        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        # 构造四个残差块序列以进一步提炼融合特征
        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        # 将输入的多尺度特征图依次拆分
        x0, x1, x2, x3 = x

        # 通过1x1卷积层对各尺度特征图进行通道调整
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        # ASFF_2特征融合：将x0与x1进行上下采样后融合
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        # 通过残差块序列进一步提炼特征
        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        # ASFF_3特征融合：将经过不同采样的x0、x1和x2融合
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        # 再次通过残差块序列进一步处理融合后的特征
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        # ASFF_4特征融合：融合x0、x1、x2和x3经过不同尺度采样的结果
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)

        # 通过最后的残差块序列进一步优化融合结果
        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        # 输出最终融合后的多尺度特征图
        return scalezero, scaleone, scaletwo, scalethree

class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],  # 各输入特征图的通道数列表
                 out_channels=256):  # 最终输出特征图的通道数
        super(AFPN, self).__init__()

        # 配置是否启用半精度(fp16)运算
        self.fp16_enabled = False

        # 使用1x1卷积层对输入特征图通道数进行压缩
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        # 利用BlockBody模块实现多尺度特征融合
        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        # 通过1x1卷积层将融合后的特征图调整至统一的out_channels
        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)  # 利用池化生成额外的下采样特征图

        # 初始化各层权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 针对卷积层
                nn.init.xavier_normal_(m.weight, gain=0.02)  # 使用Xavier正态分布进行初始化
            elif isinstance(m, nn.BatchNorm2d):  # 针对批归一化层
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # 权重采用正态分布初始化
                torch.nn.init.constant_(m.bias.data, 0.0)  # 偏置初始化为0

    def forward(self, x):
        # 拆分输入的多尺度特征图
        x0, x1, x2, x3 = x

        # 通过1x1卷积层对输入特征图进行通道压缩
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # 经过BlockBody模块实现多尺度特征融合
        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        # 用1x1卷积层将融合后的特征图调整为统一的输出通道数
        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        # 利用池化层生成额外的下采样特征
        out4 = self.conv44(out3)

        # 返回最终的多尺度融合结果
        return out0, out1, out2, out3, out4

if __name__ == "__main__":
    # 构造AFPN模型实例
    block = AFPN()

    # 生成随机输入张量1，尺寸为 [batch, 256, 200, 200]
    input1 = torch.rand(16, 256, 200, 200)
    # 生成随机输入张量2，尺寸为 [batch, 512, 100, 100]
    input2 = torch.rand(16, 512, 100, 100)
    # 生成随机输入张量3，尺寸为 [batch, 1024, 50, 50]
    input3 = torch.rand(16, 1024, 50, 50)
    # 生成随机输入张量4，尺寸为 [batch, 2048, 25, 25]
    input4 = torch.rand(16, 2048, 25, 25)

    # 将所有输入张量组合成元组
    x = (input1, input2, input3, input4)
    output = block(x)
    output1, output2, output3, output4, output5 = output
    # ai缝合大王
    # 输出各层特征图的尺寸信息
    print(output1.size())
    print(output2.size())
    print(output3.size())
    print(output4.size())
    print(output5.size())
