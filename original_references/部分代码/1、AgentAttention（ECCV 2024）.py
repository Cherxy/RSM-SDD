import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_  # 用于权重初始化

"""
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
论文题目：Agent Attention: On the Integration of Softmax and Linear Attention (ECCV 2024)
"""

class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子，稳定点积计算

        # QKV线性映射层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Agent相关参数
        self.agent_num = agent_num  # Agent数量
        self.window = window        # 特征图窗口尺寸

        # 特征增强模块：深度可分离卷积用于局部信息补充
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)

        # Agent生成模块：自适应平均池化生成固定尺寸Agent表示
        pool_size = int(agent_num ** 0.5)  # 例如，49的平方根为7
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # 第一阶段注意力的位置编码参数
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # Agent-Token块偏置
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))  # Agent-Token行偏置
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))  # Agent-Token列偏置

        # 第二阶段注意力的位置编码参数
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # Token-Agent块偏置
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))  # Token-Agent行偏置
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))  # Token-Agent列偏置

        # 初始化所有位置编码参数
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (num_windows*B, N, C)
            B: 批次大小
            N: Token数量 (H*W)
            C: 通道数
        """
        b, n, c = x.shape
        h = w = int(n ** 0.5)  # 假设输入为正方形Token序列
        num_heads = self.num_heads
        head_dim = c // num_heads

        # 1. 生成QKV
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 生成Agent Token（使用Q的局部信息）
        q_t = q.reshape(b, h, w, c).permute(0, 3, 1, 2)
        agent_tokens = self.pool(q_t).reshape(b, c, -1).permute(0, 2, 1)

        # 3. 重塑Q, K, V为多头格式
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, num_heads, A, head_dim)

        # 4. 第一次注意力：Agent从Token获取信息，agent_tokens作为查询
        # 第一部分位置编码
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # 第二部分位置编码
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        # 这里可以添加 attn_drop，如果需要防止过拟合
        agent_v = agent_attn @ v  # Agent从Token中汇聚Value信息

        # 5. 第二次注意力：Token从Agent获取信息，agent_tokens作为键
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = q_attn  # 若有Dropout，可在此应用
        x = q_attn @ agent_v  # Token利用Agent信息增强

        # 6. 特征增强与输出
        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    # 假设输入形状为 (num_windows*B, N, C), 例如 (1, 196, 768)
    X = torch.randn(1, 196, 768)
    Model = AgentAttention(dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                           agent_num=49, window=14)
    out = Model(X)
    print(out.shape)
    # ai缝合大王
