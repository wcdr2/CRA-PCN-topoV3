# models/topo_module_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意：dual_graph_convolution.py 和 wConv.py 在工程根目录
from dual_graph_convolution import DGC
from wConv import wConv1d


class TopoReasoningBlockV2(nn.Module):
    """
    DGC(双图卷积) + wConv(小波卷积) 的拓扑推理块。

    输入:
        xyz:         (B, N, 3)   当前层点坐标 (这里是 p2)
        feat:        (B, N, C)   对应的局部语义特征（来自 PN / Decoder）
        global_feat: (B, Cg)     编码器输出的全局特征

    输出:
        delta_raw:   (B, N, 3)   原始位移（未缩放、未做零均值）
        score:       (B, N, 1)   每个点的置信度 (0~1)
        topo_feat:   (B, N, H)   拓扑推理后的隐藏特征，用于 Feature Consistency Loss
    """
    def __init__(self,
                 feat_dim,          # feat 通道数 C
                 global_dim,        # global_feat 通道数 Cg
                 hidden_dim=128,    # 图上隐藏特征维度 H
                 k=16,
                 wave_kernel_size=5,
                 num_layers=3):     # 多层 DGC（默认为 3 层）
        super().__init__()

        self.k = k
        self.num_layers = num_layers

        # 1. 多层 DGC 双图卷积：每层一套独立参数
        self.dgc_layers = nn.ModuleList([
            DGC(input_features_dim=feat_dim if i == 0 else hidden_dim)
            for i in range(num_layers)
        ])

        # 2. 第一层之后，需要把通道数对齐到 hidden_dim
        self.proj_in = nn.Conv1d(feat_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        # 3. 小波卷积，提取高频
        den = [0.5, 0.75]
        padding = wave_kernel_size // 2
        self.wconv = wConv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=wave_kernel_size,
            den=den,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn_wave = nn.BatchNorm1d(hidden_dim)

        # 4. 全局特征映射到同一维度
        self.global_proj = nn.Linear(global_dim, hidden_dim)

        # 5. 输出头：位移 + 置信度
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, xyz, feat, global_feat):
        """
        xyz:  (B, N, 3)
        feat: (B, N, C)
        """
        B, N, C = feat.shape

        # --- 1. 调整到 DGC 输入格式 ---
        xyz_t = xyz.permute(0, 2, 1).contiguous()   # (B,3,N)
        feat_t = feat.permute(0, 2, 1).contiguous() # (B,C,N)

        # --- 2. 第一层：feat_dim -> hidden_dim ---
        h = self.proj_in(feat_t)                    # (B,H,N)
        h = self.bn_in(h)
        h = F.relu(h, inplace=True)

        # --- 3. 多层 DGC 图卷积（每层独立参数） ---
        for i, dgc in enumerate(self.dgc_layers):
            # 第 1 层：输入 feat_t；后续层：输入上一层的 h
            if i == 0:
                h_in = feat_t
            else:
                h_in = h
            h_msg = dgc(xyz_t, h_in, k=self.k)      # (B,*,N)
            # 确保通道维为 hidden_dim
            if h_msg.shape[1] != h.shape[1]:
                # 保险起见，再做一次 1x1 卷积对齐
                h_msg = F.conv1d(h_msg, self.proj_in.weight, bias=None)
            h = h + h_msg                           # 残差累加
            h = F.relu(h, inplace=True)

        # --- 4. 小波卷积提取高频，并与低频相加 ---
        w = self.wconv(h)                           # (B,H,N)
        w = self.bn_wave(w)
        w = F.relu(w, inplace=True)

        h = h + w                                   # (B,H,N)
        h = h.permute(0, 2, 1).contiguous()        # (B,N,H)
        topo_feat = h                               # 拓扑特征（用于特征一致性损失）

        # --- 5. 融合 global feature ---
        g_global = self.global_proj(global_feat)    # (B,H)
        g_global = g_global.unsqueeze(1).expand(-1, N, -1)  # (B,N,H)

        h_all = torch.cat([h, g_global], dim=-1)    # (B,N,2H)

        # --- 6. 预测 Δx & score ---
        delta_raw = self.delta_head(h_all)          # (B,N,3)
        score = self.score_head(h_all)              # (B,N,1)

        return delta_raw, score, topo_feat
