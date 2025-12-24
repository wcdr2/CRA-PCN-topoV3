import torch 
import numpy as np
import ot 
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Aggregates the sliced transport plans between keys/queries
# L = # of slicers
class GammaAggregator(nn.Module):
    def __init__(self, temperature=0.1):
        super(GammaAggregator, self).__init__()
        self.temperature = temperature 

    def forward(self, Gamma, x=None, y=None):
        cost = torch.cdist(x, y, p=2)
        swds = (cost.unsqueeze(2) * Gamma).sum(dim=(-1, -2)) 
        min_swds = swds.min(dim=-1, keepdim=True).values 
        exp_swds = torch.exp(-self.temperature * (swds - min_swds)) #higher temp => lower cost transport plans are favored; temp = 0 => mean
        weights = exp_swds / exp_swds.sum(dim=-1, keepdim=True) 
        Gamma_weighted = Gamma * weights.unsqueeze(-1).unsqueeze(-1)
        return Gamma_weighted.sum(dim=2)  # Sum over slices, shape: [B, H, N, N]


class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=1e-3, hard=False):
        super(SoftSort_p2, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        scores = scores.transpose(3,2).unsqueeze(-1) # Shape: B x H x N x L -> B x H x L x N -> B x H x L x N x 1
        sorted_scores, _ = scores.sort(dim=3, descending=True)  # Shape: B x H x L x N x 1
        pairwise_diff = ((scores.transpose(4, 3) - sorted_scores) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(dim=-1)
        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat 
        return P_hat.squeeze(-1)


# Attention with key padding mask
class Esp(nn.Module):
    def __init__(self, d_in, heads=8, tau=1e-3, interp=None, temperature=0.1):
        super(Esp, self).__init__()

        self.softsort = SoftSort_p2(tau=tau)
        self.interp = interp
        self.aggregator = GammaAggregator(temperature=temperature)
            
    def forward(self, X, Y, mask=None):  # Keys, Queries (B, H, N, L)
        B, H, N, L = X.shape

        if mask is not None:
            # We assume a mask of shape (B, N), where 1 means padded, 0 means not padded
            expanded_mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, L)
            fill_value = 1e+18
            # Apply large value to padded elements to push them to the end after sorting
            X = torch.where(expanded_mask, torch.full_like(X, fill_value), X)
            Y = torch.where(expanded_mask, torch.full_like(Y, fill_value), Y)
        
        Pu = self.softsort(X)  
        Pv = self.softsort(Y)  
    
        # Compute Gamma
        if self.interp is None:
            # Sliced OT via dot product of soft-sorted projections
            Gamma = Pu.transpose(-1, -2) @ Pv  # Shape: [B, H, L, N, N]
        else:
            # Sliced OT with interpolation matrix
            interp_expanded = self.interp.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1).to(X.device)
            Pu = Pu.unsqueeze(-1) if Pu.shape[-1] == 1 else Pu
            Pv = Pv.unsqueeze(-1) if Pv.shape[-1] == 1 else Pv
            Gamma = Pu.transpose(-1, -2) @ interp_expanded @ Pv
    
        # Aggregate Gamma
        Gamma_hat = self.aggregator(Gamma, x=X, y=Y)
        return Gamma_hat




class EspAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 interp=None, learnable=True, temperature=10,
                 qkv_bias=False, max_points: int = 1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_points = max_points  # 🔹 限制参与 OT 的最大点数

        # Initialize Esp module
        self.esp = Esp(d_in=dim_head, heads=heads, interp=interp, temperature=temperature)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward_full(self, x, mask=None):
        """
        原始完整版本：对所有点做 ESP（可能 OOM）
        仅在点数较小或调试时使用。
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if mask is not None:
            if mask.dim() not in [2]:
                raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected 2D tensor of shape (B, N).")

        attn = self.esp(q * self.scale, k * self.scale, mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

    def forward(self, x, mask=None):
        """
        子采样版本：当点数 N > max_points 时，只在子集上做 OT，
        再把结果 scatter 回去，避免 N^2 爆显存。
        """
        b, n, c = x.shape

        # 如果点数不大，直接走原始版本
        if (self.max_points is None) or (n <= self.max_points):
            return self.forward_full(x, mask)

        # -------- 1) 子采样点索引 --------
        # 简单做一个全局随机采样：对所有 batch 复用同一组索引即可
        idx = torch.randperm(n, device=x.device)[: self.max_points]  # (N_sub,)
        x_sub = x[:, idx, :]                                        # (B, N_sub, C)

        if mask is not None:
            if mask.dim() != 2 or mask.shape[0] != b or mask.shape[1] != n:
                raise ValueError(f"Unexpected mask shape: {mask.shape}, expected (B, N)")
            mask_sub = mask[:, idx]                                 # (B, N_sub)
        else:
            mask_sub = None

        # -------- 2) 在子集上做 ESP 注意力 --------
        qkv = self.to_qkv(x_sub).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attn = self.esp(q * self.scale, k * self.scale, mask_sub)   # (B, H, N_sub, N_sub)
        out_sub = torch.matmul(attn, v)                             # (B, H, N_sub, d)
        out_sub = rearrange(out_sub, 'b h n d -> b n (h d)')        # (B, N_sub, C)
        out_sub = self.to_out(out_sub)                              # (B, N_sub, C)

        # -------- 3) scatter 回完整点云 --------
        # 关键改动：未采样点不再是 0，而是保留原输入特征 x（identity）
        out = x.clone()                                # (B, N, C)
        out[:, idx, :] = out_sub

        
        return out, attn



"""
====================================================================
🧩 ESPAttention 模块说明
====================================================================

ESPAttention (Exact Sliced-Wasserstein Attention) 是一种基于最优传输 (Optimal Transport, OT) 理论的新型注意力机制。
它通过**切片 Wasserstein 距离**来计算查询 (query) 和键 (key) 之间的相似性，从而取代传统的点积注意力。这种方法能
更准确地捕捉高维向量空间中的结构信息和全局关系。

====================================================================
✅ 模块创新点
====================================================================

1. **最优传输注意力**: 
   - 传统注意力机制使用点积来衡量相似性，这在高维空间中可能失效（"curse of dimensionality"）。
   - ESPAttention 采用 **切片 Wasserstein 距离** (Sliced Wasserstein Distance, SWD) 作为相似性度量。SWD 通过将高维
     数据投影到多个一维空间上，计算这些一维投影的 Wasserstein 距离，然后进行聚合。
   - 这使得注意力机制能够学习到更丰富的几何结构和数据分布信息。

2. **可微分的排序与传输**:
   - 核心组件是 `SoftSort_p2`，它实现了可微分的软排序。这允许模型通过梯度下降来学习排序和排列。
   - `Pu` 和 `Pv` 分别是查询和键的软排序排列矩阵，它们构成了切片最优传输的"传输计划" (`Gamma`)。

3. **智能的传输计划聚合**:
   - `GammaAggregator` 模块负责聚合来自不同切片的传输计划 `Gamma`。
   - 它使用切片 Wasserstein 距离 (SWD) 作为成本函数，并根据这些成本为每个切片分配权重。成本越低的切片（即投影距离越短），
     其对应的传输计划权重越高，从而更有效地指导注意力计算。

4. **与标准 Transformer 兼容**:
   - ESPAttention 模块的输入和输出格式与标准的 `Multi-Head Attention` 模块完全兼容。
   - 它可以无缝地集成到现有的 Transformer 架构中，例如在 ViT (Vision Transformer) 或各种 NLP 模型中替代自注意力层。

====================================================================
🚀 应用场景举例
====================================================================

1. **计算机视觉**:
   - 在图像识别、目标检测等任务中，用于捕捉图像块 (patch) 之间复杂的空间关系。由于 SWD 对高维特征更鲁棒，
     能更好地处理图像中多样的特征分布。

2. **自然语言处理**:
   - 在处理长文本或复杂句法结构时，ESPAttention 可以更准确地建模词汇之间的语义关系，特别是对于那些传统点积
     可能难以捕捉的隐式联系。

3. **多模态学习**:
   - 在需要融合不同模态（如图像和文本）信息的任务中，ESPAttention 可以用来计算不同模态特征之间的最优传输，
     从而实现更有效的特征对齐和信息融合。

4. **图神经网络**:
   - 在图注意力网络中，ESPAttention 可用于计算节点特征之间的注意力权重，更好地处理复杂的图结构。

"""