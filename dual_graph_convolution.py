import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Finds the K nearest neighbors for each point in a point cloud.
    
    Args:
        x (torch.Tensor): Input point cloud tensor of shape (batch_size, num_dims, num_points).
        k (int): Number of neighbors to find.
        
    Returns:
        torch.Tensor: Tensor of neighbor indices of shape (batch_size, num_points, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_neighbors(x, feature, k=20, idx=None):
    """
    Finds the K-nearest neighbors for a point cloud and its features,
    and constructs neighbor-centric representations.
    
    Args:
        x (torch.Tensor): Point cloud coordinates of shape (B, 3, N).
        feature (torch.Tensor): Point features of shape (B, C, N).
        k (int): Number of neighbors.
        idx (torch.Tensor, optional): Pre-computed neighbor indices.
        
    Returns:
        tuple: A tuple containing:
            - neighbor_x (torch.Tensor): Concatenated relative and absolute coordinates
              of neighbors, shape (B, 6, N, K).
            - neighbor_feat (torch.Tensor): Concatenated relative and absolute features
              of neighbors, shape (B, 2C, N, K).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x_tiled = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Concatenate relative coordinates (neighbor - self) and absolute coordinates (self)
    neighbor_x = torch.cat((neighbor_x - x_tiled, x_tiled), dim=3).permute(0, 3, 1, 2)

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2,1).contiguous()
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    feature_tiled = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Concatenate relative features (neighbor - self) and absolute features (self)
    neighbor_feat = torch.cat((neighbor_feat - feature_tiled, feature_tiled), dim=3).permute(0, 3, 1, 2)

    return neighbor_x, neighbor_feat


class DGC(nn.Module):
    """
    Dynamic Graph Convolution (DGC) module for point cloud feature extraction.
    
    This module creates two types of graphs—one based on geometry (coordinates) and one based on features—
    to learn a comprehensive local representation for each point.
    """
    def __init__(self, input_features_dim):
        """
        Initializes the DGC module.
        
        Args:
            input_features_dim (int): The number of input feature dimensions per point.
        """
        super(DGC, self).__init__()

        # MLP for processing geometric graph features
        self.conv_mlp1 = nn.Conv2d(6, input_features_dim // 2, 1)
        self.bn_mlp1 = nn.BatchNorm2d(input_features_dim // 2)

        # MLP for processing feature graph features
        self.conv_mlp2 = nn.Conv2d(input_features_dim * 2, input_features_dim // 2, 1)
        self.bn_mlp2 = nn.BatchNorm2d(input_features_dim // 2)


    def forward(self, xyz, features, k):
        """
        Forward pass of the DGC module.
        
        Args:
            xyz (torch.Tensor): Point cloud coordinates of shape (B, 3, N).
            features (torch.Tensor): Point features of shape (B, C, N).
            k (int): Number of neighbors to consider.
            
        Returns:
            torch.Tensor: Learned graph-encoded features of shape (B, C, N).
        """
        # Step 1: Find neighbors and create local representations
        neighbor_xyz, neighbor_feat = get_neighbors(xyz, features, k=k)

        # Step 2: Process geometric graph information
        # Input shape: (B, 6, N, K) -> Output shape: (B, C/2, N, K)
        geometric_encoding = F.relu(self.bn_mlp1(self.conv_mlp1(neighbor_xyz)))

        # Step 3: Process feature graph information
        # Input shape: (B, 2C, N, K) -> Output shape: (B, C/2, N, K)
        feature_encoding = F.relu(self.bn_mlp2(self.conv_mlp2(neighbor_feat)))

        # Step 4: Concatenate and perform max pooling
        # Concatenate: (B, C/2, N, K) + (B, C/2, N, K) -> (B, C, N, K)
        graph_encoding = torch.cat((geometric_encoding, feature_encoding), dim=1)
        
        # Max pool over the neighbor dimension to get a single feature vector for each point
        # Output shape: (B, C, N)
        graph_encoding = graph_encoding.max(dim=-1, keepdim=False)[0]

        return graph_encoding

"""
====================================================================
🧩 DGC (Dynamic Graph Convolution) 模块说明
====================================================================

DGC (Dynamic Graph Convolution) 模块是一种专门为点云数据设计的特征提取层。它通过构建动态图来捕获点云中
每个点的局部几何和特征上下文。与传统的卷积不同，DGC 不依赖于网格结构，而是**动态地为每个点寻找其邻居**，
然后将邻居信息汇聚到中心点。

====================================================================
✅ 模块创新点
====================================================================

1. **双重图结构学习**:
   - DGC 模块同时利用**几何图**和**特征图**来学习点云的表示。
   - **几何图**: 基于点的三维坐标 (`xyz`)，通过 `get_neighbors` 函数计算每个点与其 k 个最近邻点之间的相对位置和绝对位置，
     从而捕获局部空间结构。
   - **特征图**: 基于点的特征 (`features`)，通过 `get_neighbors` 函数计算每个点与其 k 个最近邻点之间的相对特征和绝对特征，
     从而捕获局部特征相似性。
   - 这种双重机制使模型能够同时理解点的空间排列和特征分布。

2. **动态邻居搜索**:
   - 与预定义的图结构不同，DGC 在每次前向传播时都通过 `knn` (K-Nearest Neighbors) 算法**动态地**为每个点寻找邻居。
   - 这种动态性使得网络能够适应点云不规则的稀疏性，并避免了因固定邻域而导致的偏差。

3. **Max Pooling 聚合**:
   - 在处理完每个邻居的几何和特征编码后，DGC 使用 `max` 函数在邻居维度上进行池化 (`max(dim=-1)`)。
   - 这种聚合操作可以有效地**捕获邻域中最具代表性的特征**，并对邻居点的顺序不敏感，这是点云处理中的一个重要特性。

4. **输入特征与维度拼接**:
   - `get_neighbors` 函数不仅返回邻居的坐标和特征，还将**相对信息**（邻居减去中心点）和**绝对信息**（中心点本身）
     进行拼接。这种设计为后续的 MLP 提供了更丰富和更具判别力的输入。

====================================================================
🚀 应用场景举例
====================================================================

1. **点云分类**:
   - DGC 可以作为点云分类网络中的一个基础层，例如在 PointNet++ 的基础上，替换或增强其分组和特征提取模块，
     以更好地捕捉局部特征。

2. **点云分割**:
   - 在点云语义分割、实例分割等任务中，DGC 能够为每个点学习到包含其邻域信息的丰富特征，这对于准确地
     将点分配到正确的类别至关重要。

3. **三维物体检测与识别**:
   - 它可以用于从原始点云中提取物体特征，作为三维检测器的骨干网络。

4. **与Transformer结合**:
   - DGC 提取的局部特征可以作为 Transformer 编码器的输入，进一步学习点云的全局关系。这种混合架构结合了局部
     细节和全局上下文，在许多任务中表现出色。
"""