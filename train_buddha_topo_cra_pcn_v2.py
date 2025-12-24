import os
import sys
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------- 基本路径设置 -----------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

# 统一佛像项目的路径
BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
sys.path.append(BUDDHA_ROOT)

from dataset_buddha import BuddhaPairDataset
from metrics import chamfer_distance, fscore

from models.crapcn import TopoCRAPCN_V2
from models.utils import query_knn, grouping_operation


# =================== 1. 拓扑相关损失 ===================

def laplacian_smooth_loss(xyz, k=16):
    """
    图拉普拉斯平滑损失：
      对每个点 i，惩罚它与邻域平均坐标的差值 ||x_i - mean_j x_j||^2

    xyz: (B, N, 3)  一般用 p2_refined
    """
    B, N, _ = xyz.shape

    # 在 xyz 上构建 kNN 图
    idx_knn = query_knn(k, xyz, xyz)                      # (B,N,k)

    # 取邻居坐标
    xyz_t = xyz.permute(0, 2, 1).contiguous()             # (B,3,N)
    neigh = grouping_operation(xyz_t, idx_knn)            # (B,3,N,k)
    neigh = neigh.permute(0, 2, 3, 1)                     # (B,N,k,3)

    # 邻域平均
    neigh_mean = neigh.mean(dim=2)                        # (B,N,3)

    diff = xyz - neigh_mean                               # (B,N,3)
    loss = (diff ** 2).sum(dim=-1).mean()
    return loss


def disp_l2_loss(delta_xyz):
    """
    位移 L2 正则：惩罚位移过大
    delta_xyz: (B, N, 3)
    """
    return (delta_xyz ** 2).sum(dim=-1).mean()


def feature_consistency_loss(feat_backbone, feat_topo):
    """
    特征一致性损失：
      约束 Topo 模块输出的 topo_feat 不要偏离 backbone 的 p2 特征。

    feat_backbone: (B,N,C)  来自 PN / Decoder 的特征（这里会 detach）
    feat_topo:     (B,N,C)  TopoReasoner 输出的 topo_feat（hidden feature）
    """
    return F.l1_loss(feat_topo, feat_backbone)


# =================== 2. 评估函数（val/test 共用） ===================

def evaluate(model, dataloader, device):
    model.eval()
    all_cd = []
    all_f = []

    with torch.no_grad():
        for batch in dataloader:
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            if partial.dim() == 2:
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

            out = model(partial, return_all=True)
            pred_dense = out["p3_refined"]   # 最终稠密点云

            cd_batch = chamfer_distance(pred_dense, gt)       # (B,)
            f_batch = fscore(pred_dense, gt, tau=0.01)        # (B,)

            all_cd += cd_batch.cpu().tolist()
            all_f += f_batch.cpu().tolist()

    mean_cd = float(np.mean(all_cd))
    mean_f = float(np.mean(all_f))
    return mean_cd, mean_f


# =================== 3. 主训练流程 ===================

def main():
    # ----------- 基本配置 -----------
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    # 已经在佛像数据上微调好的 CRA-PCN 权重
    pretrain_ckpt = "./pretrain/pcn/ckpt-buddha-ft-best.pth"
    # 带拓扑模块后的新模型
    save_ckpt = "./pretrain/pcn/ckpt-buddha-topo-ft-best.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_epochs = 50
    batch_size = 1          # 为避免 Chamfer OOM，继续用 1
    lr = 1e-4
    weight_decay = 1e-4

    # 拓扑模块参数
    topo_hidden_dim = 128
    topo_k = 16

    # 各损失权重（可以后面再细调）
    lambda_lap = 0.01       # 拉普拉斯平滑损失权重
    lambda_disp = 0.001     # 位移 L2 正则
    lambda_feat = 0.1       # 特征一致性损失

    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ----------- 数据集 & DataLoader -----------
    train_dataset = BuddhaPairDataset(
        root_dir=data_root,
        n_partial=2048,
        n_full=16384,
        split="train",
    )
    val_dataset = BuddhaPairDataset(
        root_dir=data_root,
        n_partial=2048,
        n_full=16384,
        split="val",
    )
    test_dataset = BuddhaPairDataset(
        root_dir=data_root,
        n_partial=2048,
        n_full=16384,
        split="test",
    )

    print(f"[Dataset] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # ----------- 模型：TopoCRAPCN_V2 -----------
    model = TopoCRAPCN_V2(
        topo_hidden_dim=topo_hidden_dim,
        topo_k=topo_k,
        delta_scale=0.2,   # 先试 0.2，后面可以调成 0.1 等
    ).to(device)

    # 加载 CRA-PCN 原预训练权重（只会匹配 encoder / decoder / seed 部分）
    if os.path.exists(pretrain_ckpt):
        print(f"[Init] 加载预训练权重: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        print(f"[Init] 未找到预训练权重 {pretrain_ckpt}，将完全从随机初始化开始训练")

    # ====== 冻结 CRA-PCN 主干，只训练拓扑模块 ======
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.seed_generator.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    # 只优化 topo_reasoner（位移传播没有参数）
    optim_params = [p for p in model.topo_reasoner.parameters() if p.requires_grad]
    print(f"[Init] 可训练参数数量: {sum(p.numel() for p in optim_params)}")

    optimizer = torch.optim.AdamW(
        optim_params, lr=lr, weight_decay=weight_decay
    )

    best_val_cd = float("inf")
    best_epoch = -1

    # ----------- 训练循环 -----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            if partial.dim() == 2:
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

            # 前向：return_all=True 方便取出各种中间量
            out = model(partial, return_all=True)
            pred_dense = out["p3_refined"]           # (B,N3,3)
            p2_refined = out["p2_refined"]           # (B,N2,3)
            delta_p2 = out["delta_p2"]               # (B,N2,3)
            feat_p2 = out["feat_p2_backbone"]        # (B,N2,C)
            topo_feat = out["topo_feat"]             # (B,N2,H) 一般与 C 相同

            # 基础 CD 损失
            loss_cd = chamfer_distance(pred_dense, gt).mean()

            # 拉普拉斯平滑损失（约束 p2 点云在 kNN 图上更平滑）
            loss_lap = laplacian_smooth_loss(p2_refined, k=topo_k)

            # 位移 L2 正则
            loss_disp = disp_l2_loss(delta_p2)

            # 特征一致性损失（backbone 特征不参与梯度）
            loss_feat = feature_consistency_loss(feat_p2.detach(), topo_feat)

            # 总损失
            loss = loss_cd \
                   + lambda_lap * loss_lap \
                   + lambda_disp * loss_disp \
                   + lambda_feat * loss_feat

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(1, num_batches)

        # ----------- 每个 epoch 做一次 val 评估 -----------
        val_cd, val_f = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss: {train_loss:.6f} | "
              f"val_CD: {val_cd:.6f}, val_F: {val_f:.6f}")

        # 保存 val CD 最好的模型
        if val_cd < best_val_cd:
            best_val_cd = val_cd
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_cd": best_val_cd,
                },
                save_ckpt,
            )
            print(f"  -> 新的最佳模型已保存到 {save_ckpt}")

    print(f"\n训练结束，最佳 epoch = {best_epoch}, 最佳 val CD = {best_val_cd:.6f}")

    # ----------- 加载最佳模型，在 test 集上评估 -----------
    if os.path.exists(save_ckpt):
        print(f"\n加载最佳模型 {save_ckpt} 做 test 评估...")
        best_model = TopoCRAPCN_V2(
            topo_hidden_dim=topo_hidden_dim,
            topo_k=topo_k,
            delta_scale=0.2,
        ).to(device)

        ckpt = torch.load(save_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        best_model.load_state_dict(state, strict=False)

        test_cd, test_f = evaluate(best_model, test_loader, device)
        print("========================================")
        print("[Topo-CRA-PCN Buddha Finetune - Test Set]")
        print(f"样本数: {len(test_loader.dataset)}")
        print(f"平均 Chamfer Distance: {test_cd:.6f}")
        print(f"平均 F-score@0.01:    {test_f:.6f}")
        print("========================================")
    else:
        print("没有找到保存的最佳模型，跳过 test 评估。")


if __name__ == "__main__":
    main()
