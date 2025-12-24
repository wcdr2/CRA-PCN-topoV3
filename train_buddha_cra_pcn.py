import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----------------- 基本路径设置 -----------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

# 统一佛像工程的路径 ★★ 如果你放的位置不一样，这里要改 ★★
BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
sys.path.append(BUDDHA_ROOT)

from dataset_buddha import BuddhaPairDataset
from metrics import chamfer_distance, fscore

from models.crapcn import TopoCRAPCN


# =================== 1. 模型封装 ===================

class CRAPCNWrapper(nn.Module):
    def __init__(self,
                 ckpt_path="./pretrain/pcn/ckpt-best.pth",
                 device="cuda:0",
                 eval_mode=False):
        super().__init__()
        self.device = torch.device(device)
        self.model =  TopoCRAPCN(
                topo_hidden_dim=128,
                topo_k=16,
                topo_layers=2
            ).to(self.device)

        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_ckpt(ckpt_path)
        else:
            print(f"[CRAPCNWrapper] 注意: 找不到预训练权重 {ckpt_path}, 将随机初始化")

        if eval_mode:
            self.model.eval()

    def _load_ckpt(self, ckpt_path):
        print(f"[CRAPCNWrapper] 加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        if missing:
            print("[CRAPCNWrapper] Warning: missing keys:", len(missing))
        if unexpected:
            print("[CRAPCNWrapper] Warning: unexpected keys:", len(unexpected))

    def forward(self, partial_pc: torch.Tensor) -> torch.Tensor:
        """
        partial_pc: (B, N, 3)
        return:     (B, M, 3)
        """
        assert partial_pc.dim() == 3 and partial_pc.size(-1) == 3
        x = partial_pc.to(self.device)

        out = self.model(x)      # CRA-PCN 返回多级输出 list/tuple
        pred = out[-1]           # 取最稠密一级 (B,M,3) 或 (B,3,M)

        if pred.dim() == 3 and pred.size(1) == 3 and pred.size(2) != 3:
            pred = pred.transpose(1, 2).contiguous()

        return pred


# =================== 2. 评估函数（val/test 共用） ===================

def evaluate(model: CRAPCNWrapper, dataloader, device):
    model.eval()
    all_cd = []
    all_f = []
    with torch.no_grad():
        for batch in dataloader:
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            if partial.dim() == 2:  # 单样本防护
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

            pred = model(partial)

            cd_batch = chamfer_distance(pred, gt)      # (B,)
            f_batch = fscore(pred, gt, tau=0.01)       # (B,)

            all_cd += cd_batch.cpu().tolist()
            all_f += f_batch.cpu().tolist()

    mean_cd = float(np.mean(all_cd))
    mean_f = float(np.mean(all_f))
    return mean_cd, mean_f


# =================== 3. 训练主流程 ===================

def main():
    # ----------- 基本配置（可以根据需要改） -----------
    # 佛像点云根目录（里面有 full_pc / partial_pc）
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    # CRA-PCN 的 PCN 预训练权重
    pretrain_ckpt = "./pretrain/pcn/ckpt-best.pth"
    # 微调后，保存 val CD 最好的模型
    save_ckpt = "./pretrain/pcn/ckpt-buddha-ft-best.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_epochs = 50
    batch_size = 1            # 为了 Chamfer 不 OOM，保持 1
    lr = 1e-4
    weight_decay = 1e-4

    # 固定随机种子（使训练更稳定，可复现）
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ----------- 数据集 & DataLoader -----------
    # 这里直接用我们带 split 的 BuddhaPairDataset
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

    # ----------- 模型 & 优化器 -----------
    model = CRAPCNWrapper(
        ckpt_path=pretrain_ckpt,
        device=device,
        eval_mode=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
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

            pred = model(partial)          # (B,M,3)
            cd_batch = chamfer_distance(pred, gt)  # (B,)

            loss = cd_batch.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(1, num_batches)

        # ----------- 每个 epoch 做一次 val -----------
        val_cd, val_f = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:03d}] "
              f"train_CD: {train_loss:.6f} | "
              f"val_CD: {val_cd:.6f}, val_F: {val_f:.6f}")

        # 保存当前最好 val CD 的模型
        if val_cd < best_val_cd:
            best_val_cd = val_cd
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.model.state_dict(),  # 只保存底层 CRAPCN
                    "best_val_cd": best_val_cd,
                },
                save_ckpt,
            )
            print(f"  -> 新的最佳模型已保存到 {save_ckpt}")

    print(f"\n训练结束，最佳 epoch = {best_epoch}, 最佳 val CD = {best_val_cd:.6f}")

    # ----------- 使用最佳模型在 test 集上评估 -----------
    if os.path.exists(save_ckpt):
        print(f"\n加载最佳模型 {save_ckpt} 做 test 评估...")
        best_model = CRAPCNWrapper(
            ckpt_path=save_ckpt,
            device=device,
            eval_mode=True,
        )
        test_cd, test_f = evaluate(best_model, test_loader, device)
        print("========================================")
        print("[CRA-PCN Buddha Finetune - Test Set]")
        print(f"样本数: {len(test_loader.dataset)}")
        print(f"平均 Chamfer Distance: {test_cd:.6f}")
        print(f"平均 F-score@0.01:    {test_f:.6f}")
        print("========================================")
    else:
        print("没有找到保存的最佳模型，跳过 test 评估。")


if __name__ == "__main__":
    main()
