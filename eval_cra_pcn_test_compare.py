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

# 统一佛像工程的路径 ★★ 如果你的路径不一样，这里要改 ★★
BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
sys.path.append(BUDDHA_ROOT)

from dataset_buddha import BuddhaPairDataset
from metrics import chamfer_distance, fscore

from models.crapcn import CRAPCN


# =================== 1. 模型封装 ===================

class CRAPCNWrapper(nn.Module):
    def __init__(self,
                 ckpt_path,
                 device="cuda:0",
                 eval_mode=True):
        super().__init__()
        self.device = torch.device(device)
        self.model = CRAPCN().to(self.device)

        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_ckpt(ckpt_path)
        else:
            raise FileNotFoundError(f"[CRAPCNWrapper] 找不到权重文件: {ckpt_path}")

        if eval_mode:
            self.model.eval()

    def _load_ckpt(self, ckpt_path):
        print(f"[CRAPCNWrapper] 加载权重: {ckpt_path}")
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


# =================== 2. 评估函数 ===================

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


# =================== 3. 主函数：同一 test 集上比较两种权重 ===================

def build_test_loader(data_root, seed=42):
    """
    为了保证两次评估的采样完全一致，这里每次构建 DataLoader 前都重置随机种子。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = BuddhaPairDataset(
        root_dir=data_root,
        n_partial=2048,
        n_full=16384,
        split="test",     # ★★ 只用 test 集那 25 对样本 ★★
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    return loader


def main():
    # 佛像数据根目录
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    # 两个要对比的权重
    pretrain_ckpt = "./pretrain/pcn/ckpt-best.pth"
    finetune_ckpt = "./pretrain/pcn/ckpt-buddha-ft-best.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) zero-shot：PCN 预训练权重，在 test split 上评估
    print("============== Zero-shot (PCN 预训练) on TEST ==============")
    test_loader = build_test_loader(data_root, seed=42)
    model_pre = CRAPCNWrapper(ckpt_path=pretrain_ckpt, device=device, eval_mode=True)
    cd_pre, f_pre = evaluate(model_pre, test_loader, device)
    print(f"样本数: {len(test_loader.dataset)}")
    print(f"平均 Chamfer Distance: {cd_pre:.6f}")
    print(f"平均 F-score@0.01:    {f_pre:.6f}")
    print("============================================================\n")

    # 2) finetune：你刚刚训练好的佛像微调权重，在同一 test split 上评估
    print("============== Finetune (Buddha 微调) on TEST ==============")
    test_loader = build_test_loader(data_root, seed=42)   # 再次重置种子，保证采样一致
    model_ft = CRAPCNWrapper(ckpt_path=finetune_ckpt, device=device, eval_mode=True)
    cd_ft, f_ft = evaluate(model_ft, test_loader, device)
    print(f"样本数: {len(test_loader.dataset)}")
    print(f"平均 Chamfer Distance: {cd_ft:.6f}")
    print(f"平均 F-score@0.01:    {f_ft:.6f}")
    print("============================================================\n")

    # 3) 打个小总结
    print("============== 对比总结（同一 TEST 集） ==============")
    print(f"Zero-shot : CD={cd_pre:.6f}, F@0.01={f_pre:.6f}")
    print(f"Finetuned : CD={cd_ft:.6f}, F@0.01={f_ft:.6f}")
    print("  提升比（CD） : {:.2f}x 更小".format(cd_pre / cd_ft if cd_ft > 0 else float('inf')))
    print("  提升比（F-score） : {:.2f}x 更高".format(f_ft / f_pre if f_pre > 0 else float('inf')))
    print("===================================================")


if __name__ == "__main__":
    main()
