import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append("/root/autodl-tmp/buddha_completion_project")
from dataset_buddha import BuddhaPairDataset
from metrics import chamfer_distance, fscore


# 让 Python 找到 CRA-PCN 的 models 包
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from models.crapcn import CRAPCN   # 官方 infer.py 里就是这么 import 的


# =========================================================
# 1. 佛像数据集：full_pc + partial_pc 成对读取 + 采样 + 归一化
# =========================================================


# =========================================================
# 2. 简单的 CD & F-score 指标
#    （朴素版本，直接构造 (B,N,M) 距离矩阵）
# =========================================================




# =========================================================
# 3. CRA-PCN 模型封装
# =========================================================

class CRAPCNWrapper(nn.Module):
    def __init__(self,
                 ckpt_path="./pretrain/pcn/ckpt-best.pth",
                 device="cuda:0",
                 eval_mode=True):
        super().__init__()

        self.device = torch.device(device)
        self.model = CRAPCN().to(self.device)

        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)

        if eval_mode:
            self.model.eval()

    def _load_ckpt(self, ckpt_path):
        print(f"[CRAPCNWrapper] 加载权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 官方训练脚本里是 checkpoint['model']
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # 把 DataParallel 的 'module.' 前缀去掉
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

        # CRAPCN 在 infer.py 里就是直接吃 (B,N,3)
        re = self.model(x)     # re 是多级输出的 list / tuple
        pred = re[-1]          # 取最稠密那一级 (B,M,3) 或 (B,3,M)

        # 如果输出是 (B,3,M)，转成 (B,M,3)
        if pred.dim() == 3 and pred.size(1) == 3 and pred.size(2) != 3:
            pred = pred.transpose(1, 2).contiguous()

        return pred


# =========================================================
# 4. 主函数：遍历佛像数据，计算平均指标
# =========================================================

def main():
    # ★★ 这里改成你佛像数据所在路径 ★★
    # 比如你之前在 PoinTr 里是 data/MyDataset，就写成绝对路径：
    # data_root = "/root/autodl-tmp/PoinTr/data/MyDataset"
    data_root = "./data/MyDataset"

    ckpt_path = "./pretrain/pcn/ckpt-best.pth"   # 官方提供的 PCN 预训练权重
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = BuddhaPairDataset(root_dir=data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=1,   # 为了 CD 不 OOM，保持 1
        shuffle=False,
        num_workers=0,
    )

    model = CRAPCNWrapper(
        ckpt_path=ckpt_path,
        device=device,
        eval_mode=True,
    )

    all_cd = []
    all_f = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            partial = batch["partial"].to(device).float()  # (B,N,3)
            gt = batch["gt"].to(device).float()            # (B,M,3)

            if partial.dim() == 2:   # 单样本时补一个 batch 维
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

            pred = model(partial)          # (B,M,3)

            cd_batch = chamfer_distance(pred, gt)      # (B,)
            f_batch = fscore(pred, gt, tau=0.01)       # (B,)

            all_cd += cd_batch.cpu().tolist()
            all_f += f_batch.cpu().tolist()

    mean_cd = sum(all_cd) / len(all_cd)
    mean_f = sum(all_f) / len(all_f)

    print("========================================")
    print("[CRA-PCN Zero-shot on Buddha Dataset]")
    print(f"样本数: {len(dataset)}")
    print(f"平均 Chamfer Distance: {mean_cd:.6f}")
    print(f"平均 F-score@0.01:    {mean_f:.6f}")
    print("========================================")


if __name__ == "__main__":
    main()
