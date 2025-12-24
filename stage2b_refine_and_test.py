# stage2b_refine_and_test.py
import os
import sys
import math
import random
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============================================================
# ✅ 关键修复：自动把可能的 dataset_buddha 位置加入 sys.path
#   - 当前目录
#   - /root/autodl-tmp/buddha_completion_project (与你之前日志一致的项目位置)
#   - 当前目录的上一级同级目录下的 buddha_completion_project
#   - 如果存在 datasets/ 子包，也能 import
# ============================================================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

# 你之前报错里出现过这个路径：/root/autodl-tmp/buddha_completion_project/dataset_buddha.py
ABS_BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
if os.path.isdir(ABS_BUDDHA_ROOT):
    sys.path.insert(0, ABS_BUDDHA_ROOT)

# 更通用：假设脚本在 /root/autodl-tmp/CRA-PCN-topoV3 下
# 那 buddha_completion_project 往往在同级目录
SIBLING_BUDDHA_ROOT = os.path.join(os.path.dirname(CUR_DIR), "buddha_completion_project")
if os.path.isdir(SIBLING_BUDDHA_ROOT):
    sys.path.insert(0, SIBLING_BUDDHA_ROOT)

# --- 尝试多种 import 路径（不改你的数据目录） ---
try:
    from dataset_buddha import BuddhaPairDataset
except ModuleNotFoundError:
    try:
        from datasets.dataset_buddha import BuddhaPairDataset
    except ModuleNotFoundError:
        # 如果 buddha_completion_project 是一个包，也可能这样导
        from buddha_completion_project.dataset_buddha import BuddhaPairDataset

from metrics import chamfer_distance, fscore
from models.crapcn import TopoCRAPCN_V3


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(ckpt_obj):
    """兼容 ckpt={'model':...} / {'state_dict':...} / 直接 state_dict"""
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


# ----------------------------
# Low-mem helper losses
# ----------------------------


def topo_smooth_loss_sub(delta_xyz, xyz, k=16, n_samples=1024):
    from models.utils import query_knn, grouping_operation

    B, N, _ = xyz.shape
    if (n_samples is not None) and (N > n_samples):
        idx = torch.randperm(N, device=xyz.device)[:n_samples]
        xyz_s = xyz[:, idx, :].contiguous()
        delta_s = delta_xyz[:, idx, :].contiguous()
    else:
        xyz_s = xyz.contiguous()
        delta_s = delta_xyz.contiguous()

    idx_knn = query_knn(k, xyz_s, xyz_s)              # (B,Ns,k)
    idx_knn = idx_knn.contiguous().int()              # ✅ 关键：contiguous + int

    delta_t = delta_s.permute(0, 2, 1).contiguous()   # (B,3,Ns)
    neigh = grouping_operation(delta_t, idx_knn)      # (B,3,Ns,k)
    neigh = neigh.permute(0, 2, 3, 1).contiguous()    # (B,Ns,k,3)

    center = delta_s.unsqueeze(2).expand_as(neigh)
    diff = center - neigh
    return (diff ** 2).sum(dim=-1).mean()


def disp_l2_loss(delta_xyz):
    return (delta_xyz ** 2).mean()


def repulsion_loss_subsample(pred_xyz, k=4, h=0.03, n_samples=2048):
    from models.utils import query_knn, grouping_operation

    B, N, _ = pred_xyz.shape
    if (n_samples is not None) and (N > n_samples):
        idx = torch.randperm(N, device=pred_xyz.device)[:n_samples]
        xyz = pred_xyz[:, idx, :].contiguous()
    else:
        xyz = pred_xyz.contiguous()

    # query_knn -> (B,Ns,k+1)，切片后容易变非连续，所以必须 contiguous
    idx_knn = query_knn(k + 1, xyz, xyz)[:, :, 1:]    # (B,Ns,k)
    idx_knn = idx_knn.contiguous().int()              # ✅ 关键：contiguous + int

    xyz_t = xyz.permute(0, 2, 1).contiguous()         # (B,3,Ns)
    neigh = grouping_operation(xyz_t, idx_knn)         # (B,3,Ns,k)
    neigh = neigh.permute(0, 2, 3, 1).contiguous()     # (B,Ns,k,3)

    center = xyz.unsqueeze(2).expand_as(neigh)
    dist = torch.norm(center - neigh, dim=-1)          # (B,Ns,k)
    w = torch.clamp(h - dist, min=0.0)
    return (w ** 2).mean()



def feat_consistency_loss_norm(topo_feat, feat_p2_backbone, n_samples=1024):
    """
    FeatLoss：子采样 + L2归一化后再算 (1-cos)，并且 backbone_feat detach
    """
    if topo_feat is None or feat_p2_backbone is None:
        dev = topo_feat.device if topo_feat is not None else feat_p2_backbone.device
        return torch.zeros([], device=dev)

    # 统一成 (B,N,C)
    if topo_feat.dim() == 3 and topo_feat.shape[1] < topo_feat.shape[2]:
        topo_feat = topo_feat.permute(0, 2, 1).contiguous()
    if feat_p2_backbone.dim() == 3 and feat_p2_backbone.shape[1] < feat_p2_backbone.shape[2]:
        feat_p2_backbone = feat_p2_backbone.permute(0, 2, 1).contiguous()

    B, N, C = feat_p2_backbone.shape
    if (n_samples is not None) and (N > n_samples):
        idx = torch.randperm(N, device=feat_p2_backbone.device)[:n_samples]
        a = topo_feat[:, idx, :]
        b = feat_p2_backbone[:, idx, :].detach()
    else:
        a = topo_feat
        b = feat_p2_backbone.detach()

    d = min(a.shape[-1], b.shape[-1])
    a = F.normalize(a[..., :d], dim=-1)
    b = F.normalize(b[..., :d], dim=-1)
    cos = F.cosine_similarity(a, b, dim=-1)  # (B,N)
    return (1.0 - cos).mean()


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, taus=(0.005, 0.01, 0.02), pred_key="p3_final"):
    model.eval()
    cd_sum = 0.0
    f_sum = {t: 0.0 for t in taus}
    n = 0

    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()

        out = model(partial, return_all=True)
        pred = out.get(pred_key, out.get("p3_refined"))

        cd = chamfer_distance(pred, gt).mean().item()
        cd_sum += cd

        for t in taus:
            f, _, _ = fscore(pred, gt, threshold=float(t))  # returns (F,P,R)
            f_sum[t] += float(f.mean().item())

        n += 1

    n = max(1, n)
    return cd_sum / n, {t: f_sum[t] / n for t in taus}


# ----------------------------
# Stage2b refine (low LR + early stop)
# ----------------------------
def stage2b_refine(
    model,
    train_loader,
    val_loader,
    device,
    save_path,
    max_epoch=40,
    accum_steps=4,
    grad_clip=1.0,
    beta_f=0.005,
    pick_tau=0.01,
    patience=8,
    min_delta=1e-6,
    lr_new=2e-6,
    lr_topo=4e-7,
    lr_dec=2e-7,
    weight_decay=1e-4,
):
    # ---- freeze/unfreeze for Stage2 ----
    for p in model.parameters():
        p.requires_grad = True

    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "encoder"):
            for p in model.backbone.encoder.parameters():
                p.requires_grad = False
        if hasattr(model.backbone, "seed_generator"):
            for p in model.backbone.seed_generator.parameters():
                p.requires_grad = False
        if hasattr(model.backbone, "decoder"):
            for p in model.backbone.decoder.parameters():
                p.requires_grad = True
        if hasattr(model.backbone, "topo_reasoner"):
            for p in model.backbone.topo_reasoner.parameters():
                p.requires_grad = True

    new_params, topo_params, dec_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.topo_reasoner" in n:
            topo_params.append(p)
        elif "backbone.decoder" in n:
            dec_params.append(p)
        else:
            new_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": new_params, "lr": lr_new},
            {"params": topo_params, "lr": lr_topo},
            {"params": dec_params, "lr": lr_dec},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=min(lr_dec, lr_topo, lr_new) * 0.2
    )

    # loss weights (与你当前稳定版一致)
    lambda_smooth = 0.005
    lambda_disp   = 0.003
    lambda_rep    = 0.01
    lambda_fine   = 0.003
    lambda_feat   = 0.001
    lambda_stab   = 0.002

    best_score = float("inf")
    best_epoch = -1
    no_improve = 0

    val_cd, val_f = evaluate(model, val_loader, device, taus=(0.005, 0.01, 0.02))
    score = val_cd - beta_f * val_f.get(pick_tau, 0.0)
    print(f"[Stage2b][Sanity] val CD={val_cd:.6f}, F@0.01={val_f.get(0.01,0):.4f}, score={score:.6f}")

    for epoch in range(1, max_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        run_loss = 0.0
        nb = 0

        for it, batch in enumerate(train_loader, start=1):
            partial = batch["partial"].to(device, non_blocking=True).float()
            gt = batch["gt"].to(device, non_blocking=True).float()

            out = model(partial, return_all=True)
            p3_final   = out.get("p3_final", out.get("p3_refined"))
            p3_refined = out.get("p3_refined", None)

            p2_refined = out.get("p2_refined", None)
            delta_p2   = out.get("delta_p2", None)
            delta_fine = out.get("delta_fine", None)

            topo_feat = out.get("topo_feat", None)
            feat_p2_backbone = out.get("feat_p2_backbone", None)

            loss_cd = chamfer_distance(p3_final, gt).mean()

            loss_smooth = topo_smooth_loss_sub(delta_p2, p2_refined, k=16, n_samples=1024) if (delta_p2 is not None and p2_refined is not None) else torch.zeros([], device=device)
            loss_disp = disp_l2_loss(delta_p2) if (delta_p2 is not None) else torch.zeros([], device=device)
            loss_fine = disp_l2_loss(delta_fine) if (delta_fine is not None) else torch.zeros([], device=device)

            loss_rep = repulsion_loss_subsample(p3_final, k=4, h=0.03, n_samples=2048)
            loss_feat = feat_consistency_loss_norm(topo_feat, feat_p2_backbone, n_samples=1024)

            if p3_refined is not None:
                loss_stab = F.smooth_l1_loss(p3_final, p3_refined.detach())
            else:
                loss_stab = torch.zeros([], device=device)

            loss = (
                loss_cd
                + lambda_smooth * loss_smooth
                + lambda_disp   * loss_disp
                + lambda_rep    * loss_rep
                + lambda_fine   * loss_fine
                + lambda_feat   * loss_feat
                + lambda_stab   * loss_stab
            )

            (loss / accum_steps).backward()

            if (it % accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().cpu())
            nb += 1

        if (len(train_loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        val_cd, val_f = evaluate(model, val_loader, device, taus=(0.005, 0.01, 0.02))
        score = val_cd - beta_f * val_f.get(pick_tau, 0.0)

        lr_str = ",".join([f"{g['lr']:.1e}" for g in optimizer.param_groups])
        print(
            f"[Stage2b][Epoch {epoch:03d}/{max_epoch}] lr={lr_str} "
            f"train(loss={run_loss/max(1,nb):.6f}) | "
            f"val(CD={val_cd:.6f}, F@0.01={val_f.get(0.01,0):.4f}, F@0.02={val_f.get(0.02,0):.4f}) "
            f"score={score:.6f} no_improve={no_improve}/{patience}"
        )

        improved = score < (best_score - min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"epoch": epoch, "best_score": best_score, "model": model.state_dict()}, save_path)
            print(f"  -> saved best: {save_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[Stage2b][EarlyStop] stop at epoch={epoch}, best_epoch={best_epoch}, best_score={best_score:.6f}")
            break

    return best_epoch, best_score


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 不改你的数据位置
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    stage2_best = "./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth"
    stage1_best = "./pretrain/pcn/ckpt-buddha-topoV3-ft-best.pth"
    stage2b_best = "./pretrain/pcn/ckpt-buddha-topoV3-stage2b-ft-best.pth"

    train_dataset = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_dataset   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_dataset  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")

    print(f"[Dataset] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = TopoCRAPCN_V3(
        topo_hidden_dim=128,
        topo_k=16,
        topo_layers=2,
        delta_scale=0.2,
        esp_feat_dim=64,
        use_topo_v2=True,
    ).to(device)

    ckpt_path = stage2_best if os.path.exists(stage2_best) else stage1_best
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 ckpt：{stage2_best} 也找不到 {stage1_best}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] ckpt={ckpt_path}")
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")

    # Test BEFORE refine
    test_cd, test_f = evaluate(model, test_loader, device, taus=(0.005, 0.01, 0.02))
    print("========================================")
    print("[TEST - before Stage2b refine]")
    print(f"CD={test_cd:.6f} | F@0.005={test_f[0.005]:.4f} F@0.01={test_f[0.01]:.4f} F@0.02={test_f[0.02]:.4f}")
    print("========================================")

    # Stage2b refine
    best_epoch, best_score = stage2b_refine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=stage2b_best,
        max_epoch=40,
        accum_steps=4,
        grad_clip=1.0,
        beta_f=0.005,
        pick_tau=0.01,
        patience=8,
        min_delta=1e-6,
        lr_new=2e-6,
        lr_topo=4e-7,
        lr_dec=2e-7,
        weight_decay=1e-4,
    )
    print(f"[Stage2b] done. best_epoch={best_epoch}, best_score={best_score:.6f}, ckpt={stage2b_best}")

    # Test AFTER refine
    if os.path.exists(stage2b_best):
        ckpt2 = torch.load(stage2b_best, map_location="cpu")
        state2 = unwrap_state_dict(ckpt2)
        model.load_state_dict(state2, strict=False)

    test_cd2, test_f2 = evaluate(model, test_loader, device, taus=(0.005, 0.01, 0.02))
    print("========================================")
    print("[TEST - after Stage2b refine]")
    print(f"CD={test_cd2:.6f} | F@0.005={test_f2[0.005]:.4f} F@0.01={test_f2[0.01]:.4f} F@0.02={test_f2[0.02]:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()
