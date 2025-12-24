import os
import sys
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================================================
# Stage2-only 训练脚本（从 Stage1 best ckpt 直接开始）
# ✅ 严格保持你的原始数据路径：/root/autodl-tmp/CRA-PCN-main/data/MyDataset
# ✅ 解决 Stage2 OOM：不做 chamfer(p3_final, p3_refined) 一致性
# ✅ 兼容 metrics.py：fscore(threshold=...) 返回 (F,P,R)
# ✅ 不用 AMP（pointnet2_ops 对 fp16 不稳定）
# =========================================================

# allocator 碎片兜底（可有可无，不影响正确性）
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

# 如果你确实需要从 buddha_completion_project 引用模块，可以保留；但不再把数据根指到那里
BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
if os.path.isdir(BUDDHA_ROOT):
    sys.path.append(BUDDHA_ROOT)

from dataset_buddha import BuddhaPairDataset
from metrics import chamfer_distance, fscore
from models.crapcn import TopoCRAPCN_V3


# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def gather_random(x: torch.Tensor, m: int) -> torch.Tensor:
    """x: (B,N,3) -> (B,m,3) 随机子采样（仅用于 loss 省显存；m=0 不采样）"""
    if m <= 0:
        return x
    B, N, C = x.shape
    if N <= m:
        return x
    idx = torch.randint(0, N, (B, m), device=x.device)
    idx = idx.unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(x, 1, idx)


# ---------------- low-mem losses ----------------
def topo_smooth_loss_sub(delta_xyz, xyz, k=16, n_samples=1024):
    """邻域位移平滑：仅在子采样点上算，降低 Stage2 显存/耗时"""
    from models.utils import query_knn, grouping_operation

    B, N, _ = xyz.shape
    if (n_samples > 0) and (N > n_samples):
        idx = torch.randperm(N, device=xyz.device)[:n_samples]
        xyz_s = xyz[:, idx, :]
        delta_s = delta_xyz[:, idx, :]
    else:
        xyz_s = xyz
        delta_s = delta_xyz

    idx_knn = query_knn(k, xyz_s, xyz_s)  # (B,Ns,k)

    delta_t = delta_s.permute(0, 2, 1).contiguous()        # (B,3,Ns)
    neigh = grouping_operation(delta_t, idx_knn)           # (B,3,Ns,k)
    neigh = neigh.permute(0, 2, 3, 1).contiguous()         # (B,Ns,k,3)
    center = delta_s.unsqueeze(2).expand_as(neigh)         # (B,Ns,k,3)
    return (center - neigh).pow(2).sum(dim=-1).mean()


def disp_l2_loss(delta_xyz):
    return (delta_xyz ** 2).mean()


def repulsion_loss_sub(pred_xyz, k=4, h=0.03, n_samples=2048):
    """Repulsion：只在子采样点上算"""
    from models.utils import query_knn, grouping_operation

    B, N, _ = pred_xyz.shape
    if (n_samples > 0) and (N > n_samples):
        idx = torch.randperm(N, device=pred_xyz.device)[:n_samples]
        xyz = pred_xyz[:, idx, :]
    else:
        xyz = pred_xyz

    idx_knn = query_knn(k + 1, xyz, xyz)[:, :, 1:].contiguous().int()  # (B,Ns,k)

    xyz_t = xyz.permute(0, 2, 1).contiguous()  # (B,3,Ns)
    neigh = grouping_operation(xyz_t, idx_knn).permute(0, 2, 3, 1).contiguous()  # (B,Ns,k,3)
    center = xyz.unsqueeze(2).expand_as(neigh)

    dist = torch.norm(center - neigh, dim=-1)  # (B,Ns,k)
    return F.relu(h - dist).pow(2).mean()


def feat_loss_sub(topo_feat, feat_p2_backbone, n_samples=1024):
    """特征一致性：子采样 + L2 归一化，避免特征损失主导训练"""
    if (topo_feat is None) or (feat_p2_backbone is None):
        dev = topo_feat.device if topo_feat is not None else feat_p2_backbone.device
        return torch.zeros([], device=dev)

    B, N, _ = feat_p2_backbone.shape
    if (n_samples > 0) and (N > n_samples):
        idx = torch.randperm(N, device=feat_p2_backbone.device)[:n_samples]
        t = topo_feat[:, idx, :]
        b = feat_p2_backbone[:, idx, :]
    else:
        t = topo_feat
        b = feat_p2_backbone

    d = min(t.shape[-1], b.shape[-1])
    t = F.normalize(t[..., :d], dim=-1)
    b = F.normalize(b[..., :d], dim=-1)
    return (t - b).pow(2).mean()


# ---------------- eval (compatible with metrics.py) ----------------
@torch.no_grad()
def evaluate(model, dataloader, device, taus=(0.005, 0.01, 0.02)):
    model.eval()
    cds = []
    f_dict = {t: [] for t in taus}

    for batch in dataloader:
        partial = batch["partial"].to(device).float()
        gt = batch["gt"].to(device).float()
        if partial.dim() == 2:
            partial = partial.unsqueeze(0)
            gt = gt.unsqueeze(0)

        out = model(partial, return_all=True)
        pred = out.get("p3_final", out.get("p3_refined"))

        cd = unwrap_first(chamfer_distance(pred, gt)).mean()
        cds.append(float(cd.detach().cpu()))

        for t in taus:
            # metrics.py：fscore(pred, gt, threshold=...) -> (F,P,R)
            f, _, _ = fscore(pred, gt, threshold=float(t))
            f_dict[t].append(float(f.mean().detach().cpu()))

    mean_cd = float(np.mean(cds)) if len(cds) else 0.0
    mean_f = {t: float(np.mean(v)) if len(v) else 0.0 for t, v in f_dict.items()}
    return mean_cd, mean_f


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device = {device}")

    # ✅ 保持你的原始数据路径（不要改到 buddha_completion_project）
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"
    if not os.path.isdir(data_root):
        raise AssertionError(
            f"数据根目录不存在：{data_root}\n"
            f"（你原来 Stage1 跑通就是用这个路径；请确认该目录仍然存在。）"
        )

    # 额外检查：如果你的 dataset_buddha.py 期望 data_buddha_pairs 结构，这里提前给出更明确的错误
    full_dir = os.path.join(data_root, "data_buddha_pairs", "full_pc")
    part_dir = os.path.join(data_root, "data_buddha_pairs", "partial_pc")
    if (not os.path.isdir(full_dir)) or (not os.path.isdir(part_dir)):
        print("[Warn] 你的数据集类可能要求以下目录存在：")
        print("       ", full_dir)
        print("       ", part_dir)
        print("       若这里不存在，但你 Stage1 之前能跑通，说明你当前运行的 dataset_buddha.py 可能不是当时那个版本。")
        # 这里不擅自改路径，只给提示
        # 不 raise：让 BuddhaPairDataset 自己按照它的规则去读（如果能读就继续）

    stage1_best_ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-ft-best.pth"
    if not os.path.exists(stage1_best_ckpt):
        raise FileNotFoundError(f"找不到 Stage1 best ckpt：{stage1_best_ckpt}")

    save_ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth"
    os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)

    # -------- dataset --------
    train_dataset = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_dataset   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_dataset  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")
    print(f"[Dataset] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Stage2 更吃显存：batch=1 + 累积
    batch_size = 1
    accum_steps = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # -------- model --------
    model = TopoCRAPCN_V3(
        topo_hidden_dim=128,
        topo_k=16,
        topo_layers=2,
        delta_scale=0.2,
        esp_feat_dim=64,
        use_topo_v2=True,
    ).to(device)

    # load stage1 best (包含 ESP 权重)
    ckpt = torch.load(stage1_best_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] Stage1 best: {stage1_best_ckpt}")
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")

    # -------- Stage2：冻结 encoder & seed_generator；解冻 topo_reasoner + decoder + V3分支 --------
    # 小数据集更稳：不让 encoder/seed_generator 漂移
    for p in model.parameters():
        p.requires_grad = True

    for p in model.backbone.encoder.parameters():
        p.requires_grad = False
    for p in model.backbone.seed_generator.parameters():
        p.requires_grad = False

    for p in model.backbone.topo_reasoner.parameters():
        p.requires_grad = True
    for p in model.backbone.decoder.parameters():
        p.requires_grad = True

    for p in model.feat_mlp.parameters():
        p.requires_grad = True
    for p in model.esp.parameters():
        p.requires_grad = True
    for p in model.delta_head.parameters():
        p.requires_grad = True

    # -------- optimizer：分组 lr（温和）--------
    # 经验：decoder 最小，topo_reasoner 次之，V3新分支略大但仍小
    lr_new = 1e-5
    lr_topo = 2e-6
    lr_dec = 1e-6
    weight_decay = 1e-4

    params_new = list(model.feat_mlp.parameters()) + list(model.esp.parameters()) + list(model.delta_head.parameters())
    params_topo = list(model.backbone.topo_reasoner.parameters())
    params_dec = list(model.backbone.decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": params_new, "lr": lr_new},
            {"params": params_topo, "lr": lr_topo},
            {"params": params_dec, "lr": lr_dec},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    max_epoch = 120
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=lr_dec * 0.2)

    # -------- loss weights（Stage2 稳定版）--------
    lambda_smooth = 0.005
    lambda_disp   = 0.003
    lambda_rep    = 0.01
    lambda_fine   = 0.003
    lambda_feat   = 0.001

    # ✅ 关键修复：不用 chamfer(p3_final, p3_refined)（会炸显存/慢）
    # 改成点对点稳定项 SmoothL1：O(N) 显存
    lambda_stab = 0.002

    # best 选择：沿用你日志里的 score = CD - 0.005 * F@0.01
    beta_f = 0.005
    taus_eval = (0.005, 0.01, 0.02)

    best_score = float("inf")
    best_epoch = -1

    val_cd0, val_f0 = evaluate(model, val_loader, device, taus=taus_eval)
    print(f"[Sanity] val CD={val_cd0:.6f}, F@0.01={val_f0.get(0.01,0.0):.4f}")

    # -------- train loop --------
    for epoch in range(1, max_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        run_loss = 0.0
        run_cd = 0.0
        nb = 0

        for it, batch in enumerate(train_loader, start=1):
            partial = batch["partial"].to(device, non_blocking=True).float()
            gt = batch["gt"].to(device, non_blocking=True).float()
            if partial.dim() == 2:
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

            out = model(partial, return_all=True)  # fp32
            p3_final = out["p3_final"].float()
            p3_refined = out.get("p3_refined", None)

            p2_refined = out.get("p2_refined", None)
            delta_p2   = out.get("delta_p2", None)
            delta_fine = out.get("delta_fine", None)

            # 主损失（可选：loss 下采样进一步省显存；默认 m=0 不采样）
            p3_loss = gather_random(p3_final, m=0)
            gt_loss = gather_random(gt, m=0)
            loss_cd = unwrap_first(chamfer_distance(p3_loss, gt_loss)).mean()

            if (delta_p2 is not None) and (p2_refined is not None):
                loss_smooth = topo_smooth_loss_sub(delta_p2.float(), p2_refined.float(), k=16, n_samples=1024)
                loss_disp = disp_l2_loss(delta_p2.float())
            else:
                loss_smooth = torch.zeros([], device=device)
                loss_disp = torch.zeros([], device=device)

            loss_rep = repulsion_loss_sub(p3_final, k=4, h=0.03, n_samples=2048)
            loss_fine = disp_l2_loss(delta_fine.float()) if delta_fine is not None else torch.zeros([], device=device)
            loss_feat = feat_loss_sub(out.get("topo_feat", None), out.get("feat_p2_backbone", None), n_samples=1024)

            # ✅ 点对点稳定项（替代 chamfer-consistency）
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().cpu())
            run_cd += float(loss_cd.detach().cpu())
            nb += 1

        # 尾部不足 accum_steps 的梯度
        if (len(train_loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        tr_loss = run_loss / max(1, nb)
        tr_cd = run_cd / max(1, nb)

        val_cd, val_f = evaluate(model, val_loader, device, taus=taus_eval)
        f01 = val_f.get(0.01, 0.0)
        score = val_cd - beta_f * f01

        lr_str = ",".join([f"{g['lr']:.1e}" for g in optimizer.param_groups])
        print(
            f"[Epoch {epoch:03d}/{max_epoch}] stage2 lr={lr_str} "
            f"train(loss={tr_loss:.6f}, cd={tr_cd:.6f}) | "
            f"val(CD={val_cd:.6f}, F@0.005={val_f.get(0.005,0.0):.4f}, F@0.01={f01:.4f}, F@0.02={val_f.get(0.02,0.0):.4f}) "
            f"score={score:.6f}"
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_cd": val_cd,
                    "val_f": val_f,
                    "score": score,
                },
                save_ckpt,
            )
            print(f"  -> saved best to {save_ckpt}")

    print(f"\n训练结束：best_epoch={best_epoch}, best_score={best_score:.6f}")

    # test best
    if os.path.exists(save_ckpt):
        ck = torch.load(save_ckpt, map_location="cpu")
        model.load_state_dict(ck.get("model", ck), strict=False)

    test_cd, test_f = evaluate(model, test_loader, device, taus=taus_eval)
    print("========================================")
    print("[Topo-CRA-PCN V3 Stage2 - Test Set]")
    print(f"CD: {test_cd:.6f}")
    print(f"F@0.005: {test_f.get(0.005,0.0):.4f}")
    print(f"F@0.01 : {test_f.get(0.01,0.0):.4f}")
    print(f"F@0.02 : {test_f.get(0.02,0.0):.4f}")
    print("========================================")


if __name__ == "__main__":
    main()
