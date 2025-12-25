#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage3_gate_multinoise_recall_ema.py

加载：./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth
目标：门控精修 + 多噪声训练 + 召回损失 + EMA，解决 val 好 test 差（小样本过拟合）

假设你已改好模型 TopoCRAPCN_V3(return_all=True) 输出字段至少包含：
- p3_refined, p3_final
并尽量包含：
- delta_fine（用于 L2 约束）
- gate（用于稀疏正则）
"""

import os, sys, math, random, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------- path & import fallback（沿用你 stage2f 的风格） -----------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

ABS_BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
if os.path.isdir(ABS_BUDDHA_ROOT):
    sys.path.insert(0, ABS_BUDDHA_ROOT)

SIBLING_BUDDHA_ROOT = os.path.join(os.path.dirname(CUR_DIR), "buddha_completion_project")
if os.path.isdir(SIBLING_BUDDHA_ROOT):
    sys.path.insert(0, SIBLING_BUDDHA_ROOT)

try:
    from dataset_buddha import BuddhaPairDataset
except ModuleNotFoundError:
    try:
        from datasets.dataset_buddha import BuddhaPairDataset
    except ModuleNotFoundError:
        from buddha_completion_project.dataset_buddha import BuddhaPairDataset

from metrics import chamfer_distance, fscore
from models.crapcn import TopoCRAPCN_V3


# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict): return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict): return ckpt["state_dict"]
    return ckpt

def subsample_xyz(x, n):
    B, N0, _ = x.shape
    if N0 <= n:
        return x.contiguous()
    idx = torch.randperm(N0, device=x.device)[:n]
    return x[:, idx, :].contiguous()

def one_sided_nn_mean(a: torch.Tensor, b: torch.Tensor, chunk=1024):
    """
    mean_i min_j ||a_i - b_j||^2
    a: (B,Na,3), b:(B,Nb,3)
    """
    B, Na, _ = a.shape
    mins = []
    for i0 in range(0, Na, chunk):
        ai = a[:, i0:i0+chunk, :]
        d2 = (ai[:, :, None, :] - b[:, None, :, :]).pow(2).sum(-1)
        mins.append(d2.min(dim=-1).values)
    return torch.cat(mins, dim=1).mean()


# ----------------- augmentation -----------------
def augment_pair(partial, gt,
                 yaw=True, scale=True, jitter=True, dropout=True,
                 scale_range=(0.9,1.1), jitter_sigma=0.003, dropout_range=(0.0,0.25)):
    """
    yaw+scale 同步；jitter/dropout 只作用 partial
    dropout_range 下限=0：覆盖更完整的 partial，贴近 test
    """
    B, Np, _ = partial.shape
    device = partial.device

    if yaw:
        theta = torch.rand(B, device=device) * 2 * math.pi
        c = torch.cos(theta); s = torch.sin(theta)
        R = torch.zeros((B,3,3), device=device, dtype=partial.dtype)
        R[:,0,0] = c;  R[:,0,2] = s
        R[:,1,1] = 1.0
        R[:,2,0] = -s; R[:,2,2] = c
        partial = torch.bmm(partial, R.transpose(1,2))
        gt = torch.bmm(gt, R.transpose(1,2))

    if scale:
        sc = torch.empty(B, 1, 1, device=device).uniform_(scale_range[0], scale_range[1])
        partial = partial * sc
        gt = gt * sc

    if jitter:
        partial = partial + torch.randn_like(partial) * jitter_sigma

    if dropout:
        drop = torch.empty(B, device=device).uniform_(dropout_range[0], dropout_range[1])
        keep_n = (Np * (1.0 - drop)).long().clamp(min=max(64, Np//4), max=Np)
        outs = []
        for b in range(B):
            k = int(keep_n[b].item())
            idx = torch.randperm(Np, device=device)[:k]
            kept = partial[b:b+1, idx, :]
            if k < Np:
                pad_idx = torch.randint(0, k, (Np-k,), device=device)
                kept = torch.cat([kept, kept[:, pad_idx, :]], dim=1)
            outs.append(kept)
        partial = torch.cat(outs, dim=0).contiguous()

    return partial.contiguous(), gt.contiguous()

def _masked_resample(xyz: torch.Tensor, keep_mask: torch.Tensor):
    B, N, _ = xyz.shape
    out = []
    for b in range(B):
        kept = xyz[b][keep_mask[b]]
        if kept.shape[0] < 16:
            kept = xyz[b]
        if kept.shape[0] >= N:
            idx = torch.randperm(kept.shape[0], device=xyz.device)[:N]
            out.append(kept[idx])
        else:
            pad_idx = torch.randint(0, kept.shape[0], (N - kept.shape[0],), device=xyz.device)
            out.append(torch.cat([kept, kept[pad_idx]], dim=0))
    return torch.stack(out, dim=0).contiguous()

def structured_missing(partial, plane_cut_p=0.35, box_hole_p=0.25):
    """
    结构缺损（只作用 partial）：
    - plane cut：模拟断裂/遮挡
    - box hole：模拟局部缺失（手/头冠/底座）
    """
    device = partial.device
    B, N, _ = partial.shape

    if torch.rand(1, device=device).item() < plane_cut_p:
        n = F.normalize(torch.randn(B, 1, 3, device=device), dim=-1)
        proj = (partial * n).sum(-1)
        thr = torch.quantile(proj, q=torch.rand(B, device=device) * 0.25 + 0.50, dim=1, keepdim=True)
        keep = proj <= thr
        partial = _masked_resample(partial, keep)

    if torch.rand(1, device=device).item() < box_hole_p:
        center = partial.mean(dim=1, keepdim=True)
        size = torch.empty(B, 1, 3, device=device).uniform_(0.12, 0.26)
        low = center - size * 0.5
        high = center + size * 0.5
        inside = ((partial >= low) & (partial <= high)).all(dim=-1)
        keep = ~inside
        partial = _masked_resample(partial, keep)

    return partial

def mix_gt_into_partial(partial, gt, mix_ratio_max=0.35, p=0.7):
    """
    Completeness-Mix：从 gt 抽一部分点混入 partial，让训练覆盖更完整的输入分布。
    """
    if torch.rand(1).item() > p:
        return partial
    B, Np, _ = partial.shape
    _, Ng, _ = gt.shape
    device = partial.device

    r = torch.rand(1).item() * mix_ratio_max
    k = max(1, int(Np * r))
    idx = torch.randperm(Ng, device=device)[:k]
    extra = gt[:, idx, :].contiguous()
    union = torch.cat([partial, extra], dim=1)
    sel = torch.randperm(Np + k, device=device)[:Np]
    return union[:, sel, :].contiguous()


# ----------------- partial one-way loss（partial->pred） -----------------
def min_dist2_oneway(src, dst, chunk_n=512, chunk_m=512):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    device = src.device
    big = torch.finfo(src.dtype).max
    out = torch.full((B, N), big, device=device)

    for i in range(0, N, chunk_n):
        i2 = min(i + chunk_n, N)
        s = src[:, i:i2, :]
        local = torch.full((B, i2-i), big, device=device)
        for j in range(0, M, chunk_m):
            j2 = min(j + chunk_m, M)
            d = dst[:, j:j2, :]
            diff = s.unsqueeze(2) - d.unsqueeze(1)
            dist2 = (diff ** 2).sum(-1)
            local = torch.minimum(local, dist2.min(dim=2)[0])
        out[:, i:i2] = local
    return out

def partial_cover_loss_oneway(partial, pred):
    d2 = min_dist2_oneway(partial, pred, chunk_n=512, chunk_m=512)
    return d2.mean()


# ----------------- EMA -----------------
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = None
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            ema_v = self.shadow[k]
            cur_v = v.detach()
            if torch.is_floating_point(ema_v) and torch.is_floating_point(cur_v):
                if ema_v.dtype != cur_v.dtype:
                    cur_v = cur_v.to(dtype=ema_v.dtype)
                ema_v.mul_(self.decay).add_(cur_v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = cur_v.clone()

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None


@torch.no_grad()
def evaluate(model, loader, device, pred_key="p3_final", taus=(0.005,0.01,0.02)):
    model.eval()
    cds = []
    fsum = {t: [] for t in taus}
    for batch in loader:
        partial = batch["partial"].to(device).float()
        gt = batch["gt"].to(device).float()
        out = model(partial, return_all=True)
        pred = out.get(pred_key, out.get("p3_final", out.get("p3_refined")))
        cd = chamfer_distance(pred, gt).mean().item()
        cds.append(cd)
        for t in taus:
            f, _, _ = fscore(pred, gt, threshold=float(t))
            fsum[t].append(float(f.mean().item()))
    return float(np.mean(cds)), {t: float(np.mean(fsum[t])) for t in taus}


def parse_args():
    ap = argparse.ArgumentParser("stage3_gate_multinoise_recall_ema")
    ap.add_argument("--data_root", type=str, default="/root/autodl-tmp/CRA-PCN-main/data/MyDataset")
    ap.add_argument("--load_ckpt", type=str, default="./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth")
    ap.add_argument("--save_ckpt", type=str, default="./pretrain/pcn/ckpt-buddha-topoV3-stage3-gate-multinoise-recall-ema-best.pth")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)

    # phase schedule
    ap.add_argument("--phaseA_epochs", type=int, default=6,  help="只训 gate/time(+delta_head) 的轮数")
    ap.add_argument("--phaseB_epochs", type=int, default=40, help="训 V3 head 的轮数（之后进入 phaseC）")
    ap.add_argument("--lr_new", type=float, default=1e-5)
    ap.add_argument("--lr_v3",  type=float, default=3e-6)
    ap.add_argument("--lr_topo",type=float, default=3e-7)
    ap.add_argument("--weight_decay", type=float, default=2e-4)
    ap.add_argument("--ema_decay", type=float, default=0.9995)

    # augment probs
    ap.add_argument("--plane_cut_p", type=float, default=0.35)
    ap.add_argument("--box_hole_p",  type=float, default=0.25)
    ap.add_argument("--mix_p", type=float, default=0.7)
    ap.add_argument("--mix_ratio_max", type=float, default=0.35)

    # loss weights
    ap.add_argument("--alpha_recall", type=float, default=0.70, help="越大越偏 recall/F")
    ap.add_argument("--w_cd_ref",    type=float, default=0.20)
    ap.add_argument("--w_part_one",  type=float, default=0.20)
    ap.add_argument("--w_delta",     type=float, default=0.08)
    ap.add_argument("--w_stab",      type=float, default=0.02)
    ap.add_argument("--w_gate",      type=float, default=0.01)
    ap.add_argument("--gate_warmup", type=int, default=5)
    ap.add_argument("--beta_score",  type=float, default=0.01, help="选优时 F 的权重")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = args.data_root
    load_ckpt = args.load_ckpt
    save_ckpt = args.save_ckpt
    os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)

    if not os.path.exists(load_ckpt):
        raise FileNotFoundError(f"找不到 stage2 best ckpt: {load_ckpt}")

    # -------- dataset --------
    train_ds = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_ds   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_ds  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")

    print(f"[Dataset] root={data_root} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # -------- model --------
    model = TopoCRAPCN_V3(
        topo_hidden_dim=128, topo_k=16, topo_layers=2,
        delta_scale=0.2, esp_feat_dim=64, use_topo_v2=True
    ).to(device)

    ckpt = torch.load(load_ckpt, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {load_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  missing examples:", missing[:12])
    if len(unexpected) > 0:
        print("  unexpected examples:", unexpected[:12])

    # test before
    cd0, f0 = evaluate(model, test_loader, device, pred_key="p3_final")
    print("\n[TEST before Stage3]")
    print(f"p3_final: CD={cd0:.6f} F@0.01={f0[0.01]:.4f} F@0.02={f0[0.02]:.4f}")

    # -------- freeze all --------
    for p in model.parameters():
        p.requires_grad = False

    # helpers
    def _enable_module(m):
        if m is None: return
        for p in m.parameters():
            p.requires_grad = True

    # collect params
    new_params, v3_params, topo_params = [], [], []

    # phaseA: gate_layer + time_mlp + delta_head
    for name in ["gate_layer", "time_mlp"]:
        if hasattr(model, name):
            _enable_module(getattr(model, name))
            new_params += list(getattr(model, name).parameters())

    if hasattr(model, "delta_head"):
        _enable_module(model.delta_head)
        v3_params += list(model.delta_head.parameters())

    # phaseB modules（先不启用参数）
    if hasattr(model, "feat_mlp"):
        v3_params += list(model.feat_mlp.parameters())
    if hasattr(model, "esp"):
        v3_params += list(model.esp.parameters())
    for n, p in model.named_parameters():
        if ("feat_mlp" in n) or ("esp" in n):
            p.requires_grad = False

    # phaseC: topo_reasoner（先不启用）
    if hasattr(model, "backbone") and hasattr(model.backbone, "topo_reasoner"):
        for p in model.backbone.topo_reasoner.parameters():
            p.requires_grad = False
        topo_params += list(model.backbone.topo_reasoner.parameters())

    def make_optimizer(lr_new, lr_v3, lr_topo):
        groups = []
        if new_params:
            groups.append({"params": [p for p in new_params if p.requires_grad], "lr": lr_new})
        if v3_params:
            groups.append({"params": [p for p in v3_params if p.requires_grad], "lr": lr_v3})
        if topo_params:
            groups.append({"params": [p for p in topo_params if p.requires_grad], "lr": lr_topo})
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    optimizer = make_optimizer(args.lr_new, args.lr_v3, args.lr_topo)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min(args.lr_new, args.lr_v3) * 0.2
    )

    ema = EMA(model, decay=args.ema_decay)

    def score_fn(cd, f01, f02):
        beta = float(args.beta_score)
        return cd - beta * (0.8 * f01 + 0.2 * f02)

    best_score = float("inf")
    best_epoch = -1
    no_improve = 0

    # sanity val EMA
    ema.apply(model)
    vcd, vf = evaluate(model, val_loader, device, pred_key="p3_final")
    ema.restore(model)
    print(f"\n[Sanity val EMA] CD={vcd:.6f} F01={vf[0.01]:.4f} F02={vf[0.02]:.4f} score={score_fn(vcd, vf[0.01], vf[0.02]):.6f}")

    # ================ train ================
    for epoch in range(1, args.epochs + 1):

        # PhaseB 开启 feat_mlp / esp
        if epoch == args.phaseA_epochs + 1:
            if hasattr(model, "feat_mlp"): _enable_module(model.feat_mlp)
            if hasattr(model, "esp"): _enable_module(model.esp)
            optimizer = make_optimizer(args.lr_new, args.lr_v3, args.lr_topo)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(args.epochs - epoch + 1),
                eta_min=min(args.lr_new, args.lr_v3) * 0.2
            )
            print(f"[PhaseB] epoch={epoch}: enable feat_mlp/esp, rebuild optimizer")

        # PhaseC 开启 topo_reasoner（epoch = phaseA + phaseB + 1）
        if epoch == (args.phaseA_epochs + args.phaseB_epochs + 1):
            if hasattr(model, "backbone") and hasattr(model.backbone, "topo_reasoner"):
                _enable_module(model.backbone.topo_reasoner)
            optimizer = make_optimizer(args.lr_new, args.lr_v3, args.lr_topo)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(args.epochs - epoch + 1),
                eta_min=min(args.lr_new, args.lr_v3, args.lr_topo) * 0.2
            )
            print(f"[PhaseC] epoch={epoch}: enable topo_reasoner, rebuild optimizer")

        # 可选：如果你模型里暴露 sigma_max，可在后期降低噪声上限（更利于收敛细节）
        if hasattr(model, "sigma_max") and epoch > int(0.60 * args.epochs):
            ratio = 1.0 - 0.4 * (epoch - int(0.60 * args.epochs)) / max(1, args.epochs - int(0.60 * args.epochs))
            model.sigma_max = float(model.sigma_max) * max(0.6, ratio)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, nb = 0.0, 0
        t0 = time.time()

        for it, batch in enumerate(train_loader, start=1):
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            partial, gt = augment_pair(partial, gt)
            partial = structured_missing(partial, plane_cut_p=args.plane_cut_p, box_hole_p=args.box_hole_p)
            partial = mix_gt_into_partial(partial, gt, mix_ratio_max=args.mix_ratio_max, p=args.mix_p)

            out = model(partial, return_all=True)
            p3_refined = out.get("p3_refined", None)
            p3_final = out.get("p3_final", None)
            if p3_final is None:
                if isinstance(out, (list, tuple)) and len(out) >= 4:
                    p3_final = out[-1]
                else:
                    raise KeyError("找不到 p3_final，请检查 TopoCRAPCN_V3(return_all=True) 输出字段")

            # ----- 召回偏置几何损失（核心） -----
            gt_s = subsample_xyz(gt, 8192)
            pf_s = subsample_xyz(p3_final, 8192)
            loss_p2g = one_sided_nn_mean(pf_s, gt_s, chunk=1024)  # precision
            loss_g2p = one_sided_nn_mean(gt_s, pf_s, chunk=1024)  # recall
            alpha = float(args.alpha_recall)
            loss_cd_bias = (1.0 - alpha) * loss_p2g + alpha * loss_g2p

            # ref 辅助
            loss_cd_ref = torch.zeros([], device=device)
            if p3_refined is not None:
                pr_s = subsample_xyz(p3_refined, 8192)
                loss_cd_ref = chamfer_distance(pr_s, gt_s).mean()

            # partial cover
            part_s = subsample_xyz(partial, 2048)
            pred_s = subsample_xyz(p3_final, 8192)
            loss_part = partial_cover_loss_oneway(part_s, pred_s)

            # delta / gate 正则
            loss_delta = torch.zeros([], device=device)
            if out.get("delta_fine", None) is not None:
                loss_delta = (out["delta_fine"] ** 2).mean()

            loss_gate = torch.zeros([], device=device)
            if epoch > args.gate_warmup and out.get("gate", None) is not None:
                loss_gate = out["gate"].mean()

            # stability
            loss_stab = torch.zeros([], device=device)
            if p3_refined is not None:
                loss_stab = F.smooth_l1_loss(p3_final, p3_refined.detach())

            loss = (
                loss_cd_bias
                + args.w_cd_ref   * loss_cd_ref
                + args.w_part_one * loss_part
                + args.w_delta    * loss_delta
                + args.w_stab     * loss_stab
                + args.w_gate     * loss_gate
            )

            (loss / args.accum_steps).backward()

            if (it % args.accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().cpu())
            nb += 1

            if it % 30 == 0:
                lr0 = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                print(f"[E{epoch:03d}][{it:04d}/{len(train_loader)}] "
                      f"loss={run_loss/max(1,nb):.6f} lr={lr0:.2e} "
                      f"cd_bias={float(loss_cd_bias.item()):.6f} g2p={float(loss_g2p.item()):.6f} p2g={float(loss_p2g.item()):.6f} "
                      f"gate={float(loss_gate.item()):.6f}")

        if (len(train_loader) % args.accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] train_loss={run_loss/max(1,nb):.6f} time={dt:.1f}s")

        # ----- EMA val -----
        ema.apply(model)
        val_cd, val_f = evaluate(model, val_loader, device, pred_key="p3_final")
        ema.restore(model)

        score = score_fn(val_cd, val_f[0.01], val_f[0.02])
        print(f"[VAL-EMA] CD={val_cd:.6f} F01={val_f[0.01]:.4f} F02={val_f[0.02]:.4f} score={score:.6f} "
              f"no_improve={no_improve}/{args.patience}")

        if score < (best_score - 1e-6):
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "score": best_score, "model": ema.shadow}, save_ckpt)
            print(f"  -> saved best EMA: {save_ckpt}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EarlyStop] stop at epoch={epoch}, best_epoch={best_epoch}, best_score={best_score:.6f}")
                break

        # 轻量 test（每 5 epoch）
        if epoch % 5 == 0:
            ema.apply(model)
            test_cd, test_f = evaluate(model, test_loader, device, pred_key="p3_final")
            ema.restore(model)
            print(f"[TEST-EMA@E{epoch:03d}] CD={test_cd:.6f} F01={test_f[0.01]:.4f} F02={test_f[0.02]:.4f}")

    # ----- load best and test -----
    best = torch.load(save_ckpt, map_location="cpu")
    model.load_state_dict(best["model"], strict=False)
    cd1, f1 = evaluate(model, test_loader, device, pred_key="p3_final")
    print("\n[TEST after Stage3 (best EMA)]")
    print(f"p3_final: CD={cd1:.6f} F@0.01={f1[0.01]:.4f} F@0.02={f1[0.02]:.4f}")
    print(f"[Done] best_epoch={best_epoch} best_score={best_score:.6f} ckpt={save_ckpt}")


if __name__ == "__main__":
    main()
