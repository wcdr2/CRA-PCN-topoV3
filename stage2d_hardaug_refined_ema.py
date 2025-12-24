import os, sys, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# allocator 碎片兜底（可有可无）
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ----------------- path & import fallback -----------------
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

# FPS（用 pointnet2_ops，coverage 更稳）
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict): return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict): return ckpt["state_dict"]
    return ckpt

def fps_subsample_xyz(x, n):
    # x: (B,N,3) -> (B,n,3)
    B, N0, _ = x.shape
    if N0 <= n:
        return x.contiguous()
    x_t = x.permute(0, 2, 1).contiguous()  # (B,3,N)
    idx = furthest_point_sample(x_t, n)    # (B,n)
    idx = idx.contiguous()
    out = gather_operation(x_t, idx).permute(0, 2, 1).contiguous()
    return out

def pad_or_resample_to_N(x, N):
    # x: (B,k,3) -> (B,N,3) 允许重复采样
    B, k, _ = x.shape
    if k == N:
        return x.contiguous()
    if k > N:
        idx = torch.randperm(k, device=x.device)[:N]
        return x[:, idx, :].contiguous()
    # k < N
    pad_idx = torch.randint(0, k, (B, N-k), device=x.device)
    pad = torch.gather(x, 1, pad_idx.unsqueeze(-1).expand(-1, -1, 3))
    return torch.cat([x, pad], dim=1).contiguous()

@torch.no_grad()
def evaluate(model, loader, device, pred_key="p3_refined", taus=(0.005,0.01,0.02)):
    model.eval()
    cds = []
    fsum = {t: [] for t in taus}
    for batch in loader:
        partial = batch["partial"].to(device).float()
        gt = batch["gt"].to(device).float()
        out = model(partial, return_all=True)
        pred = out.get(pred_key, out.get("p3_refined", out.get("p3_final")))
        cd = chamfer_distance(pred, gt).mean().item()
        cds.append(cd)
        for t in taus:
            f, _, _ = fscore(pred, gt, threshold=float(t))
            fsum[t].append(float(f.mean().item()))
    return float(np.mean(cds)), {t: float(np.mean(fsum[t])) for t in taus}

# ----------------- structured damage augment -----------------
def random_plane_cut(partial, keep_ratio_range=(0.55, 0.9)):
    # partial: (B,N,3) -> keep points on one side of a random plane
    B, N, _ = partial.shape
    dev = partial.device
    n = torch.randn(B, 3, device=dev)
    n = F.normalize(n, dim=-1)  # (B,3)
    proj = (partial * n[:, None, :]).sum(-1)  # (B,N)

    keep_ratio = torch.empty(B, device=dev).uniform_(keep_ratio_range[0], keep_ratio_range[1])
    # 取分位数阈值：保留较大一侧
    k = (keep_ratio * N).long().clamp(min=max(64, N//4), max=N)
    outs = []
    for b in range(B):
        kb = int(k[b].item())
        thr = torch.kthvalue(proj[b], N - kb + 1).values  # kth smallest -> keep largest kb
        mask = proj[b] >= thr
        kept = partial[b:b+1, mask, :]
        kept = pad_or_resample_to_N(kept, N)
        outs.append(kept)
    return torch.cat(outs, dim=0).contiguous()

def random_box_hole(partial, drop_ratio_range=(0.15, 0.45)):
    # 随机挖掉一个轴对齐盒子内的点（结构化缺损）
    B, N, _ = partial.shape
    dev = partial.device
    mins = partial.amin(dim=1, keepdim=True)
    maxs = partial.amax(dim=1, keepdim=True)
    span = (maxs - mins).clamp(min=1e-6)

    drop_ratio = torch.empty(B, device=dev).uniform_(drop_ratio_range[0], drop_ratio_range[1])
    outs = []
    for b in range(B):
        r = float(drop_ratio[b].item())
        # 盒子边长比例：r 越大，洞越大
        side = span[b] * (0.25 + 0.75 * r)  # (1,3)
        center = mins[b] + torch.rand(1,3, device=dev) * span[b]
        lo = center - side * 0.5
        hi = center + side * 0.5

        p = partial[b]
        inside = ((p >= lo.squeeze(0)) & (p <= hi.squeeze(0))).all(dim=-1)
        kept = p[~inside]
        if kept.shape[0] < max(64, N//4):
            kept = p  # 防止挖太狠
        kept = kept[None, ...]
        kept = pad_or_resample_to_N(kept, N)
        outs.append(kept)
    return torch.cat(outs, dim=0).contiguous()

def augment_pair_hard(partial, gt,
                      yaw=True, scale=True, jitter=True,
                      plane_cut_p=0.55, box_hole_p=0.35,
                      scale_range=(0.9,1.1), jitter_sigma=0.0025):
    """
    对 partial & gt 同步做刚体/尺度（保持对应）。
    对 partial 额外做结构化缺损（plane cut / box hole）。
    """
    B, Np, _ = partial.shape
    dev = partial.device

    if yaw:
        theta = torch.rand(B, device=dev) * 2 * math.pi
        c = torch.cos(theta); s = torch.sin(theta)
        R = torch.zeros((B,3,3), device=dev)
        R[:,0,0] = c;  R[:,0,2] = s
        R[:,1,1] = 1.0
        R[:,2,0] = -s; R[:,2,2] = c
        partial = torch.bmm(partial, R.transpose(1,2))
        gt = torch.bmm(gt, R.transpose(1,2))

    if scale:
        sc = torch.empty(B, 1, 1, device=dev).uniform_(scale_range[0], scale_range[1])
        partial = partial * sc
        gt = gt * sc

    # 结构化缺损：只作用 partial
    if torch.rand(1).item() < plane_cut_p:
        partial = random_plane_cut(partial)
    if torch.rand(1).item() < box_hole_p:
        partial = random_box_hole(partial)

    if jitter:
        partial = partial + torch.randn_like(partial) * jitter_sigma

    return partial.contiguous(), gt.contiguous()

# ----------------- one-sided partial consistency -----------------
def one_sided_nn_mean(a, b, chunk=1024):
    """
    mean_i min_j ||a_i - b_j||^2
    a: (B,Na,3), b:(B,Nb,3)
    """
    B, Na, _ = a.shape
    Nb = b.shape[1]
    mins = []
    for i0 in range(0, Na, chunk):
        ai = a[:, i0:i0+chunk, :]  # (B,ci,3)
        # (B,ci,Nb)
        d2 = (ai[:,:,None,:] - b[:,None,:,:]).pow(2).sum(-1)
        mins.append(d2.min(dim=-1).values)  # (B,ci)
    mind2 = torch.cat(mins, dim=1)
    return mind2.mean()

# ----------------- pointnet2 ops: idx must be contiguous -----------------
def topo_smooth_loss_sub(delta_xyz, xyz, k=16, n_samples=1024):
    from models.utils import query_knn, grouping_operation
    if delta_xyz is None or xyz is None:
        return torch.zeros([], device=xyz.device if xyz is not None else delta_xyz.device)

    B, N, _ = xyz.shape
    if N > n_samples:
        idx = torch.randperm(N, device=xyz.device)[:n_samples]
        xyz_s = xyz[:, idx, :].contiguous()
        delta_s = delta_xyz[:, idx, :].contiguous()
    else:
        xyz_s = xyz.contiguous()
        delta_s = delta_xyz.contiguous()

    idx_knn = query_knn(k, xyz_s, xyz_s).contiguous().int()
    delta_t = delta_s.permute(0,2,1).contiguous()
    neigh = grouping_operation(delta_t, idx_knn).permute(0,2,3,1).contiguous()
    center = delta_s.unsqueeze(2).expand_as(neigh)
    return ((center - neigh) ** 2).sum(dim=-1).mean()

def repulsion_loss_subsample(pred_xyz, k=4, h=0.03, n_samples=2048):
    from models.utils import query_knn, grouping_operation
    B, N, _ = pred_xyz.shape
    m = min(N, n_samples)
    idx = torch.randperm(N, device=pred_xyz.device)[:m]
    xyz = pred_xyz[:, idx, :].contiguous()
    idx_knn = query_knn(k+1, xyz, xyz)[:, :, 1:].contiguous().int()
    xyz_t = xyz.permute(0,2,1).contiguous()
    neigh = grouping_operation(xyz_t, idx_knn).permute(0,2,3,1).contiguous()
    center = xyz.unsqueeze(2).expand_as(neigh)
    dist = torch.norm(center - neigh, dim=-1)
    w = torch.clamp(h - dist, min=0.0)
    return (w ** 2).mean()

def feat_consistency_loss_norm(topo_feat, feat_p2_backbone, n_samples=1024):
    if topo_feat is None or feat_p2_backbone is None:
        dev = topo_feat.device if topo_feat is not None else feat_p2_backbone.device
        return torch.zeros([], device=dev)

    # (B,C,N) -> (B,N,C)
    if topo_feat.dim()==3 and topo_feat.shape[1] < topo_feat.shape[2]:
        topo_feat = topo_feat.permute(0,2,1).contiguous()
    if feat_p2_backbone.dim()==3 and feat_p2_backbone.shape[1] < feat_p2_backbone.shape[2]:
        feat_p2_backbone = feat_p2_backbone.permute(0,2,1).contiguous()

    B, N, C = feat_p2_backbone.shape
    if N > n_samples:
        idx = torch.randperm(N, device=feat_p2_backbone.device)[:n_samples]
        a = topo_feat[:, idx, :]
        b = feat_p2_backbone[:, idx, :].detach()
    else:
        a = topo_feat
        b = feat_p2_backbone.detach()

    d = min(a.shape[-1], b.shape[-1])
    a = F.normalize(a[..., :d], dim=-1)
    b = F.normalize(b[..., :d], dim=-1)
    cos = F.cosine_similarity(a, b, dim=-1)
    return (1.0 - cos).mean()

# ----------------- EMA -----------------
class EMA:
    def __init__(self, model, decay=0.999):
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


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 不改你的数据路径
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    stage2_best = "./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth"
    stage1_best = "./pretrain/pcn/ckpt-buddha-topoV3-ft-best.pth"
    load_ckpt = stage2_best if os.path.exists(stage2_best) else stage1_best
    if not os.path.exists(load_ckpt):
        raise FileNotFoundError(f"找不到 ckpt: {stage2_best} / {stage1_best}")

    save_ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-stage2d-hardaug-ema-best.pth"

    train_ds = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_ds   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_ds  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")
    print(f"[Dataset] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = TopoCRAPCN_V3(
        topo_hidden_dim=128, topo_k=16, topo_layers=2,
        delta_scale=0.2, esp_feat_dim=64, use_topo_v2=True
    ).to(device)

    state = unwrap_state_dict(torch.load(load_ckpt, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {load_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    # -------- test before --------
    cd_f, f_f = evaluate(model, test_loader, device, pred_key="p3_final")
    cd_r, f_r = evaluate(model, test_loader, device, pred_key="p3_refined")
    print("\n[TEST before]")
    print(f"  p3_final  : CD={cd_f:.6f} F@0.01={f_f[0.01]:.4f} F@0.02={f_f[0.02]:.4f}")
    print(f"  p3_refined: CD={cd_r:.6f} F@0.01={f_r[0.01]:.4f} F@0.02={f_r[0.02]:.4f}")

    # -------- Stage2d freeze/unfreeze（延续 Stage2c 的稳训方式）--------
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

    # 分组LR（新模块略大、topo 次之、decoder 最小）
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

    lr_new, lr_topo, lr_dec = 2e-6, 4e-7, 2e-7
    optimizer = torch.optim.AdamW(
        [{"params": new_params, "lr": lr_new},
         {"params": topo_params, "lr": lr_topo},
         {"params": dec_params, "lr": lr_dec}],
        weight_decay=1e-4, betas=(0.9, 0.95)
    )
    max_epoch = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=min(lr_new, lr_topo, lr_dec) * 0.2
    )

    ema = EMA(model, decay=0.999)

    # --- score 更偏向 F01（你的目标 F≈0.40）---
    beta = 0.010
    def score_fn(cd, f01, f02):
        return cd - beta * (0.8 * f01 + 0.2 * f02)

    # training config
    accum_steps = 4
    grad_clip = 1.0
    patience = 12
    min_delta = 1e-6

    # sanity val EMA
    ema.apply(model)
    vcd, vf = evaluate(model, val_loader, device, pred_key="p3_refined")
    ema.restore(model)
    s = score_fn(vcd, vf[0.01], vf[0.02])
    print(f"\n[Sanity val EMA] CD={vcd:.6f} F01={vf[0.01]:.4f} F02={vf[0.02]:.4f} score={s:.6f}")

    best_score = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, max_epoch + 1):
        # ---- loss weight schedule：前稳后准 ----
        if epoch <= 10:
            w_cd_ref = 0.35
            w_part_final = 0.25
            w_part_ref = 0.05
            w_smooth = 0.006
            w_disp = 0.003
            w_rep = 0.010
            w_feat = 0.001
            w_stab = 0.002
        else:
            w_cd_ref = 0.60
            w_part_final = 0.12
            w_part_ref = 0.03
            w_smooth = 0.003
            w_disp = 0.002
            w_rep = 0.006
            w_feat = 0.001
            w_stab = 0.002

        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss = 0.0
        nb = 0

        for it, batch in enumerate(train_loader, start=1):
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            # ✅ 更贴近真实缺损的 hard augmentation
            partial, gt = augment_pair_hard(partial, gt)

            out = model(partial, return_all=True)
            p3_final = out.get("p3_final", out.get("p3_refined"))
            p3_ref  = out.get("p3_refined", None)

            p2_refined = out.get("p2_refined", None)
            delta_p2 = out.get("delta_p2", None)

            topo_feat = out.get("topo_feat", None)
            feat_p2_backbone = out.get("feat_p2_backbone", None)

            # --- FPS 子采样：比随机更稳 ---
            g_cd = fps_subsample_xyz(gt, 8192)
            p_cd = fps_subsample_xyz(p3_final, 8192)
            loss_cd = chamfer_distance(p_cd, g_cd).mean()

            loss_cd_ref = torch.zeros([], device=device)
            loss_stab = torch.zeros([], device=device)
            loss_part_ref = torch.zeros([], device=device)

            if p3_ref is not None:
                pr = fps_subsample_xyz(p3_ref, 8192)
                loss_cd_ref = chamfer_distance(pr, g_cd).mean()
                loss_stab = F.smooth_l1_loss(p3_ref, p3_final.detach())

            # --- 单向 partial 一致性：partial -> pred（更直接推 F@0.01）---
            part_s = fps_subsample_xyz(partial, 1024)
            pred_s = fps_subsample_xyz(p3_final, 4096)
            loss_part = one_sided_nn_mean(part_s, pred_s, chunk=1024)

            if p3_ref is not None:
                pred_rs = fps_subsample_xyz(p3_ref, 4096)
                loss_part_ref = one_sided_nn_mean(part_s, pred_rs, chunk=1024)

            loss_smooth = topo_smooth_loss_sub(delta_p2, p2_refined, k=16, n_samples=1024) if (delta_p2 is not None and p2_refined is not None) else torch.zeros([], device=device)
            loss_disp = (delta_p2 ** 2).mean() if delta_p2 is not None else torch.zeros([], device=device)
            loss_rep = repulsion_loss_subsample(p3_final, k=4, h=0.03, n_samples=2048)
            loss_feat = feat_consistency_loss_norm(topo_feat, feat_p2_backbone, n_samples=1024)

            loss = (
                loss_cd
                + w_cd_ref * loss_cd_ref
                + w_part_final * loss_part
                + w_part_ref * loss_part_ref
                + w_smooth * loss_smooth
                + w_disp * loss_disp
                + w_rep * loss_rep
                + w_feat * loss_feat
                + w_stab * loss_stab
            )

            (loss / accum_steps).backward()

            if (it % accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().cpu())
            nb += 1

        if (len(train_loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # EMA val（用 refined 头选优）
        ema.apply(model)
        val_cd, val_f = evaluate(model, val_loader, device, pred_key="p3_refined")
        ema.restore(model)

        score = score_fn(val_cd, val_f[0.01], val_f[0.02])
        print(f"[Epoch {epoch:03d}/{max_epoch}] train_loss={run_loss/max(1,nb):.6f} | "
              f"val(CD={val_cd:.6f}, F01={val_f[0.01]:.4f}, F02={val_f[0.02]:.4f}) score={score:.6f} "
              f"no_improve={no_improve}/{patience}")

        improved = score < (best_score - min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
            torch.save({"epoch": epoch, "score": best_score, "model": ema.shadow}, save_ckpt)
            print(f"  -> saved best EMA: {save_ckpt}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[EarlyStop] stop at epoch={epoch}, best_epoch={best_epoch}, best_score={best_score:.6f}")
            break

    # -------- load best EMA and test --------
    if os.path.exists(save_ckpt):
        best = torch.load(save_ckpt, map_location="cpu")
        model.load_state_dict(best["model"], strict=False)

    cd_f2, f_f2 = evaluate(model, test_loader, device, pred_key="p3_final")
    cd_r2, f_r2 = evaluate(model, test_loader, device, pred_key="p3_refined")

    print("\n[TEST after Stage2d (best EMA)]")
    print(f"  p3_final  : CD={cd_f2:.6f} F@0.01={f_f2[0.01]:.4f} F@0.02={f_f2[0.02]:.4f}")
    print(f"  p3_refined: CD={cd_r2:.6f} F@0.01={f_r2[0.01]:.4f} F@0.02={f_r2[0.02]:.4f}")
    print(f"\n[Done] best_epoch={best_epoch}, best_score={best_score:.6f}, ckpt={save_ckpt}")

if __name__ == "__main__":
    main()
