import os, sys, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------- path & import fallback（照抄你稳定脚本的写法） -----------------
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
    # x: (B,N,3)
    B, N0, _ = x.shape
    if N0 <= n:
        return x.contiguous()
    idx = torch.randperm(N0, device=x.device)[:n]
    return x[:, idx, :].contiguous()

def merge_partial(pred, partial):
    """
    关键：把输出的前 Np 个点强制替换为 partial（观测点不允许被 refine 挪坏）
    pred: (B,N,3), partial: (B,Np,3)
    """
    B, Np, _ = partial.shape
    if pred.shape[1] < Np:
        raise RuntimeError(f"pred N={pred.shape[1]} < partial Np={Np}")
    return torch.cat([partial, pred[:, Np:, :]], dim=1).contiguous()

def augment_light(partial, gt,
                  yaw=True, jitter=True, dropout=True,
                  jitter_sigma=0.002, dropout_keep_min=0.88):
    """
    轻增强：只做 yaw + 小 jitter + 小 dropout（避免 Stage2d 那种 hardaug 把任务训偏）
    """
    B, _, _ = partial.shape
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

    if jitter:
        partial = partial + torch.randn_like(partial) * jitter_sigma

    if dropout:
        # 随机丢点再补回（保持点数不变）
        B, N, _ = partial.shape
        keep_ratio = torch.empty(B, device=device).uniform_(dropout_keep_min, 1.0)
        out = []
        for b in range(B):
            k = max(64, int(N * float(keep_ratio[b].item())))
            idx = torch.randperm(N, device=device)[:k]
            kept = partial[b:b+1, idx, :]
            rep = torch.randint(0, k, (1, N), device=device)
            out.append(torch.gather(kept, 1, rep.unsqueeze(-1).expand(-1,-1,3)))
        partial = torch.cat(out, dim=0).contiguous()

    return partial.contiguous(), gt.contiguous()


class EMA:
    """
    只跟踪 float 权重，避免 BN 的 Long buffer 报错（你之前遇到过）
    """
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = None
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v):
                    self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in msd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model):
        self.backup = {}
        msd = model.state_dict()
        for k in self.shadow.keys():
            self.backup[k] = msd[k].detach().clone()
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=False)
        self.backup = None


@torch.no_grad()
def evaluate_merge(model, loader, device, pred_key="p3_final", taus=(0.005,0.01,0.02)):
    model.eval()
    cds = []
    fsum = {t: [] for t in taus}
    for batch in loader:
        partial = batch["partial"].to(device).float()
        gt = batch["gt"].to(device).float()

        out = model(partial, return_all=True)
        pred = out.get(pred_key, out.get("p3_final", out.get("p3_refined")))
        pred = merge_partial(pred, partial)

        # 省显存：统一下采样到 8192
        pred_s = subsample_xyz(pred, 8192)
        gt_s   = subsample_xyz(gt,   8192)

        cd = chamfer_distance(pred_s, gt_s).mean().item()
        cds.append(cd)
        for t in taus:
            f, _, _ = fscore(pred_s, gt_s, threshold=float(t))
            fsum[t].append(float(f.mean().item()))
    return float(np.mean(cds)), {t: float(np.mean(fsum[t])) for t in taus}


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 保持你的原始数据路径不变
    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"

    # ckpt：优先 stage2-ft-best，否则 stage1 best
    ckpt_stage2 = "./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth"
    ckpt_stage1 = "./pretrain/pcn/ckpt-buddha-topoV3-ft-best.pth"
    load_ckpt = ckpt_stage2 if os.path.exists(ckpt_stage2) else ckpt_stage1
    if not os.path.exists(load_ckpt):
        raise FileNotFoundError(f"找不到 ckpt: {ckpt_stage2} / {ckpt_stage1}")

    save_ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-stage2e-generalize-ema-best.pth"
    os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)

    # Dataset：对齐你原始脚本参数
    train_ds = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_ds   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_ds  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")
    print(f"[Dataset] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Model：对齐你 V3 构造参数
    model = TopoCRAPCN_V3(
        topo_hidden_dim=128, topo_k=16, topo_layers=2,
        delta_scale=0.2, esp_feat_dim=64, use_topo_v2=True
    ).to(device)

    state = unwrap_state_dict(torch.load(load_ckpt, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {load_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    # =========================
    # 关键：冻结 backbone，只训练 V3 新增模块（最抗 val 过拟合）
    # =========================
    for p in model.parameters():
        p.requires_grad = False
    # V3 新增
    for p in model.feat_mlp.parameters():
        p.requires_grad = True
    for p in model.esp.parameters():
        p.requires_grad = True
    for p in model.delta_head.parameters():
        p.requires_grad = True

    # optimizer（只更新新模块）
    lr = 1e-5
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )
    max_epoch = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=2e-6)

    # EMA + early stop
    ema = EMA(model, decay=0.999)
    patience = 10
    best_score = float("inf")
    best_epoch = -1
    no_improve = 0

    # score：更贴近你目标（提升 F01）
    beta_f = 0.02  # 比你之前 0.005 更重视 F@0.01
    def score_fn(cd, f01):
        return cd - beta_f * f01

    # Test before
    cd0, f0 = evaluate_merge(model, test_loader, device, pred_key="p3_final")
    print("\n[TEST before Stage2e]")
    print(f"p3_final(merge): CD={cd0:.6f} F@0.01={f0[0.01]:.4f} F@0.02={f0[0.02]:.4f}")

    # Train
    for epoch in range(1, max_epoch + 1):
        model.train()
        run_loss = 0.0
        nb = 0

        for batch in train_loader:
            partial = batch["partial"].to(device).float()
            gt = batch["gt"].to(device).float()

            partial, gt = augment_light(partial, gt)

            out = model(partial, return_all=True)
            p3_refined = out["p3_refined"]
            p3_final   = out["p3_final"]
            delta_fine = out["delta_fine"]

            # merge（训练/验证/测试一致）
            p3_final_m   = merge_partial(p3_final, partial)
            p3_refined_m = merge_partial(p3_refined, partial)

            # 下采样算 CD（防显存）
            gt_s = subsample_xyz(gt, 8192)
            pf_s = subsample_xyz(p3_final_m, 8192)
            pr_s = subsample_xyz(p3_refined_m, 8192)

            cd_final = chamfer_distance(pf_s, gt_s).mean()
            cd_ref   = chamfer_distance(pr_s, gt_s).mean()

            # 关键：限制 V3 精修位移幅度（避免“补出伪细节”）
            loss_delta = (delta_fine ** 2).mean()

            # 稳定：final 不要偏离 refined 太多（只约束新增层）
            loss_stab = F.smooth_l1_loss(p3_final, p3_refined.detach())

            # 权重：非常保守（目的：不让 test 继续变差）
            loss = 1.0 * cd_final + 0.3 * cd_ref + 0.02 * loss_delta + 0.01 * loss_stab

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            ema.update(model)

            run_loss += float(loss.detach().cpu())
            nb += 1

        scheduler.step()

        # EMA val
        ema.apply(model)
        val_cd, val_f = evaluate_merge(model, val_loader, device, pred_key="p3_final")
        ema.restore(model)

        score = score_fn(val_cd, val_f[0.01])
        print(f"[Epoch {epoch:03d}/{max_epoch}] loss={run_loss/max(1,nb):.6f} | "
              f"val(CD={val_cd:.6f}, F01={val_f[0.01]:.4f}, F02={val_f[0.02]:.4f}) score={score:.6f} "
              f"no_improve={no_improve}/{patience}")

        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "score": best_score, "model": ema.shadow}, save_ckpt)
            print(f"  -> saved best EMA: {save_ckpt}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] stop at epoch={epoch}, best_epoch={best_epoch}, best_score={best_score:.6f}")
                break

    # Load best + test
    best = torch.load(save_ckpt, map_location="cpu")
    model.load_state_dict(best["model"], strict=False)

    cd1, f1 = evaluate_merge(model, test_loader, device, pred_key="p3_final")
    print("\n[TEST after Stage2e (best EMA)]")
    print(f"p3_final(merge): CD={cd1:.6f} F@0.01={f1[0.01]:.4f} F@0.02={f1[0.02]:.4f}")
    print(f"[Done] best_epoch={best_epoch}, best_score={best_score:.6f}, ckpt={save_ckpt}")


if __name__ == "__main__":
    main()
