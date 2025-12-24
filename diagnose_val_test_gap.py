import os, sys, random
import numpy as np
import torch
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

ABS_BUDDHA_ROOT = "/root/autodl-tmp/buddha_completion_project"
if os.path.isdir(ABS_BUDDHA_ROOT):
    sys.path.insert(0, ABS_BUDDHA_ROOT)

# 尝试多路径 import
try:
    from dataset_buddha import BuddhaPairDataset
except ModuleNotFoundError:
    try:
        from datasets.dataset_buddha import BuddhaPairDataset
    except ModuleNotFoundError:
        from buddha_completion_project.dataset_buddha import BuddhaPairDataset

from metrics import chamfer_distance, fscore
from models.crapcn import TopoCRAPCN_V3

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict): return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict): return ckpt["state_dict"]
    return ckpt

@torch.no_grad()
def eval_split(model, loader, device, pred_key, taus=(0.005,0.01,0.02)):
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

def pc_stats(x):
    # x: (N,3) torch
    x = x.detach().cpu().numpy()
    cen = x.mean(0, keepdims=True)
    r = np.sqrt(((x - cen)**2).sum(-1)).max()
    bb = x.max(0) - x.min(0)
    diag = float(np.linalg.norm(bb))
    return float(r), float(diag)

def try_get_ids(ds):
    # 尝试从 dataset 属性里找路径/名字
    cand = []
    for a in ["names","ids","id_list","file_list","files","full_paths","partial_paths","pairs","samples"]:
        if hasattr(ds, a):
            v = getattr(ds, a)
            if isinstance(v, (list,tuple)) and len(v)>0 and isinstance(v[0], str):
                cand = v
                break
    if cand:
        ids = [os.path.splitext(os.path.basename(p))[0] for p in cand]
        return ids

    # fallback：看 __getitem__ 是否带 name/id/path 字段（不保证有）
    ids = []
    for i in range(len(ds)):
        item = ds[i]
        for k in ["name","id","uid","model_id","basename","partial_path","gt_path","full_path","path"]:
            if isinstance(item, dict) and k in item and isinstance(item[k], str):
                ids.append(os.path.splitext(os.path.basename(item[k]))[0])
                break
        else:
            ids.append(str(i))
    return ids

def scale_report(ds, tag):
    rs_gt, dg_gt, rs_p, dg_p = [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        gt = it["gt"]; p = it["partial"]
        if isinstance(gt, np.ndarray): gt = torch.from_numpy(gt)
        if isinstance(p, np.ndarray): p = torch.from_numpy(p)
        r1,d1 = pc_stats(gt)
        r2,d2 = pc_stats(p)
        rs_gt.append(r1); dg_gt.append(d1)
        rs_p.append(r2); dg_p.append(d2)
    print(f"\n[{tag}] scale stats:")
    print(f"  gt   radius mean={np.mean(rs_gt):.4f} std={np.std(rs_gt):.4f}  diag mean={np.mean(dg_gt):.4f}")
    print(f"  part radius mean={np.mean(rs_p):.4f} std={np.std(rs_p):.4f}  diag mean={np.mean(dg_p):.4f}")
    ratio = (np.mean(rs_p) / (np.mean(rs_gt)+1e-9))
    print(f"  part/gt radius ratio (mean) = {ratio:.4f}")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "/root/autodl-tmp/CRA-PCN-main/data/MyDataset"
    ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-stage2-ft-best.pth"
    if not os.path.exists(ckpt):
        ckpt = "./pretrain/pcn/ckpt-buddha-topoV3-ft-best.pth"
    print("[Load ckpt]", ckpt)

    train_ds = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="train")
    val_ds   = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="val")
    test_ds  = BuddhaPairDataset(root_dir=data_root, n_partial=2048, n_full=16384, split="test")

    # 泄漏检查（名字交集）
    tr_ids = set(try_get_ids(train_ds))
    va_ids = set(try_get_ids(val_ds))
    te_ids = set(try_get_ids(test_ds))
    print("\n[Leakage check]")
    print("  |train∩val| =", len(tr_ids & va_ids))
    print("  |train∩test| =", len(tr_ids & te_ids))
    print("  |val∩test| =", len(va_ids & te_ids))

    # 尺度统计
    scale_report(val_ds, "VAL")
    scale_report(test_ds, "TEST")

    val_loader  = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = TopoCRAPCN_V3(
        topo_hidden_dim=128, topo_k=16, topo_layers=2,
        delta_scale=0.2, esp_feat_dim=64, use_topo_v2=True
    ).to(device)
    state = unwrap_state_dict(torch.load(ckpt, map_location="cpu"))
    model.load_state_dict(state, strict=False)

    for key in ["p3_final", "p3_refined"]:
        vcd, vf = eval_split(model, val_loader, device, key)
        tcd, tf = eval_split(model, test_loader, device, key)
        print(f"\n[Output={key}]")
        print(f"  VAL : CD={vcd:.6f}  F@0.01={vf[0.01]:.4f}  F@0.02={vf[0.02]:.4f}")
        print(f"  TEST: CD={tcd:.6f}  F@0.01={tf[0.01]:.4f}  F@0.02={tf[0.02]:.4f}")

if __name__ == "__main__":
    main()
