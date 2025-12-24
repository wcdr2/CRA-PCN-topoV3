def build_dataloaders(
    data_root,
    n_partial=2048,
    n_full=16384,
    batch_size=1,
    seed=42,
):
    """
    使用统一的 BuddhaPairDataset，然后做 7:2:1 随机划分。
    """
    full_dataset = BuddhaPairDataset(
        root_dir=data_root,
        n_partial=n_partial,
        n_full=n_full,
    )

    n_total = len(full_dataset)
    assert n_total >= 5, "样本太少，至少需要 5 个样本做 train/val/test 划分"

    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)

    n_train = math.floor(0.7 * n_total)
    n_val = math.floor(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"[Split] train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader