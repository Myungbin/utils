from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kfold.split(train.dataset)):
    print(f"Fold {i}")

    train_ds = Subset(train.dataset, train_idx)
    val_ds = Subset(train.dataset, val_idx)
    train_dl = DataLoader(train_ds, batch_size=n_batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=n_batch, shuffle=False)

    for epoch in range(epochs):
        train_step
        val_step
        ...