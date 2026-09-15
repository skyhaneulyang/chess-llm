import numpy as np
import torch
import torch.nn as nn
import copy

X = torch.from_numpy(np.load("value_x.npy")).float()
Y = torch.from_numpy(np.load("value_y.npy")).float().unsqueeze(1)

n = X.shape[0]
split = int(n * 0.9)
X_train, X_test = X[:split].cuda(), X[split:].cuda()
Y_train, Y_test = Y[:split].cuda(), Y[split:].cuda()

head = nn.Sequential(
    nn.LayerNorm(X.shape[1]),
    nn.Linear(X.shape[1], 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),
    nn.Sigmoid(),
).cuda()

opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.1)
loss_fn = nn.MSELoss()

BATCH = 256
MAX_EPOCHS = 60
PATIENCE = 8

baseline = ((Y_test - Y_train.mean()) ** 2).mean().item()
print(f"Constant-prediction baseline MSE: {baseline:.4f}\n")

best_mse = float("inf")
best_state = None
best_epoch = 0
best_acc = 0.0

for epoch in range(1, MAX_EPOCHS + 1):
    head.train()
    perm = torch.randperm(X_train.shape[0], device="cuda")
    total = 0.0
    for i in range(0, X_train.shape[0], BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(head(X_train[idx]), Y_train[idx])
        loss.backward()
        opt.step()
        total += loss.item() * idx.shape[0]

    head.eval()
    with torch.no_grad():
        pred = head(X_test)
        test_mse = loss_fn(pred, Y_test).item()
        mask = Y_test.squeeze(1) != 0.5
        acc = (((pred.squeeze(1) > 0.5).float() == Y_test.squeeze(1))[mask]).float().mean().item()

    flag = ""
    if test_mse < best_mse:
        best_mse, best_epoch, best_acc = test_mse, epoch, acc
        best_state = copy.deepcopy(head.state_dict())
        flag = "  *"

    print(f"epoch {epoch:>2}  train {total/X_train.shape[0]:.4f}  "
          f"test {test_mse:.4f}  winner-acc {acc:.1%}{flag}")

    if epoch - best_epoch >= PATIENCE:
        print(f"\nNo improvement in {PATIENCE} epochs — stopping.")
        break

torch.save(best_state, "value_head.pt")
print(f"\nBest: epoch {best_epoch}, test MSE {best_mse:.4f} "
      f"(baseline {baseline:.4f}, {(baseline - best_mse) / baseline:+.1%}), "
      f"winner-acc {best_acc:.1%}")
print("Saved value_head.pt (best checkpoint, not final)")