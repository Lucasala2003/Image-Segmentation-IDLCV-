import torch, os
from torch.utils.data import DataLoader
from torch import nn, optim
from lib.dataset.CMPFacade import CMPFacade
from lib.model.UNetModel import UNet_multi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size = 256; bs = 4; epochs = 50

trainset = CMPFacade(split="train", size=size)
valset   = CMPFacade(split="val",   size=size)
train_loader = DataLoader(trainset, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(valset,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

model = UNet_multi().to(device)   # assicurati out=12 nella classe
criterion = nn.CrossEntropyLoss(ignore_index=255)   # (opz.) ignore_index=255 se ci sono pixel ignoti
opt = optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, epochs+1):
    model.train(); run_loss = 0.0
    for x,y in train_loader:
        x, y = x.to(device), y.to(device)              # y: LongTensor H×W (no one-hot)
        opt.zero_grad()
        logits = model(x)                               # (N,12,H,W)
        loss = criterion(logits, y)
        loss.backward(); opt.step()
        run_loss += loss.item()
    print(f"Ep {ep}/{epochs} - loss: {run_loss/len(train_loader):.4f}")

    # mini-val mIoU grezza (facoltativa, velocissima)
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)                  # (N,H,W)
            correct += (pred==y).sum().item()
            total   += y.numel()
    print(f"  Val pixel-acc: {correct/total:.3f}")

os.makedirs("runs", exist_ok=True)
torch.save(model.state_dict(), "runs/unet_facade_12cls.pth")
print("Done.")
