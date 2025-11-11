import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CMPFacade(Dataset):
    def __init__(self, root="/dtu/datasets1/02516/CMP_facade_DB_base/base",
                 split="train", size=256, val_split=0.2, seed=42):
        # 🔹 1. Lista di immagini e maschere
        imgs = sorted(glob.glob(os.path.join(root, "*.jpg")))
        masks = sorted(glob.glob(os.path.join(root, "*.png")))
        assert len(imgs) == len(masks), f"Found {len(imgs)} imgs but {len(masks)} masks!"

        # 🔹 2. Crea split train/val
        n = len(imgs)
        idx = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(idx)
        split_idx = int(n * (1 - val_split))

        if split == "train":
            idx = idx[:split_idx]
        elif split == "val":
            idx = idx[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        # 🔹 3. Salva nelle variabili d’istanza (manca nel tuo codice)
        self.imgs = [imgs[i] for i in idx]
        self.masks = [masks[i] for i in idx]

        # 🔹 4. Trasformazioni
        self.tf_img = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.tf_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        x = self.tf_img(img)
        y = np.array(self.tf_mask(mask), dtype=np.int64)

        # 🔹 Clamp valori fuori range per CrossEntropyLoss
        y[y > 11] = 255
        y[y < 0] = 255

        y = torch.from_numpy(y)
        return x, y
