import os
import time
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torch import amp
import kornia.augmentation as K
import numpy as np
from torchvision.transforms.v2 import functional as F

torch.backends.cudnn.benchmark = True
os.makedirs("checkpoints_finetuned", exist_ok=True)

simclr_aug = nn.Sequential(
    K.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur((9, 9), sigma=(0.1, 2.0)),
).cuda()

class SimCLRDataset(Dataset):
    def __init__(self, root):
        self.paths = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.paths.append(os.path.join(dirpath, f))

        print(f"[INFO] Found {len(self.paths)} training images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # FIX: normalize all image shapes
        img = img.resize((224, 224))

        # Convert to tensor safely
        img = F.to_image(img)
        img = F.to_dtype(img, torch.float32, scale=True)

        return img, img


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)       # 2N x D
    z = nn.functional.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature  
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)

    pos = torch.cat([
        torch.arange(N, 2*N),
        torch.arange(0, N)
    ]).to(z.device)

    return nn.CrossEntropyLoss()(sim, pos)


def train_simclr():

    dataset = SimCLRDataset("crops_fine_grained/yolov11_4dh")

    torch.multiprocessing.set_sharing_strategy("file_system")
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # ---- Encoder ----
    encoder = models.resnet50(weights=None)
    dim_mlp = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder = encoder.to("cuda", memory_format=torch.channels_last)

    # ---- Projection head ----
    proj = ProjectionHead(dim_mlp).to("cuda", memory_format=torch.channels_last)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=1e-3, weight_decay=1e-6
    )

    scaler = amp.GradScaler()

    EPOCHS = 100
    print(f"[INFO] Starting SimCLR training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        proj.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for x, _ in pbar:
            x = x.to("cuda", non_blocking=True)

            # GPU augmentations
            x1 = simclr_aug(x)
            x2 = simclr_aug(x)

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                h1 = encoder(x1)
                h2 = encoder(x2)

                z1 = proj(h1)
                z2 = proj(h2)

                loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Loss = {epoch_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(
                {"encoder": encoder.state_dict(), "proj": proj.state_dict()},
                f"checkpoints_finetuned/simclr_epoch_{epoch}.pth"
            )

    print("[DONE] Training complete.")


if __name__ == "__main__":
    train_simclr()
