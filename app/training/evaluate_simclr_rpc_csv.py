import os
import argparse
import time
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision import models
import torch.optim as optim

# optional
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.svm import LinearSVC
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        hidden = 512

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        # 2–3 residual blocks
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)

        self.fc_out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.fc_out(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLP4(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Dataset helpers
# -------------------------
class FolderDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            p = self.root / cls
            for f in p.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(f), self.class_to_idx[cls]))
        print(f"[INFO] FolderDataset: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB").resize((224, 224))
        # use torchvision.transforms v2 functional pipeline
        img = T.ToImage()(img)                 # uint8 HWC -> tensor CHW uint8
        img = T.ToDtype(torch.float32, scale=True)(img)   # 0-255 -> 0-1 float32
        return img, label, p


class CSVDataset(Dataset):
    def __init__(self, crops_dir, csv_path):
        df = pd.read_csv(csv_path)
        # ensure expected columns
        assert "image_file_name" in df.columns and "label_name" in df.columns
        self.crops_dir = Path(crops_dir)
        df = df.dropna(subset=["image_file_name", "label_name"])
        self.class_names = sorted(df["label_name"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []
        for _, row in df.iterrows():
            p = self.crops_dir / row["image_file_name"]
            if p.exists():
                self.samples.append((str(p), self.class_to_idx[row["label_name"]]))
            else:
                print(f"[WARN] missing {p}")
        print(f"[INFO] CSVDataset: {len(self.samples)} samples, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB").resize((224, 224))
        img = T.ToImage()(img)
        img = T.ToDtype(torch.float32, scale=True)(img)
        return img, label, p


# -------------------------
# Projection head class (must match training)
# -------------------------
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


# -------------------------
# Embedding extraction
# -------------------------
@torch.no_grad()
def extract_embeddings(encoder, proj, loader, device):
    encoder.eval()
    proj.eval()
    all_embs = []
    all_labels = []
    all_paths = []
    pbar = tqdm(loader, desc="Extracting embeddings")
    for imgs, labels, paths in pbar:
        imgs = imgs.to(device, non_blocking=True)
        feats = encoder(imgs)           # 2048-dim features
        feats = feats.cpu().numpy()
        all_embs.append(feats)
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        all_paths.extend(paths)
    embs = np.concatenate(all_embs, axis=0)
    labs = np.concatenate(all_labels, axis=0)
    return embs, labs, all_paths


# -------------------------
# Utility: plot & save confusion matrix
# -------------------------
def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="crops_fine_grained_with_classes/yolov11_4dh")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV with image_file_name,label_name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_finetuned/simclr_epoch_100.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="eval_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # dataset
    if args.csv:
        ds = CSVDataset(args.data_root, args.csv)
        class_names = ds.class_names
    else:
        ds = FolderDataset(args.data_root)
        class_names = ds.classes

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # load model architecture
    print("[INFO] Creating encoder + proj head")
    encoder = models.resnet50(weights=None)
    dim_mlp = encoder.fc.in_features
    encoder.fc = nn.Identity()
    proj = ProjectionHead(dim_mlp)

    # move to device
    encoder = encoder.to(device).eval()
    proj = proj.to(device).eval()

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    # try a few possible key names:
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    elif "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "encoder_state" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state"])
    elif "encoder_weights" in ckpt:
        encoder.load_state_dict(ckpt["encoder_weights"])
    elif "encoder_state_dict" not in ckpt and "encoder" not in ckpt and "visual_state_dict" in ckpt:
        # some scripts saved as "visual_state_dict"
        encoder.load_state_dict(ckpt["visual_state_dict"])
    else:
        # fallback: try to load by names used earlier
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
    # proj keys
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"])
    elif "proj_state_dict" in ckpt:
        proj.load_state_dict(ckpt["proj_state_dict"])
    elif "proj_state" in ckpt:
        proj.load_state_dict(ckpt["proj_state"])
    elif "proj_head" in ckpt:
        proj.load_state_dict(ckpt["proj_head"])
    elif "proj_state_dict" not in ckpt and "proj" not in ckpt and "proj_state" not in ckpt and "proj_state_dict" in ckpt:
        proj.load_state_dict(ckpt["proj_state_dict"])
    else:
        # try keys from earlier training: "proj"
        if "proj" in ckpt:
            proj.load_state_dict(ckpt["proj"])

    print("[OK] Loaded checkpoint:", args.checkpoint)

    # extract embeddings
    emb_file = os.path.join(args.out_dir, "embeddings.npz")
    start = time.time()
    embeddings, labels, paths = extract_embeddings(encoder, proj, loader, device)
    print(f"[INFO] Extracted {embeddings.shape[0]} embeddings in {time.time()-start:.1f}s")
    np.savez_compressed(emb_file, embeddings=embeddings, labels=labels, paths=np.array(paths))
    print("[INFO] Saved embeddings →", emb_file)

    # split train/test
    rng = np.random.RandomState(args.seed)
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=args.seed, stratify=labels
        )
    else:
        # simple stratified split fallback
        from collections import defaultdict
        idx_by_label = defaultdict(list)
        for i, l in enumerate(labels):
            idx_by_label[int(l)].append(i)
        train_idx, test_idx = [], []
        for l, idxs in idx_by_label.items():
            rng.shuffle(idxs)
            cut = max(1, int(0.8 * len(idxs)))
            train_idx += idxs[:cut]
            test_idx += idxs[cut:]
        X_train = embeddings[train_idx]; y_train = labels[train_idx]
        X_test = embeddings[test_idx]; y_test = labels[test_idx]

    # # Train linear classifier
    # if SKLEARN_AVAILABLE:
    #     print("[INFO] Training scikit-learn LogisticRegression (this may take a minute)...")
    #     clf = LogisticRegression(max_iter=2000, C=1.0, verbose=1, n_jobs=-1, multi_class="multinomial", solver="saga")
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     print(f"[RESULT] Linear eval accuracy: {acc*100:.2f}%")
    #     print("\nClassification report:\n")
    #     print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    #     cm = confusion_matrix(y_test, y_pred)
    #     plot_confusion(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))
    #     print("[INFO] Confusion matrix saved to", os.path.join(args.out_dir, "confusion_matrix.png"))

    # else:
    #     print("[WARN] scikit-learn not available — using PyTorch linear head fallback")
    #     device_t = "cuda" if torch.cuda.is_available() else "cpu"
    #     Xtr = torch.from_numpy(X_train).to(device_t)
    #     Ytr = torch.from_numpy(y_train).long().to(device_t)
    #     Xte = torch.from_numpy(X_test).to(device_t)
    #     Yte = torch.from_numpy(y_test).long().to(device_t)
    #     num_classes = int(labels.max() + 1)
    #     head = nn.Linear(Xtr.shape[1], num_classes).to(device_t)
    #     opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.0)
    #     loss_fn = nn.CrossEntropyLoss()
    #     for e in range(200):
    #         head.train()
    #         opt.zero_grad()
    #         logits = head(Xtr)
    #         loss = loss_fn(logits, Ytr)
    #         loss.backward()
    #         opt.step()
    #         if (e+1) % 20 == 0:
    #             print(f"Epoch {e+1}/200 loss={loss.item():.4f}")
    #     head.eval()
    #     with torch.no_grad():
    #         preds = head(Xte).argmax(dim=1).cpu().numpy()
    #     acc = (preds == y_test).mean()
    #     print(f"[RESULT] Linear eval accuracy (torch head): {acc*100:.2f}%")
    #     if len(class_names) <= 200:
    #         print(classification_report(y_test, preds, target_names=class_names))
#----------------------------------------------------------------------------------------------------------
    # if SKLEARN_AVAILABLE:
    #     print("[INFO] Training LinearSVC (fast + high accuracy)...")

    #     clf = LinearSVC(
    #         C=1.0,
    #         max_iter=5000,
    #         verbose=1
    #     )

    #     clf.fit(X_train, y_train)

    #     # Predict
    #     y_pred = clf.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)

    #     print(f"[RESULT] Linear eval accuracy (LinearSVC): {acc*100:.2f}%")
    #     print("\nClassification report:\n")
    #     print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    #     # Confusion matrix
    #     cm = confusion_matrix(y_test, y_pred)
    #     plot_confusion(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))
    #     print("[INFO] Confusion matrix saved to", os.path.join(args.out_dir, "confusion_matrix.png"))

    #     # SAVE THE CLASSIFIER
    #     model_path = os.path.join(args.out_dir, "linear_svc_classifier.joblib")
    #     joblib.dump(clf, model_path)
    #     print(f"[INFO] Saved LinearSVC classifier to: {model_path}")

    # else:
    #     print("[WARN] scikit-learn not available — using PyTorch linear head fallback")
    #     device_t = "cuda" if torch.cuda.is_available() else "cpu"

    #     Xtr = torch.from_numpy(X_train).to(device_t)
    #     Ytr = torch.from_numpy(y_train).long().to(device_t)
    #     Xte = torch.from_numpy(X_test).to(device_t)
    #     Yte = torch.from_numpy(y_test).long().to(device_t)

    #     num_classes = int(labels.max() + 1)
    #     head = nn.Linear(Xtr.shape[1], num_classes).to(device_t)

    #     opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    #     loss_fn = nn.CrossEntropyLoss()

    #     for e in range(200):
    #         head.train()
    #         opt.zero_grad()
    #         logits = head(Xtr)
    #         loss = loss_fn(logits, Ytr)
    #         loss.backward()
    #         opt.step()
    #         if (e + 1) % 20 == 0:
    #             print(f"Epoch {e+1}/200 loss={loss.item():.4f}")

    #     head.eval()
    #     with torch.no_grad():
    #         preds = head(Xte).argmax(dim=1).cpu().numpy()

    #     acc = (preds == y_test).mean()
    #     print(f"[RESULT] Linear eval accuracy (torch head): {acc*100:.2f}%")

    #     # Save PyTorch classifier head
    #     torch.save(head.state_dict(), os.path.join(args.out_dir, "linear_head_classifier.pt"))
    #     print("[INFO] Saved PyTorch classifier head")
#----------------------------------------------------------------------------------------------------------

    # print("\n[INFO] Training 2-Layer MLP classifier...")

    # device_t = "cuda" if torch.cuda.is_available() else "cpu"

    # Xtr = torch.from_numpy(X_train).float().to(device_t)
    # Ytr = torch.from_numpy(y_train).long().to(device_t)
    # Xte = torch.from_numpy(X_test).float().to(device_t)
    # Yte = torch.from_numpy(y_test).long().to(device_t)

    # num_classes = len(class_names)
    # in_dim = Xtr.shape[1]
    # hidden_dim = 512   # You can try 256, 512, 1024

    # model = MLPClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device_t)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # criterion = nn.CrossEntropyLoss()

    # epochs = 50
    # for epoch in range(1, epochs+1):
    #     model.train()
    #     optimizer.zero_grad()

    #     logits = model(Xtr)
    #     loss = criterion(logits, Ytr)
    #     loss.backward()
    #     optimizer.step()

    #     if epoch % 5 == 0:
    #         print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    # # Evaluate
    # model.eval()
    # with torch.no_grad():
    #     preds = model(Xte).argmax(dim=1).cpu().numpy()

    # acc = (preds == y_test).mean()
    # print(f"\n[RESULT] MLP accuracy: {acc*100:.2f}%")

    # # Classification report
    # from sklearn.metrics import classification_report
    # print("\n", classification_report(y_test, preds, target_names=class_names, digits=4))

    # # Save model
    # save_path = os.path.join(args.out_dir, "mlp_classifier.pt")
    # torch.save(model.state_dict(), save_path)
    # print(f"[INFO] Saved MLP model → {save_path}")
#----------------------------------------------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    Yte = torch.tensor(y_test, dtype=torch.long).to(device)

    num_classes = len(set(y_train))
    model = MLP4(X_train.shape[1], num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 50
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, Ytr)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss={loss.item():.4f}")

    # Evaluation
    model.eval()
    preds = model(Xte).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    print("\nMLP-4 Accuracy:", acc*100, "%")
    print(classification_report(y_test, preds))
    torch.save(model.state_dict(), "eval_out/mlp4_classifier.pt")
    print("[INFO] Saved MLP-4 classifier → eval_out/mlp4_classifier.pt")
#----------------------------------------------------------------------------------------------------------


    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        weights="distance"
    )

    knn.fit(X_train, y_train)
    preds_knn = knn.predict(X_test)

    acc_knn = accuracy_score(y_test, preds_knn)
    print("\nKNN (cosine) Accuracy:", acc_knn*100, "%")
    print(classification_report(y_test, preds_knn))
    joblib.dump(knn, "eval_out/knn_cosine_classifier.joblib")
    print("[INFO] Saved kNN classifier → eval_out/knn_cosine_classifier.joblib")

#----------------------------------------------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    Yte = torch.tensor(y_test, dtype=torch.long).to(device)

    num_classes = len(set(y_train))
    resmlp_model = ResMLP(X_train.shape[1], num_classes).to(device)

    optimizer = optim.Adam(resmlp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 50
    for epoch in range(EPOCHS):
        resmlp_model.train()
        optimizer.zero_grad()
        loss = criterion(resmlp_model(Xtr), Ytr)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss={loss.item():.4f}")

    # Evaluation
    resmlp_model.eval()
    preds = resmlp_model(Xte).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    print("\nResidual MLP Accuracy:", acc*100, "%")
    print(classification_report(y_test, preds))
    torch.save(resmlp_model.state_dict(), "eval_out/resmlp_classifier.pt")
    print("[INFO] Saved Residual MLP classifier → eval_out/resmlp_classifier.pt")
    metadata = {
        "input_dim": X_train.shape[1],
        "num_classes": int(y_train.max() + 1),
        "class_names": list(class_names)
    }

    json.dump(metadata, open("eval_out/model_metadata.json", "w"))
    print("[INFO] Saved metadata")

    print("[DONE] evaluation. Artifacts in:", args.out_dir)


if __name__ == "__main__":
    main()
