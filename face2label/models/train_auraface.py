import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from auraface import AuraFaceExtractor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from auraface import train
from datasets.fake_artists_dataset import FakeArtistsDataset
from visualizations.vis_utils import *
from predictor import ArtistPredictor


# ------------------------------------------------
# Config
# ------------------------------------------------
REPO_ROOT = "/home/cvcadmin/cruilla/Festival-Cruilla"
ROOT_DIR   = "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artists"
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.2
HIDDEN_DIM = 256
DROPOUT    = 0.5
weight_decay = 1e-4
IMG_AUG = True
EVAL_ONLY = True

# ------------------------------------------------
# Helper function to get the embeddings
# ------------------------------------------------

def get_embeddings_list(dataset, extractor, label2idx):

    embeddings = []
    for image_paths, artist_name in dataset.samples:
        idx = label2idx[artist_name]

        for path in image_paths:
            embedding = extractor.get_embedding(path)

            if embedding is None:
                embedding = np.zeros(512, dtype=np.float32)

            embeddings.append((embedding, idx))
        
    return embeddings

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Image Augmentation ---
    if IMG_AUG:
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        augmentation = None

    # --- Load dataset ---
    CANONICAL_CSV = "/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artist.csv"
    base_dataset = FakeArtistsDataset(root_dir=ROOT_DIR, csv_path=CANONICAL_CSV, transform=augmentation)

    # Build label: index mapping from artist names — preserve CSV row order
    artist_names = [name for _, name in base_dataset.samples]
    seen: set[str] = set()
    unique_artists: list[str] = []
    for name in artist_names:
        if name not in seen:
            seen.add(name)
            unique_artists.append(name)
    label2idx = {name: i for i, name in enumerate(unique_artists)}
    idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(unique_artists)
    print(f"Artists: {num_classes}")

    # --- Extract embeddings ---
    extractor = AuraFaceExtractor()
    full_dataset = get_embeddings_list(base_dataset, extractor, label2idx)
    print(f"Total images: {len(full_dataset)}")

    # --- Train/val split ---
    embeddings, labels = zip(*full_dataset)
    print("Total Embeddings:", len(embeddings))

    # Visualize model embedding space
    visualize_embeddings(embeddings, labels, method="tsne")
    visualize_embeddings(embeddings, labels, method="pca")

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(np.array(X_train), dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(np.array(X_val), dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
    print(f"Train: {len(X_train)} images | Val: {len(X_val)} images")

    if not EVAL_ONLY:
        # --- Train ---
        model = train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=weight_decay, 
            device=device,
        )

        # --- Visualize predictions ---
        # comment this code if no inference or visualization is needed
        visualize_predictions(
            model=model,
            extractor=extractor,
            samples=base_dataset.samples,
            label2idx=label2idx,
            device=device,
            n=5  
        )

        # --- Save model pth file and idx2label ---
        model_savedir = os.path.join(REPO_ROOT, "face2label/logs")

        model_path = os.path.join(model_savedir, "artists_mlp.pth")
        labels_path = os.path.join(model_savedir, "labels.json")

        torch.save(model.state_dict(), model_path)

        with open(labels_path, "w") as f:
            json.dump(idx2label, f)

    # load only model + labels, skip re-extracting embeddings
    predictor = ArtistPredictor(
        model_path=os.path.join(REPO_ROOT, "face2label/logs/artists_mlp.pth"),
        labels_path=os.path.join(REPO_ROOT, "face2label/logs/labels.json"),
        metadata_path="/home/cvcadmin/cruilla/Festival-Cruilla/Fake_Artist.csv",
    )

    results = predictor.evaluate(val_loader, ks=(1, 3, 5))
    for k, acc in results.items():
        print(f"Top-{k} accuracy: {acc:.2%}")

    predict_topk(
        image_path="/home/cvcadmin/cruilla/Festival-Cruillauser.png",
        model_path=os.path.join(REPO_ROOT, "face2label/logs/artists_mlp.pth"), 
        extractor=extractor,
        idx2label=idx2label,
        label2idx=label2idx,
        dataset=base_dataset,
        device=device,
        k=5
    )

if __name__ == "__main__":
    main()
