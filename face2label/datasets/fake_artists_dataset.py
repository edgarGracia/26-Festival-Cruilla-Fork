import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CANONICAL_CSV = str(Path(__file__).resolve().parents[2] / "Fake_Artist.csv")

class FakeArtistsDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str = CANONICAL_CSV, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.artist_df = pd.read_csv(csv_path)

        # Build samples: one entry per artist folder
        # Sample: (list_of_image_paths, artist_name)
        self.samples = []

        for folder_idx in range(2, len(self.artist_df)+2):
            artist_name = self.artist_df.iloc[folder_idx-2, 0]
            folder_path = os.path.join(root_dir, str(folder_idx))

            if not os.path.isdir(folder_path):
                print(f"Folder path not found: {folder_path}")
                continue

            image_paths = sorted([
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ])

            if image_paths:
                self.samples.append((image_paths, artist_name))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, artist_name = self.samples[idx]

        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        return images, artist_name