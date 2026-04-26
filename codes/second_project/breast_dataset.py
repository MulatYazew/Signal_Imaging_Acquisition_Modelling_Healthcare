import cv2
import numpy as np 
import torch
from torch.utils.data import Dataset


class BUSIUNetDataset(Dataset):
    def __init__(self, df, augment=None, image_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.image_size = image_size  # (W, H) for cv2.resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image: {row['image_path']}")
        if mask is None:
            raise ValueError(f"Could not read mask: {row['mask_path']}")

        # First resize everything to a common size
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Then apply joint augmentation if provided
        if self.augment is not None:
            augmented = self.augment(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Safety check
        if img.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes differ: img={img.shape}, mask={mask.shape}, "
                f"paths=({row['image_path']}, {row['mask_path']})"
            )

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        x = np.stack([img], axis=0)   # [2, H, W]
        y = np.expand_dims(mask, axis=0)        # [1, H, W]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)