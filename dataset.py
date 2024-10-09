from datasets import load_dataset
import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split

import os


class ImageDataset(Dataset):
    def __init__(self, pil_dataset, transform):
        self.pil_dataset = []
        self.pil = pil_dataset["train"]
        self.transform = transform

        for x in self.pil:
            if x['image'].size == (50,50):
                self.pil_dataset.append(x)

    def __len__(self):
        return len(self.pil_dataset)

    def __getitem__(self, idx):
        item = self.pil_dataset[idx]
        image = item["image"]
        label = item["label"]

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label)

        return image_tensor, label_tensor

    def __repr__(self):
        return self.pil_dataset.__repr__()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        #),
    ]
)


def image_dataset():
    cache_dir = "datasets_cache"
    train_path = os.path.join(cache_dir, "train_dataset.pt")
    val_path = os.path.join(cache_dir, "val_dataset.pt")
    test_path = os.path.join(cache_dir, "test_dataset.pt")

    if (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        train_dataset = torch.load(train_path)
        val_dataset = torch.load(val_path)
        test_dataset = torch.load(test_path)
        print("Loaded datasets from cache.")
    else:
        
        os.makedirs(cache_dir, exist_ok=True)

        dataset = load_dataset(
            "EulerianKnight/breast-histopathology-images-train-test-valid-split"
        )
        dataset = ImageDataset(dataset, transform)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)
        print("Saved datasets to cache.")

    return train_dataset, val_dataset, test_dataset
