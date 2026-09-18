import io
import os
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ParquetCIFAR10(Dataset):
    def __init__(self, parquet_path, transform=None):
        table = pq.read_table(parquet_path)
        d = table.to_pydict()
        self.images = d["img"]
        self.labels = d["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_bytes = self.images[idx]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_dataloaders(data_root, batch_size=128):
    train_path = os.path.join(data_root, "plain_text", "train-00000-of-00001.parquet")
    test_path = os.path.join(data_root, "plain_text", "test-00000-of-00001.parquet")

    tf = transforms.ToTensor()

    train_dataset = ParquetCIFAR10(train_path, transform=tf)
    test_dataset = ParquetCIFAR10(test_path, transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
