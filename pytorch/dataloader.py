import torchvision.datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import cfg


class DatasetLoader:
    def __init__(self, num_workers=0):
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = torchvision.datasets.ImageFolder(cfg.TRAIN_PATH, transform=self.transform)
        self.shuffle = True

    @property
    def train_test_split(self):
        dataset_size = len(self.train_dataset)
        train_size = int(dataset_size * cfg.TRAIN_SIZE)
        validation_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(self.train_dataset, [train_size, validation_size])
        return train_dataset, val_dataset

    @property
    def init_dataloader(self):
        train_dataset, val_dataset = self.train_test_split
        train_dataloader = DataLoader(train_dataset, shuffle=self.shuffle, batch_size=cfg.BATCH_SIZE,
                                      pin_memory=True, num_workers=self.num_workers)
        val_dataloader = DataLoader(val_dataset, shuffle=self.shuffle, batch_size=cfg.BATCH_SIZE,
                                    pin_memory=True, num_workers=self.num_workers)

        return train_dataloader, val_dataloader

    def load(self):
        train_loader, val_loader = self.init_dataloader
        return train_loader, val_loader


if __name__ == "__main__":
    ...
