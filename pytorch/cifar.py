from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split


class CIFARData:
    def __init__(self, validation=False):
        self.train_dataset = datasets.CIFAR10(
            root="C:\Project\Data\\",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.test_dataset = datasets.CIFAR10(
            root="C:\Project\Data\\",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.validation = validation

    @staticmethod
    def train_test_split(train_data):
        dataset_size = len(train_data)
        train_size = int(dataset_size * 0.8)
        validation_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(
            train_data, [train_size, validation_size]
        )
        return train_dataset, val_dataset

    def load(self):
        train_data = self.train_dataset
        if self.validation:
            train_data, val_data = self.train_test_split(self.train_dataset)
            val_dataloader = DataLoader(val_data, shuffle=True, batch_size=128)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=128)
        test_dataloader = DataLoader(self.test_dataset, batch_size=128)

        return (train_dataloader, val_dataloader, test_dataloader) if self.validation else (train_dataloader, test_dataloader)


if __name__ == "__main__":
    CIFAR = CIFARData(validation=False)
    train, val, test = CIFAR.load()


