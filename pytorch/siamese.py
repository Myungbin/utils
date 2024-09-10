from torch.utils.data import Dataset, DataLoader, random_split


class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __getitem__(self, index):
        img1_path, label1 = random.choice(self.image_folder.imgs)
        img1 = Image.open(img1_path).convert("RGB")

        same_class = random.randint(0, 1)

        if same_class:
            while True:
                img2_path, label2 = random.choice(self.image_folder.imgs)
                if label1 == label2:
                    break
        else:
            while True:
                img2_path, label2 = random.choice(self.image_folder.imgs)
                if label1 != label2:
                    break

        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(same_class, dtype=torch.float32)

    def __len__(self):
        return len(self.image_folder)


class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def _choose_image(self, label1, same_class):
        while True:
            img_path, label = random.choice(self.image_folder.imgs)
            if (same_class and label1 == label) or (not same_class and label1 != label):
                return img_path

    def __getitem__(self, index):
        img1_path, label1 = random.choice(self.image_folder.imgs)
        img1 = Image.open(img1_path).convert("RGB")

        same_class = random.randint(0, 1)
        img2_path = self._choose_image(label1, same_class)
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(same_class, dtype=torch.float32).to("cpu")

    def __len__(self):
        return len(self.image_folder)

    @staticmethod
    def img_load(data_path, image_list):
        label_map = os.listdir(data_path)
        paths = []
        labels = []

        for file in image_list:
            label_name = file.split("\\")[-2]
            labels.append(label_map.index(label_name))
            paths.append(file)
        return paths, labels
