from torch.utils.data import Dataset
import torchvision.transforms as T


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * 4

    def __getitem__(self, idx):
        img, label = self.get_flipped_image(idx)
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_flipped_image(self, idx):
        image_idx = idx // 4
        augment_idx = idx % 4
        img, label = self.dataset[image_idx]
        if augment_idx == 1:  # Horizontal Flip
            img = T.functional.hflip(img)
        elif augment_idx == 2:  # Vertical Flip
            img = T.functional.vflip(img)
        elif augment_idx == 3:  # Horizontal + Vertical Flip
            img = T.functional.hflip(T.functional.vflip(img))
        return img,label