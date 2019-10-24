import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms(image_size=128):
    mean = 0.5, 0.5, 0.5
    std = 0.5, 0.5, 0.5

    transform1 = transforms.Compose([transforms.Resize(image_size)])
    transform2 = transforms.Compose([transforms.RandomCrop(image_size),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std),
                                     ])
    return transform1, transform2


class MotorbikeDataset(Dataset):
    def __init__(self, path, transform1=None, transform2=None):
        self.path = path
        img_list = os.listdir(self.path)
        self.img_list = img_list
        self.transform1 = transform1
        self.transform2 = transform2
        self.load_data(img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform2 is not None:
            img = self.transform2(img)
        return img

    def load_data(self, img_list):
        self.images = []

        for idx, path in enumerate(img_list):
            origin_img = Image.open(os.path.join(self.path, self.img_list[idx]))
            img = origin_img.copy()
            origin_img.close()
            img = self.transform1(img)
            self.images.append(img)
        return self.images

    def sample(self, n=5):
        return [self.__getitem__(i) for i in range(n)]


