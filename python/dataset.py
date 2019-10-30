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


IGNORE_IMAGE = [
    '9280.png' '3785.png' '4983.png' '1637.png' '5647.png' '6428.png'
    '2712.png' '6789.png' '6010.png' '1019.png' '7326.png' '1030.png'
    '8982.png' '4410.png' '5652.png' '2673.png' '2854.png' '5487.png'
    '8837.png' '6615.png' '9081.png' '3584.png' '3590.png' '4572.png'
    '9693.png' '767.png' '3382.png' '2077.png' '1556.png' '3802.png'
    '6991.png' '4177.png' '6947.png' '3340.png' '9719.png' '5651.png'
    '174.png' '9094.png' '7907.png' '1153.png' '9282.png' '1631.png'
    '170.png' '7042.png' '5735.png' '8577.png' '8946.png' '1989.png'
    '3621.png' '9318.png' '6942.png' '7123.png' '3386.png' '70.png'
    '6413.png' '6349.png' '4990.png' '6188.png' '6175.png' '1168.png'
    '7096.png' '601.png' '8366.png' '4206.png' '985.png' '8560.png' '211.png'
    '1785.png' '3637.png' '4170.png' '7652.png' '1223.png' '8367.png'
    '7915.png' '6174.png' '4993.png' '8857.png' '9552.png' '1640.png'
    '9744.png' '7190.png' '6932.png' '3123.png' '1047.png' '3136.png'
    '6072.png' '9382.png' '2016.png' '7026.png' '9745.png' '6476.png'
    '9792.png' '8895.png' '4049.png' '2160.png' '9009.png' '9141.png'
    '2014.png' '7144.png' '1092.png' '4869.png' '5020.png' '1520.png'
    '3492.png' '2411.png' '6850.png' '8841.png' '1108.png' '5408.png'
    '2987.png' '9636.png' '1915.png' '9346.png' '9352.png' '2207.png'
    '6060.png' '4111.png' '4307.png' '5031.png' '1525.png' '8267.png'
    '5025.png' '6458.png' '7791.png' '7949.png' '7785.png' '8846.png'
    '9594.png' '2364.png' '6102.png' '6499.png' '9972.png' '5147.png'
    '7750.png' '39.png' '11.png' '2984.png' '7368.png' '273.png' '267.png'
    '3325.png' '5185.png' '9230.png' '1885.png' '5407.png' '3062.png'
    '875.png' '6244.png' '3314.png' '4686.png' '9349.png' '1072.png'
    '6906.png' '8254.png' '5770.png' '8446.png' '4526.png' '135.png'
    '7952.png' '3713.png' '8120.png' '9799.png' '9000.png' '8518.png'
    '6737.png' '3882.png' '3869.png' '4690.png' '2785.png' '1298.png'
    '7010.png' '687.png' '7004.png' '4041.png' '5439.png' '2381.png'
    '4086.png' '8643.png' '5011.png' '9617.png' '3845.png' '8050.png'
    '1706.png' '8911.png' '4508.png' '3501.png' '5831.png' '8483.png'
    '4913.png' '6123.png' '119.png' '30.png' '2972.png' '1260.png' '8091.png'
    '6094.png' '3852.png' '3648.png' '5239.png' '9359.png' '8721.png'
    '508.png' '8084.png' '4279.png' '6650.png' '7572.png' '3099.png'
    '2486.png' '4975.png' '6179.png' '2109.png' '3571.png' '2082.png'
    '6582.png' '1560.png' '7474.png' '4155.png' '1944.png' '6597.png'
    '6420.png' '2875.png' '630.png' '1603.png' '2322.png' '6636.png'
    '4747.png' '4974.png' '6187.png' '4786.png' '424.png' '3980.png'
    '3599.png' '632.png' '3200.png' '6740.png' '8023.png' '5513.png'
    '586.png' '9466.png' '97.png' '6392.png' '431.png' '5489.png' '8807.png'
    '7261.png' '7705.png' '44.png' '4542.png' '3417.png' '9106.png'
    '8797.png' '1764.png' '1770.png' '9476.png' '1003.png' '4437.png'
    '2046.png' '2091.png' '8392.png' '5311.png' '408.png' '4796.png'
    '2483.png' '8810.png' '8390.png' '5688.png' '6424.png' '5918.png'
    '2087.png' '2078.png' '4186.png' '7316.png' '2520.png' '594.png'
    '1014.png' '4420.png' '7897.png' '8232.png' '9879.png' '9925.png'
    '6196.png' '423.png']


class MotorbikeDataset(Dataset):
    def __init__(self, path, transform1=None, transform2=None,
                 ignore=IGNORE_IMAGE):
        self.path = path
        img_list = os.listdir(self.path)
        img_list = [i for i in img_list if i not in ignore]
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
