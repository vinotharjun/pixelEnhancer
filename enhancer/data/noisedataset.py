from enhancer.imports import *
from enhancer.utils import *
import torch.utils.data as data


class noiseDataset(data.Dataset):
    def __init__(self, dataset_path, size=32, normalize=True):
        super(noiseDataset, self).__init__()
        self.path = dataset_path
        self.normalize = normalize
        if self.path[-1] != "/":
            self.path = self.path + "/"
        self.noise_imgs_png = sorted(glob.glob(self.path + "*.png"))
        self.noise_imgs_jpg = sorted(glob.glob(self.path + "*.jpg"))
        self.noise_imgs = sum([self.noise_imgs_png, self.noise_imgs_jpg], [])
        self.pre_process = transforms.Compose(
            [transforms.RandomCrop(size), transforms.ToTensor()]
        )

    def __getitem__(self, index):
        noise = self.pre_process(Image.open(self.noise_imgs[index]))
        if self.normalize:
            noise = noise - torch.mean(noise, dim=[1, 2], keepdim=True)
        return noise

    def __len__(self):
        return len(self.noise_imgs)