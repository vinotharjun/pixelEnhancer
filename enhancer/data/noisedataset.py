from enhancer.imports import *
from enhancer.utils import *
import torch.utils.data as data


class noiseDataset(data.Dataset):
    def __init__(self, dataset_path, size=32, normalize=True):
        super(noiseDataset, self).__init__()
        self.normalize = normalize
        if type(dataset_path) == type(""):
            dataset_path = [dataset_path]
        noise_paths = []
        for i in dataset_path:
            if i[-1] != "/":
                i = i + "/"
            png_imgs = sorted(glob.glob(i + "*.png"))
            jpg_imgs = sorted(glob.glob(i + "*.jpg"))
            all_imgs = sum([png_imgs, jpg_imgs], [])
            noise_paths.append(all_imgs)
        self.noise_imgs = sum(noise_paths, [])
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