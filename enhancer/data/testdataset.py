from enhancer.imports import *
from enhancer.utils import *
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, folder_predicted,folder_target):
        super(TestDataset, self).__init__()
        if folder_predicted[-1]!="/":
            folder_predicted=folder_predicted+"/"
        if folder_target[-1]!="/":
            folder_target=folder_target+"/"
        self.predicted_imgs = sorted([folder_predicted+i for i in os.listdir(folder_predicted)])
        self.target_imgs = sorted([folder_target+i for i in os.listdir(folder_target)])
        self.pre_process = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, index):
        pred_img = self.pre_process(Image.open(self.predicted_imgs[index]))
        target_img = self.pre_process(Image.open(self.target_imgs[index]))
        return {
            "predicted":pred_img,
            "target":target_img
        }
    def __len__(self):
        return len(self.predicted_imgs)