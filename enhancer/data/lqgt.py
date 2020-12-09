from enhancer.imports import *
from enhancer.utils import *
from .data_utils import *
import torch.utils.data as data
from .noisedataset import noiseDataset


class LQGT(data.Dataset):
    def __init__(
        self,
        lq_path,
        gt_path,
        noiseds_path=None,
        gtsize=128,
        scale=4,
        noise_rate_random=False,
        normalize_noise=True,
    ):
        super().__init__()
        self.GT_SIZE = gtsize
        self.scale = scale
        self.LQ_env, self.GT_env = None, None  # environment for lmdb
        self.data_type = "img"
        self.paths_LQ = None
        self.paths_GT = None
        self.noise_rate_random = noise_rate_random
        if lq_path is not None:
            if type(lq_path) == type(""):
                lq_path = [lq_path] 
            lqpaths = []
            for i in lq_path:
                lqpaths.append(get_image_paths(self.data_type, i)[0])
            self.paths_LQ = sum(lqpaths, [])
        if gt_path is not None :
            if type(gt_path) == type(""):
                gt_path = [gt_path]
            gtpaths = []
            for i in gt_path:
                gtpaths.append(get_image_paths(self.data_type, i)[0])
            self.paths_GT = sum(gtpaths, [])
        
        self.random_scale_list = [1]
        self.noiseds_path = noiseds_path
        if self.noiseds_path is not None:
            self.noises = noiseDataset(
                noiseds_path, self.GT_SIZE / self.scale, normalize=normalize_noise
            )
        else:
            self.noises = None

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.scale
        GT_size = self.GT_SIZE
        # get GT image
        GT_path = self.paths_GT[index]
        resolution = None
        img_GT = read_img(self.GT_env, GT_path, resolution)
        img_GT = channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]
        # get LQ image
        if self.paths_LQ and len(self.paths_LQ)>0:
            LQ_path = self.paths_LQ[index]
            resolution = None
            img_LQ = read_img(self.LQ_env, LQ_path, resolution)
        else:
            img_LQ = imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)
                
            # if the image size is too small
        H, W, _ = img_GT.shape
        if H < GT_size or W < GT_size:
            img_GT = cv2.resize(
                np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR
            )
            # using matlab imresize
            img_LQ = imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        H, W, C = img_LQ.shape
        LQ_size = GT_size // scale

        # randomly crop
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - LQ_size))

        img_LQ = img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]
        rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
        img_GT = img_GT[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]

        img_LQ = channel_convert(C, "RGB", [img_LQ])[0]  # TODO during val no definition

        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()
        # if doesnt matches
        lr_h, lr_w = img_LQ.size()[-1], img_LQ.size()[-2]
        gt_h, gt_w = img_GT.size()[-1], img_GT.size()[-2]

        if gt_h != lr_h * self.scale or gt_w != lr_w * self.scale:
            img_GT = torch.nn.functional.interpolate(
                img_GT.unsqueeze(0), size=(self.GT_SIZE, self.GT_SIZE), mode="bicubic",align_corners=True
            ).squeeze(0)
            img_LQ = torch.nn.functional.interpolate(
                img_LQ.unsqueeze(0),
                size=(self.GT_SIZE // self.scale, self.GT_SIZE // self.scale),
                mode="bicubic",align_corners=True
            ).squeeze(0)
        if self.noise_rate_random ==True:
            noise_mode = bool(random.getrandbits(1))
        else:
            noise_mode = True
        if self.noises is not None and noise_mode == True:
            noise_rnd = np.random.randint(0, len(self.noises))
            noise = self.noises[noise_rnd]
            img_LQ = torch.clamp(img_LQ + noise, 0, 1)
        else:
            noise = None
        if LQ_path is None:
            LQ_path = GT_path
        return {"lr": img_LQ, "hr": img_GT}

    def __len__(self):
        return len(self.paths_GT)
