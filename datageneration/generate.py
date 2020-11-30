import argparse
import os
import torch.utils.data
import yaml
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(
    description="Apply the trained model to create a dataset"
)
parser.add_argument(
    "--dataset", default="df2k", type=str, help="selecting different datasets"
)
parser.add_argument(
    "--cleanup_factor", default=2, type=int, help="downscaling factor for image cleanup"
)
parser.add_argument(
    "--upscale_factor",
    default=4,
    type=int,
    choices=[4],
    help="super resolution upscale factor",
)
parser.add_argument(
    "--compress", default=True, type=bool, choices=[True, False], help="compress image"
)
opt = parser.parse_args()

with open("./datapaths.yml", "r") as stream:
    PATHS = yaml.load(stream)

path_data = PATHS["datasets"][opt.dataset]
input_source_dir = PATHS[opt.dataset]["source"]
input_target_dir = PATHS[opt.dataset]["target"]
source_files = [
    os.path.join(input_source_dir, x)
    for x in os.listdir(input_source_dir)
    if utils.is_image_file(x)
]
target_files = [
    os.path.join(input_target_dir, x)
    for x in os.listdir(input_target_dir)
    if utils.is_image_file(x)
]

hr_dir = path_data + "/HR"
lr_dir = path_data + "/LR"
if not os.path.exists(hr_dir):
    os.makedirs(hr_dir)
if not os.path.exists(lr_dir):
    os.makedirs(lr_dir)

with torch.no_grad():
    """
    # for file in tqdm(source_files, desc='Generating images from source'):
    #     # load HR image
        input_img = Image.open(file).convert("RGB")
        input_img = TF.to_tensor(input_img)

        # Resize HR image to clean it up and make sure it can be resized again
        resize2_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)
        _, w, h = resize2_img.size()
        w = w - w % opt.upscale_factor
        h = h - h % opt.upscale_factor
        resize2_cut_img = resize2_img[:, :w, :h]

        # Save resize2_cut_img as HR image
        path = os.path.join(hr_dir, os.path.basename(file))
        TF.to_pil_image(resize2_cut_img).save(path, 'PNG')

      #     # Generate resize3_cut_img and apply model
        resize3_cut_img = utils.imresize(resize2_cut_img, 1.0 / opt.upscale_factor, True)

    #     # Save resize3_cut_noisy_img as LR image
        path = os.path.join(lr_dir, os.path.basename(file))
        TF.to_pil_image(resize3_cut_img).save(path, 'PNG')
    """
    for file in tqdm(target_files, desc="Generating images from target"):
        # load HR image
        input_img = Image.open(file).convert("RGB")
        input_img = TF.to_tensor(input_img)
        input_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)

        # Save input_img as HR image
        path = os.path.join(hr_dir, os.path.basename(file))
        TF.to_pil_image(input_img).save(".".join(path.split(".")[:-1]) + ".png", "PNG")

        # generate resized version of input_img
        resize_img = utils.imresize(input_img, 1.0 / opt.upscale_factor, True)

        # Save resize_noisy_img as LR image
        path = os.path.join(lr_dir, os.path.basename(file))
        if opt.compress:
            TF.to_pil_image(resize_img).save(
                ".".join(path.split(".")[:-1]) + ".jpg",
                "JPEG",
                quality=random.randint(2, 10) * 10,
            )
        else:
            TF.to_pil_image(resize_img).save(
                ".".join(path.split(".")[:-1]) + ".png", "PNG"
            )
