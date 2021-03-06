from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml
import random

parser = argparse.ArgumentParser(description="create a dataset")
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
    choices=[4, 2, 8, 16],
    help="super resolution upscale factor",
)
parser.add_argument(
    "--compress", default=0, type=int, choices=[0, 1], help="compress image"
)
opt = parser.parse_args()

# define input and target directories
with open("./datapaths.yml", "r") as stream:
    PATHS = yaml.load(stream)


def noise_patch(rgb_img, sp, max_var, min_mean):
    img = rgb_img.convert("L")
    rgb_img = np.array(rgb_img)
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i : i + sp, j : j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            # print(var_global, mean_global)

            if var_global < max_var and mean_global > min_mean:
                print(var_global, mean_global)
                rgb_patch = rgb_img[i : i + sp, j : j + sp, :]
                collect_patchs.append(rgb_patch)

    return collect_patchs


if __name__ == "__main__":

    if opt.dataset == "coco":
        img_dir = PATHS[opt.dataset]["source"]
        noise_dir = PATHS["datasets"]["coco"] + "/Corrupted_noise"
        sp = 256
        max_var = 20
        min_mean = 0
    else:
        img_dir = PATHS[opt.dataset]["source"]
        noise_dir = PATHS["datasets"][opt.dataset] + "/Corrupted_noise"
        sp = 256
        max_var = 20
        min_mean = 0

    if not os.path.exists(noise_dir):
        os.mkdir(noise_dir)

    img_paths_png = sorted(glob.glob(osp.join(img_dir, "*.png")))
    img_paths_jpeg = sorted(glob.glob(osp.join(img_dir, "*.jpg")))
    img_paths = sum([img_paths_png, img_paths_jpeg], [])
    cnt = 0
    for i, path in enumerate(img_paths):
        img_name = osp.splitext(osp.basename(path))[0]
        print("**********", i, img_name, "**********")
        img = Image.open(path).convert("RGB")
        patchs = noise_patch(img, sp, max_var, min_mean)
        for idx, patch in enumerate(patchs):
            save_path = osp.join(noise_dir, "{}_{:03}.png".format(img_name, idx))
            cnt += 1
            print("collect:", cnt, save_path)
            if opt.compress == 1:
                save_path = ".".join(save_path.split(".")[:-1]) + ".jpg"
                Image.fromarray(patch).save(save_path, "JPEG", quality=70)
            else:
                Image.fromarray(patch).save(save_path)
