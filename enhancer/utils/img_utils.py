from enhancer.imports import *

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966])
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = (
    torch.FloatTensor([0.485, 0.456, 0.406])
    .to(device)
    .unsqueeze(0)
    .unsqueeze(2)
    .unsqueeze(3)
)
imagenet_std_cuda = (
    torch.FloatTensor([0.229, 0.224, 0.225])
    .to(device)
    .unsqueeze(0)
    .unsqueeze(2)
    .unsqueeze(3)
)

####################
# image convert
####################


def im_convert(
    tensor,
    denormalize=True,
    denormalize_mean=(0.485, 0.456, 0.406),
    denormalize_std=(0.229, 0.224, 0.225),
):
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    if denormalize:
        image = image * np.array(denormalize_std) + np.array(denormalize_mean)
        image = image.clip(0, 1)
    return image


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode="RGB"):
    cv2.imwrite(img_path, img)


def common_converter(img, source, target):
    """
    Convert an image from a source format to a target format.
    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    # assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source

    # assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',

    #                 'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == "pil":
        img = transforms.ToTensor()(img)

    elif source == "[0, 1]":
        pass  # already in [0, 1]
    elif source == "imagenet-norm":
        if img.ndimension() == 3:
            img = (img * imagenet_std) + imagenet_mean

    elif source == "[-1, 1]":
        img = (img + 1.0) / 2.0

    # Convert from [0, 1] to target
    if target == "pil":
        img = F.to_pil_image(img)

    elif target == "[0, 255]":
        img = 255.0 * img

    elif target == "[0, 1]":
        pass  # already in [0, 1]

    elif target == "[-1, 1]":
        img = 2.0 * img - 1.0

    elif target == "imagenet-norm":
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == "y-channel":
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = (
            torch.matmul(255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights)
            / 255.0
            + 16.0
        )

    return img


def load_img(filepath):
    img = Image.open(filepath).convert("YCbCr")
    y, _, _ = img.split()
    return y


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return (
            16.0
            + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])
            / 256.0
        )
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return (
            16.0
            + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :])
            / 256.0
        )
    else:
        raise Exception("Unknown Type", type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = (
            16.0
            + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2])
            / 256.0
        )
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = (
            16.0
            + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :])
            / 256.0
        )
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256.0 + 408.583 * img[:, :, 2] / 256.0 - 222.921
        g = (
            298.082 * img[:, :, 0] / 256.0
            - 100.291 * img[:, :, 1] / 256.0
            - 208.120 * img[:, :, 2] / 256.0
            + 135.576
        )
        b = 298.082 * img[:, :, 0] / 256.0 + 516.412 * img[:, :, 1] / 256.0 - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256.0 + 408.583 * img[2, :, :] / 256.0 - 222.921
        g = (
            298.082 * img[0, :, :] / 256.0
            - 100.291 * img[1, :, :] / 256.0
            - 208.120 * img[2, :, :] / 256.0
            + 135.576
        )
        b = 298.082 * img[0, :, :] / 256.0 + 516.412 * img[1, :, :] / 256.0 - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))
