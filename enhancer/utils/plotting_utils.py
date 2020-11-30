from .common_utils import *
from .img_utils import *
from enhancer import *


def show_batches(tensor, n_row=0, n_col=0, denormalize=True, figsize=(12, 12)):
    try:
        batch_size = len(tensor)
        # if len(tensor.size()) != 4:
        #     raise Exception("expected 4 dimensional tensor")
        if not n_row or not n_col:
            raise Exception("expected the number of rows and columns")
        if batch_size != n_row * n_col:
            raise Exception("column or row is invalid count")
        if type(tensor) != type(torch.tensor(1)):
            raise Exception("expected tensor")
        else:

            _, fig = plt.subplots(n_row, n_col, figsize=figsize)
            fig = fig.flatten()
            i = 0
            for f in fig:
                if denormalize:
                    f.imshow(im_convert(tensor[i]))
                    plt.plot()
                else:

                    f.imshow(im_convert(tensor[i], denormalize=False))
                    plt.plot()
                i = i + 1
    except Exception as e:
        print(e)


def versus_plot(
    input_tensor,
    target_tensor,
    preupscaling=True,
    figsize=(10, 10),
    denormalize=True,
    i_text="input image",
    t_text="target image",
):
    if preupscaling:
        input_tensor = F.interpolate(
            input_tensor.unsqueeze(0), target_tensor.shape[-1], mode="bicubic"
        ).squeeze(0)
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].set_title(i_text)
    axarr[0].imshow(im_convert(input_tensor, denormalize=denormalize))

    f.tight_layout()

    axarr[1].set_title(t_text)
    axarr[1].imshow(im_convert(target_tensor, denormalize=denormalize))


def batch_versus_plot(
    input_tensor,
    target_tensor,
    batch_limit=None,
    preupscaling=True,
    figsize=(10, 10),
    denormalize=True,
    i_text="input image",
    t_text="target image",
):
    if batch_limit == None or batch_limit > input_tensor.shape[0]:
        batch_limit = input_tensor.shape[0]
    if preupscaling:
        input_tensor = F.interpolate(
            input_tensor, target_tensor.shape[-1], mode="bicubic"
        )
    f, axarr = plt.subplots(input_tensor.shape[0], 2, figsize=figsize)
    for i in range(batch_limit):
        if i == 0:
            axarr[i][0].set_title(i_text)
            axarr[i][1].set_title(t_text)
        axarr[i][0].imshow(im_convert(input_tensor[i], denormalize=denormalize))
        axarr[i][1].imshow(im_convert(target_tensor[i], denormalize=denormalize))

        f.tight_layout()


def n_versus_plot(
    data, texts, norm_req, figsize=(10, 10), preupscaling=True, preupscaling_idx=0
):
    if preupscaling:
        data[preupscaling_idx] = F.interpolate(
            data[preupscaling_idx].unsqueeze(0),
            data[preupscaling_idx].shape[-1],
            mode="bicubic",
        ).squeeze(0)
    f, axarr = plt.subplots(1, len(data), figsize=figsize)
    for i, image in enumerate(data):
        axarr[i].set_title(texts[i])
        axarr[i].imshow(im_convert(image, denormalize=norm_req[i]))
        f.tight_layout()


def show_comparision(
    input_image,
    target_image,
    bicubic_result,
    model_result,
    preupscaling=True,
    zoom=False,
):
    if preupscaling:
        input_image = F.interpolate(input_image.unsqueeze(0), 144)
        input_image = input_image.squeeze(0)
    f, axarr = plt.subplots(1, 4, figsize=(10, 10))

    axarr[0].set_title("input image")
    axarr[0].imshow(im_convert(input_image))

    f.tight_layout()
    axarr[1].set_title("target image")
    axarr[1].imshow(im_convert(target_image))
    if zoom:
        number = randint(40, 100)
        axarr[1].axis(xmin=number - 30, xmax=number)
        axarr[1].axis(ymin=number - 30, ymax=number)
    f.tight_layout()
    axarr[2].set_title("bicubic_result")
    axarr[2].imshow(im_convert(bicubic_result))
    if zoom:
        axarr[2].axis(xmin=number - 30, xmax=number)
        axarr[2].axis(ymin=number - 30, ymax=number)
    f.tight_layout()
    axarr[3].set_title("model result")
    axarr[3].imshow(im_convert(model_result))
    if zoom:
        axarr[3].axis(xmin=number - 30, xmax=number)
        axarr[3].axis(ymin=number - 30, ymax=number)


def show(img):
    npimg = img.cpu().clone().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).clip(0, 1), interpolation="nearest")
