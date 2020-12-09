from enhancer import *
from enhancer.utils import *


def save_epoch_result(epoch, model, image, dest_path=None, normalize=False):
    with torch.no_grad():
        model.eval()
        if dest_path is None:
            dest_path = "./epoch_results"
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        if type(image) == str:
            image = Image.open(image)
            if normalize is True:
                target = "imagenet-norm"
            else:
                target = "[0, 1]"
            test_data = common_converter(image, source="pil", target=target).unsqueeze(
                0
            )
        else:
            test_data = image.unsqueeze(0)
        predicted = model(test_data.to(device))
        numpy_result = im_convert(predicted, denormalize=True)
        pil_image = Image.fromarray((numpy_result * 255).astype(np.uint8))
        pil_image.save(f"{dest_path}/{epoch}_{round(random.random()*1000000)}.png")

        

