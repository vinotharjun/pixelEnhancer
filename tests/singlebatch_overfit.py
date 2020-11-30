import sys

sys.path.append("../")
from enhancer import *
from enhancer.networks.esrgan import SuperResolution2x
from enhancer.networks.rfdn import RFDN
from enhancer.data.lqgt import LQGT
from enhancer.utils import getsample
from fastprogress.fastprogress import master_bar, progress_bar


def overfit_on_single_batch(batched_data, model, criterion, optimizer):
    model.train()
    mb = progress_bar(range(0, 140))
    for e in mb:
        input_image = batched_data["lr"].to(device).float()
        target = batched_data["hr"].to(device).float()
        optimizer.zero_grad()
        outputs = model(input_image)
        loss_outputs = criterion(outputs, target)
        loss_outputs.backward()
        optimizer.step()
        print(f"Finished loop {e}    :     Loss :{loss_outputs.item()}")


if __name__ == "__main__":

    print("loading model")
    # model = SuperResolution2x().to(device)
    model = RFDN(upscale=2).to(device)
    print("loading loss")
    criterion = nn.L1Loss()
    print("loading optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    print("loading dataset")
    dataset = LQGT(
        ".././dataset/df2k/LR",
        ".././dataset/df2k/HR",
        None,
        gtsize=48,
        scale=2,
        normalize_noise=False,
    )
    print("loading dataloaedr")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, num_workers=4, pin_memory=True
    )
    print("getsample")
    batched_data = getsample(data_loader)
    print("start tests")
    overfit_on_single_batch(batched_data, model, criterion, optimizer)