import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    if phase == "train":
        num_workers = dataset_opt["n_workers"]
        batch_size = dataset_opt["batch_size"]
        shuffle = True
        return torch.utils.data.DataLoader (
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )


def create_dataset(opt):
    if mode == opt["mode"]:
        from .lqgt import LGQT as D
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))
    lq_path = opt["lr_path"]
    gt_path = opt["hr_path"]
    noiseds_path = opt["noise_path"]
    noise_needed = opt["noise_enable"]
    if noise_needed is False:
        noiseds_path = None
    gtsize = opt["crop_size"]
    scale = opt["scale"]
    dataset = D(lq_path, gt_path, noiseds_path, gtsize, scale))
    return dataset