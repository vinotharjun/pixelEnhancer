import torch


def create_dataloader(
    dataset, phase, num_workers=4, batch_size=1, is_shuffle=True, sampler=None
):
    if phase == "train":
        shuffle = is_shuffle
        return torch.utils.data.DataLoader(
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
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )


def create_dataset(mode, lq_path, gt_path, noiseds_path, noise_needed, gtsize, scale):
    if mode == "LQGT":
        from .lqgt import LQGT as D
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))
    if noise_needed is False:
        noiseds_path = None
    ds = D(lq_path, gt_path, noiseds_path, gtsize, scale)
    return ds