from enhancer import *

from .options import *
from enhancer.data import create_dataloader, create_dataset
from enhancer.networks import *


def get_dataloader_from_yml(yml_file_path):
    if yml_file_path is None:
        raise Exception("need yml file")
    ops = parse_yml(yml_file_path)
    train_dataset = create_dataset(
        ops["datasets"]["train"]["mode"],
        ops["datasets"]["train"]["dataroot_LQ"],
        ops["datasets"]["train"]["dataroot_GT"],
        ops["datasets"]["train"]["noise_data"],
        ops["datasets"]["train"]["noise_needed"],
        ops["datasets"]["train"]["GT_size"],
        ops["scale"],
    )
    val_dataset = create_dataset(
        ops["datasets"]["val"]["mode"],
        ops["datasets"]["val"]["dataroot_LQ"],
        ops["datasets"]["val"]["dataroot_GT"],
        ops["datasets"]["val"]["noise_data"],
        ops["datasets"]["val"]["noise_needed"],
        ops["datasets"]["val"]["GT_size"],
        ops["scale"],
    )
    train_loader = create_dataloader(
        train_dataset,
        "train",
        ops["datasets"]["train"]["n_workers"],
        ops["datasets"]["train"]["batch_size"],
        ops["datasets"]["train"]["use_shuffle"],
    )
    val_loader = create_dataloader(val_dataset, "validation")
    return {
        "train_dataset": train_dataset,
        "validation_dataset": val_dataset,
        "train_dataloader": train_loader,
        "validation_dataloader": val_loader,
    }


def get_generator_from_yml(yml_file_path):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)

    if opt["name"] == "train_big_model":
        in_c = opt["structure"]["network_G"]["in_c"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        nb = in_c = opt["structure"]["network_G"]["nb"]
        if opt["scale"] == 2:
            model = SuperResolution2x(in_c, out_c, nf, nb)
        elif opt["scale"] == 8:
            model = SuperResolution8x(in_c, out_c, nf, nb)
        elif opt["scale"] == 16:
            model = SuperResolution16x(in_c, out_c, nf, nb)
        else:
            model = SuperResolution4x(in_c, out_c, nf, nb)
    else:
        in_c = opt["structure"]["network_G"]["in_c"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        num_modules = in_c = opt["structure"]["network_G"]["num_modules"]
        scale = opt["scale"]
        if scale in [2, 4, 8, 16]:
            pass
        else:
            scale = 4
        model = RFDN(in_c, nf, num_modules, out_c, scale)

    return model.to(device)


def get_discriminator_from_yml(yml_file_path):
    pass


def get_trainer_from_yml(
    model_G, yml_file_path, train_loader, val_loader=None, model_D=None
):
    pass
