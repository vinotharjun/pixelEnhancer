import os
import os.path as osp
import logging
import yaml
from enhancer.utils import OrderedYaml
from enhancer.data import create_dataloader, create_dataset

Loader, Dumper = OrderedYaml()


def parse_yml(opt_path):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)
    # dataset parse
    for phase, dataset in opt["datasets"].items():
        dataset["dataroot_GT"] = os.path.abspath(dataset["dataroot_GT"])
        dataset["dataroot_LQ"] = os.path.abspath(dataset["dataroot_LQ"])
        if phase == "train":
            if dataset["noise_needed"] is True and dataset["noise_data"] is not None:
                dataset["noise_data"] = os.path.abspath(dataset["noise_data"])
            else:
                dataset["noise_data"] = None
                dataset["noise_needed"] = False

    # pretrain parse
    for _, path_name in opt["pretraining_settings"].items():
        path_name["pretrained_model_path"] = os.path.abspath(
            path_name["pretrained_model_path"]
        )

    # resume state
    if opt["epoch_settings"]["resume_state_batch"] is None:
        opt["epoch_settings"]["resume_state_batch"] = -1
    if opt["epoch_settings"]["resume_state_epoch"] is None:
        opt["epoch_settings"]["resume_state_epoch"] = 0
    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
