from enhancer import *
import enhancer
from .options import *
from enhancer.data import create_dataloader, create_dataset
from enhancer.networks import *
from enhancer.training import GANTrainer, SimpleTrainer
from enhancer.losses import LossCalculator
import importlib


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
        ops["datasets"]["train"]["noise_rate_random"],
    )
    val_dataset = create_dataset(
        ops["datasets"]["val"]["mode"],
        ops["datasets"]["val"]["dataroot_LQ"],
        ops["datasets"]["val"]["dataroot_GT"],
        ops["datasets"]["val"]["noise_data"],
        ops["datasets"]["val"]["noise_needed"],
        ops["datasets"]["val"]["GT_size"],
        ops["scale"],
        ops["datasets"]["val"]["noise_rate_random"],
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


def load_partial(generator, path, strict=False, key=None):
    model_dict = generator.state_dict()
    if key is not None:
        pretrained_dict = torch.load(path, map_location=device)[key]
    else:
        pretrained_dict = torch.load(path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    generator.load_state_dict(model_dict, strict=strict)
    return generator


def get_generator_from_yml(yml_file_path, pretrain_path=None, key=None, strict=True):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)

    if opt["name"] == "train_big_model":
        in_c = opt["structure"]["network_G"]["in_nc"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        num_modules = opt["structure"]["network_G"]["num_modules"]
        scale = opt["scale"]
        model_name = opt["structure"]["network_G"]["which_model_G"]
        if "gc" in opt["structure"]["network_G"]:
            gc = opt["structure"]["network_G"]["gc"]
        else:
            gc = 32
        if scale in [2, 4, 8, 16, 3, 6]:
            pass
        else:
            scale = 4
        net = importlib.import_module(
            "enhancer.networks.{}".format(model_name)
        ).SmallEnhancer
        model = net(
            in_nc=in_c,
            nf=nf,
            num_modules=num_modules,
            out_nc=out_c,
            upscale=scale,
            gc=gc,
        )
        model = model.to(device)
    else:
        in_c = opt["structure"]["network_G"]["in_nc"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        num_modules = opt["structure"]["network_G"]["num_modules"]
        scale = opt["scale"]
        model_name = opt["structure"]["network_G"]["which_model_G"]
        if scale in [2, 4, 8, 16, 3, 6]:
            pass
        else:
            scale = 4
        net = importlib.import_module(
            "enhancer.networks.{}".format(model_name)
        ).SmallEnhancer
        model = net(
            in_nc=in_c, nf=nf, num_modules=num_modules, out_nc=out_c, upscale=scale
        )
        model = model.to(device)

    if pretrain_path is False:
        return model
    if pretrain_path is not None:
        model = load_partial(model.to(device), pretrain_path, strict, key)
        print(f"generator is loaded from {pretrain_path}")
        model.to(device)
    else:
        if opt["pretraining_settings"]["network_G"]["want_load"] is True:
            pretrain_path = opt["pretraining_settings"]["network_G"][
                "pretrained_model_path"
            ]
            strict = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
            key = opt["pretraining_settings"]["network_G"]["key"]
            model = load_partial(model.to(device), pretrain_path, strict, key)
            print(f"generator is loaded from {pretrain_path}")
            model.to(device)
    return model


def get_discriminator_from_yml(
    yml_file_path, pretrain_path=None, key=None, strict=True
):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)
    if "network_D" not in opt["pretraining_settings"]:
        return
    model = Discriminator().to(device)
    if pretrain_path is False:
        return model
    if pretrain_path is not None:
        if key is not None:
            model.to(device).load_state_dict(
                torch.load(pretrain_path, map_location=device)[key], strict=strict
            )
            print(f"critic is loaded from {pretrain_path}")
        else:
            model.to(device).load_state_dict(
                torch.load(pretrain_path, map_location=device), strict=strict
            )
            print(f"critic is loaded from {pretrain_path}")
    else:
        if opt["pretraining_settings"]["network_D"]["want_load"] is True:
            pretrain_path = opt["pretraining_settings"]["network_D"][
                "pretrained_model_path"
            ]
            strict = opt["pretraining_settings"]["network_D"]["pretrained_model_path"]
            key = opt["pretraining_settings"]["network_D"]["key"]
            if key is not None:
                model.to(device).load_state_dict(
                    torch.load(pretrain_path, map_location=device)[key], strict=strict
                )
                print(f"critic is loaded from {pretrain_path}")
            else:
                model.to(device).load_state_dict(
                    torch.load(pretrain_path, map_location=device), strict=strict
                )
                print(f"critic is loaded from {pretrain_path}")
    return model


def get_loss(loss_details):
    if type(loss_details) != type([]):
        raise Exception("expecting loss details as array of tuples")
    all_losses = []
    for loss_data, loss_name in loss_details:
        loss = importlib.import_module("enhancer.losses.{}".format(loss_name)).Loss
        all_losses += [loss(**loss_data)]
    return LossCalculator(loss_details=all_losses)


def get_trainer_from_yml(
    yml_file_path,
    model_G,
    train_loader,
    val_loader=None,
    model_D=None,
    top_score=None,
):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)

    if opt["type"] == "gan":
        if opt["train_settings"]["pixel_criterion"] is not None:
            criterion = get_loss(opt["train_settings"]["pixel_criterion"])
        else:
            raise Exception("needed loss details")
        adversarial_loss_Weight = opt["train_settings"]["gan_weight"]
        opt_beta1 = opt["train_settings"]["beta1_G"]
        opt_beta2 = opt["train_settings"]["beta2_G"]
        wd = opt["train_settings"]["weight_decay_G"]
        lr_G = opt["train_settings"]["lr_G"]
        lr_D = opt["train_settings"]["lr_D"]
        if top_score is None:
            top_score = opt["train_settings"]["top_score"]
        save_checkpoint_folder_path = opt["train_settings"][
            "save_checkpoint_folder_path"
        ]
        save_checkpoint_file_name = opt["train_settings"]["save_checkpoint_file_name"]
        save_bestmodel_file_name = opt["train_settings"]["save_bestmodel_file_name"]
        if opt["pretraining_settings"]["network_G"]["want_load"] == True:
            load_checkpoint_file_path_G = opt["pretraining_settings"]["network_G"][
                "pretrained_model_path"
            ]
        else:
            load_checkpoint_file_path_G = None
        if opt["pretraining_settings"]["network_D"]["want_load"] == True:
            load_checkpoint_file_path_D = opt["pretraining_settings"]["network_D"][
                "pretrained_model_path"
            ]
        else:
            load_checkpoint_file_path_D = None

        #         load_checkpoint_file_path = opt["train_settings"]["load_checkpoint_file_path"]
        sample_interval = opt["train_settings"]["sample_interval"]
        trainer = GANTrainer(
            generator=model_G,
            discriminator=model_D,
            train_loader=train_loader,
            val_loader=val_loader,
            adversarial_loss_weight=adversarial_loss_Weight,
            opt_beta1=opt_beta1,
            opt_beta2=opt_beta2,
            wd=wd,
            lr_G=lr_G,
            lr_D=lr_D,
            top_ssim=top_score,
            save_checkpoint_folder_path=save_checkpoint_folder_path,
            save_checkpoint_file_name=save_checkpoint_file_name,
            save_best_file_name=save_bestmodel_file_name,
            load_checkpoint_file_path_generator=load_checkpoint_file_path_G,
            load_checkpoint_file_path_critic=load_checkpoint_file_path_D,
            sample_interval=sample_interval,
            feature_criterion=criterion,
        )
        return trainer
    else:
        if opt["train_settings"]["pixel_criterion"] is not None:
            criterion = get_loss(opt["train_settings"]["pixel_criterion"])
        else:
            raise Exception("needed loss details")

        opt_beta1 = opt["train_settings"]["beta1_G"]
        opt_beta2 = opt["train_settings"]["beta2_G"]
        wd = opt["train_settings"]["weight_decay_G"]
        lr = opt["train_settings"]["lr_G"]
        if top_score is None:
            top_score = opt["train_settings"]["top_score"]
        save_checkpoint_folder_path = opt["train_settings"][
            "save_checkpoint_folder_path"
        ]
        save_checkpoint_file_name = opt["train_settings"]["save_checkpoint_file_name"]
        save_bestmodel_file_name = opt["train_settings"]["save_bestmodel_file_name"]
        if opt["pretraining_settings"]["network_G"]["want_load"] == True:
            load_checkpoint_file_path = opt["pretraining_settings"]["network_G"][
                "pretrained_model_path"
            ]
        else:
            load_checkpoint_file_path = None
        sample_interval = opt["train_settings"]["sample_interval"]
        lr_step_decay = opt["train_settings"]["lr_step_decay"]

        trainer = SimpleTrainer(
            generator=model_G,
            train_loader=train_loader,
            val_loader=val_loader,
            opt_beta1=opt_beta1,
            opt_beta2=opt_beta2,
            wd=wd,
            lr=lr,
            top_psnr=top_score,
            save_checkpoint_folder_path=save_checkpoint_folder_path,
            save_checkpoint_file_name=save_checkpoint_file_name,
            save_best_file_name=save_bestmodel_file_name,
            criterion=criterion,
            load_checkpoint_file_path=load_checkpoint_file_path,
            sample_interval=sample_interval,
            lr_step_decay=lr_step_decay,
        )
        return trainer


def load_pipeline_from_yml(yml_file_path):
    # load dataloader
    print("loading dataloader...")
    loaders = get_dataloader_from_yml(yml_file_path)
    train_loader = loaders["train_dataloader"]
    val_loader = loaders["validation_dataloader"]

    # load model
    print("loading models...")
    generator = get_generator_from_yml(yml_file_path)
    # load discriminator
    discriminator = get_discriminator_from_yml(yml_file_path)
    # load trainer
    print("constructing trainers .....")
    trainer = get_trainer_from_yml(
        yml_file_path=yml_file_path,
        model_G=generator,
        train_loader=train_loader,
        val_loader=val_loader,
        model_D=discriminator,
    )

    return {
        "loaders": loaders,
        "generator": generator,
        "critic": discriminator,
        "trainer": trainer,
    }
