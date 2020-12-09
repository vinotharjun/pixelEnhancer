from enhancer import *
import enhancer
from .options import *
from enhancer.data import create_dataloader, create_dataset
from enhancer.networks import *
from enhancer.training import GANTrainer, SimpleTrainer
from enhancer.losses import WassFeatureLoss, FeatureLoss
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
        ops["datasets"]["train"]["noise_rate_random"]
    )
    val_dataset = create_dataset(
        ops["datasets"]["val"]["mode"],
        ops["datasets"]["val"]["dataroot_LQ"],
        ops["datasets"]["val"]["dataroot_GT"],
        ops["datasets"]["val"]["noise_data"],
        ops["datasets"]["val"]["noise_needed"],
        ops["datasets"]["val"]["GT_size"],
        ops["scale"],
        ops["datasets"]["val"]["noise_rate_random"]
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


def get_generator_from_yml(yml_file_path, pretrain_path=None, key=None, strict=True):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)

    if opt["name"] == "train_big_model":
        in_c = opt["structure"]["network_G"]["in_nc"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        nb = opt["structure"]["network_G"]["nb"]
        if opt["scale"] == 2:
            model = SuperResolution2x(in_c, out_c, nf, nb)
        elif opt["scale"] == 8:
            model = SuperResolution8x(in_c, out_c, nf, nb)
        elif opt["scale"] == 16:
            model = SuperResolution16x(in_c, out_c, nf, nb)
        else:
            model = SuperResolution4x(in_c, out_c, nf, nb)
    else:
        in_c = opt["structure"]["network_G"]["in_nc"]
        out_c = opt["structure"]["network_G"]["out_nc"]
        nf = opt["structure"]["network_G"]["nf"]
        num_modules = opt["structure"]["network_G"]["num_modules"]
        scale = opt["scale"]
        model_name = opt["structure"]["network_G"]["which_model_G"]
        if scale in [2, 4, 8, 16]:
            pass
        else:
            scale = 4
        net = importlib.import_module("enhancer.networks.{}".format(model_name)).SmallEnhancer
#         in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4
        model = net(in_nc=in_c, nf=nf, num_modules=num_modules, out_nc=out_c, upscale=scale)
        model = model.to(device)

    if pretrain_path is False:
        return model 
    if pretrain_path is not None:
        if key is not None:
            model.to(device).load_state_dict(torch.load(pretrain_path,map_location=device)[key], strict=strict)
            print(f"generator is loaded from {pretrain_path}")
        else:
            model.to(device).load_state_dict(torch.load(pretrain_path,map_location=device), strict=strict)
            print(f"generator is loaded from {pretrain_path}")
    else:
        if opt["pretraining_settings"]["network_G"]["want_load"] is True:
            pretrain_path = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
            strict = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
            key = opt["pretraining_settings"]["network_G"]["key"]
            if key is not None:
                model.to(device).load_state_dict(torch.load(pretrain_path,map_location=device)[key], strict=strict)
                print(f"generator is loaded from {pretrain_path}")
            else:
                model.to(device).load_state_dict(torch.load(pretrain_path,map_location=device), strict=strict)
                print(f"generator is loaded from {pretrain_path}")
            
                
    return model


def get_discriminator_from_yml(yml_file_path):
    return Discriminator()


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
            load_checkpoint_file_path_G = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
        else:
            load_checkpoint_file_path_G = None
        if opt["pretraining_settings"]["network_D"]["want_load"] == True:
            load_checkpoint_file_path_D = opt["pretraining_settings"]["network_D"]["pretrained_model_path"]
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
            load_checkpoint_file_path_generator =load_checkpoint_file_path_G,
            load_checkpoint_file_path_critic = load_checkpoint_file_path_D,
            sample_interval=sample_interval,
        )
        return trainer
    else:
        if opt["train_settings"]["pixel_criterion"] == "l2":
            criterion = nn.L2Loss()
        elif opt["train_settings"]["pixel_criterion"] == "WassFeatureLoss":
            criterion = WassFeatureLoss()
        elif opt["train_settings"]["pixel_criterion"] == "FeatureLoss":
            criterion = FeatureLoss()
        else:
            criterion = nn.L1Loss()
        adversarial_loss_Weight = opt["train_settings"]["gan_weight"]
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
            load_checkpoint_file_path = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
        else:
            load_checkpoint_file_path = None
        sample_interval = opt["train_settings"]["sample_interval"]

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

    return {"loaders":loaders, "generator":generator, "critic":discriminator, "trainer":trainer}
