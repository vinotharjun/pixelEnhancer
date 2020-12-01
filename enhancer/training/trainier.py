from enhancer import *
from enhancer.losses import WassFeatureLoss, FeatureLoss
from enhancer.utils import *
from enhancer.inference import *


class GANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        val_loader,
        save_checkpoint_folder_path,
        load_checkpoint_file_path=None,
        load=False,
        sample_interval=100,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.feat_loss = WassFeatureLoss().to(device)
        self.val_loader = val_loader
        self.adv_loss = nn.BCEWithLogitsLoss().to(device)
        self.save_checkpoint_path = save_checkpoint_folder_path
        self.load_checkpoint_path = load_checkpoint_file_path
        self.beta = 5e-3
        self.top_ssim = 0.0
        self.sample_interval = sample_interval
        if load == True:
            if load_checkpoint_file_path is None:
                raise Exception("need checkpoint file path to load")
            self.load_checkpoint()
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0, 0)
        )

    def save_checkpoint(
        self,
        state={},
        is_best=False,
        checkpoint_file_name="checkpoint",
        best_file_name="best",
    ):
        state["generator_state_dict"] = self.generator.state_dict()
        state["discriminator_state_dict"] = self.discriminator.state_dict()
        path = self.save_checkpoint_path + "/{}.pt".format(checkpoint_file_name)
        torch.save(state, path)
        if is_best:
            best_path = self.save_checkpoint + "/{}.pt".format(best_file_name)
            shutil.copyfile(path, best_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.load_checkpoint_path)
        if "ssim" in checkpoint:
            self.top_ssim = checkpoint["ssim"]
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    def train_model(self, start=0, end=100, b=0, eb=-1, isValidate=True):
        for epoch in range(start, end):
            if epoch == start:
                b = b
            else:
                b = 0
            self.train(epoch=epoch, b=b, eb=eb)
            if isValidate == True:
                self.validate(epoch)

    def validate(self, epoch):
        with torch.no_grad():
            self.generator.eval()
            ss = AverageMeter()
            ps = AverageMeter()
            for i, imgs in enumerate(self.val_loader):
                lr_imgs, hr_imgs = imgs["lr"].to(device), imgs["hr"].to(device)
                predicted = self.generator(lr_imgs)
                ssim_value = ssim(predicted, hr_imgs)
                psnr_value = psnr(predicted, hr_imgs)
                ss.update(ssim_value.detach().item(), lr_imgs.size(0))
                ps.update(psnr_value.detach().item(), lr_imgs.size(0))
                print(
                    "Validating Image ", 1, "psnr :", psnr_value, " ssim :", ssim_value
                )
            print("Validation Completed \n")
            print("PSNR :", ps.avg, "\n")
            print("SSIM :", ss.avg, "\n")
            if ss.avg > self.top_ssim:
                self.top_ssim = ss.avg
                isBest = True
                print("Saving best")
            else:
                print("saving checkpoint")
                isBest = False
            state = {"epoch": str(epoch), "psnr": str(ps.avg), "ssim": str(ss.avg)}
            self.save_checkpoint(state, is_best=isBest)

    def predict_single(self, img):
        def preds(arr, denormalize=False):
            if denormalize is False:
                arr.clamp_(0, 1)
            a = im_convert(arr[0], denormalize=denormalize)
            im = Image.fromarray(np.uint8(a * 255)).convert("RGB")
            return im

        img_splitter = ImageSplitter(seg_size=192, scale_factor=4, boarder_pad_size=3)
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
        out = []
        self.generator.eval()
        for j, i in enumerate(img_patches):
            out.append(self.generator(i.to(device)))
        img_upscale = img_splitter.merge_img_tensor(out)
        result = preds(img_upscale)
        return result

    def train(self, epoch, b=0, eb=-1):

        self.generator.train()
        self.discriminator.train()  # training mode enables batch normalization

        losses_c = AverageMeter()  # content loss
        losses_a = AverageMeter()  # adversarial loss in the generator
        losses_d = AverageMeter()  # adversarial loss in the discriminator

        for i, imgs in enumerate(self.train_loader):
            if i <= b and epoch == eb:
                print("skipping", i)
                continue
            lr_imgs = imgs["lr"].to(device)
            hr_imgs = imgs["hr"].to(device)

            generated = self.generator(lr_imgs)
            content_loss = self.feat_loss(generated, hr_imgs)

            score_real = self.discriminator(hr_imgs)
            score_fake = self.discriminator(generated)
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = self.adv_loss(
                discriminator_rf, torch.zeros_like(discriminator_rf)
            )
            adversarial_loss_fr = self.adv_loss(
                discriminator_fr, torch.ones_like(discriminator_fr)
            )
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
            perceptual_loss = content_loss + self.beta * adversarial_loss
            self.optimizer_G.zero_grad()
            perceptual_loss.backward()
            self.optimizer_G.step()
            losses_c.update(content_loss.detach().item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.detach().item(), lr_imgs.size(0))

            # DISCRIMINATOR UPDATE

            # Discriminate super-resolution (SR) and high-resolution (HR) images
            score_real = self.discriminator(hr_imgs)
            score_fake = self.discriminator(generated.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = self.adv_loss(
                discriminator_rf, torch.ones_like(discriminator_fr)
            )
            adversarial_loss_fr = self.adv_loss(
                discriminator_fr, torch.zeros_like(discriminator_rf)
            )
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            # Back-prop.
            self.optimizer_D.zero_grad()
            adversarial_loss.backward()
            self.optimizer_D.step()
            losses_d.update(adversarial_loss.detach().item(), hr_imgs.size(0))
            if i % self.sample_interval == 0:
                with torch.no_grad():
                    state = {"epoch": str(epoch), "batch": str(i)}
                    self.save_checkpoint(state)
                    self.generator.eval()
                    print(
                        "Epoch:{} [{}/{}] content loss :{} advloss:{} discLoss:{}".format(
                            epoch,
                            i,
                            len(self.train_loader),
                            losses_c.avg,
                            losses_a.avg,
                            losses_d.avg,
                        )
                    )
                    losses_c.reset()
                    losses_a.reset()
                    losses_d.reset()
                    self.generator.train()
            del lr_imgs, hr_imgs, generated, score_real, score_fake


class SimpleTrainer:
    def __init__(
        self,
        generator,
        train_loader,
        val_loader,
        save_checkpoint_folder_path,
        criterion=nn.L1Loss(),
        load_checkpoint_file_path=None,
        load=False,
        sample_interval=100,
    ):
        self.generator = generator
        self.train_loader = train_loader
        self.feat_loss = criterion
        self.val_loader = val_loader
        self.save_checkpoint_path = save_checkpoint_folder_path
        self.load_checkpoint_path = load_checkpoint_file_path
        self.top_psnr = -1.0
        self.sample_interval = sample_interval
        if load == True:
            if load_checkpoint_file_path is None:
                raise Exception("need checkpoint file path to load")
            self.load_checkpoint()
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.9, 0.999)
        )

    def save_checkpoint(
        self,
        state={},
        is_best=False,
        checkpoint_file_name="checkpoint_mse",
        best_file_name="best_mse",
    ):
        state["generator_state_dict"] = self.generator.state_dict()
        path = self.save_checkpoint_path + "/{}.pt".format(checkpoint_file_name)
        torch.save(state, path)
        if is_best:
            best_path = self.save_checkpoint_path + "/{}.pt".format(best_file_name)
            shutil.copyfile(path, best_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.load_checkpoint_path)
        if "ssim" in checkpoint:
            self.top_ssim = checkpoint["ssim"]
        self.generator.load_state_dict(checkpoint["generator_state_dict"])

    def train_model(self, start=0, end=100, b=0, eb=-1, isValidate=False):
        for epoch in range(start, end):
            if epoch == start:
                b = b
            else:
                b = 0
            self.train(epoch=epoch, b=b, eb=eb)
            if isValidate == True:
                self.validate(epoch)

    def validate(self, epoch):
        with torch.no_grad():
            self.generator.eval()
            ss = AverageMeter()
            ps = AverageMeter()
            for i, imgs in enumerate(self.val_loader):
                lr_imgs, hr_imgs = imgs["lr"].to(device), imgs["hr"].to(device)
                predicted = self.generator(lr_imgs)
                psnr_value = psnr(predicted, hr_imgs)
                ps.update(psnr_value.detach().item(), lr_imgs.size(0))
                print("Validating Image ", 1, "psnr :", psnr_value)
            print("Validation Completed \n")
            print("PSNR :", ps.avg, "\n")
            if ps.avg > self.top_psnr:
                self.top_psnr = ps.avg
                isBest = True
                print("Saving best")
            else:
                print("saving checkpoint")
                isBest = False
            state = {"epoch": str(epoch), "psnr": str(ps.avg)}
            self.save_checkpoint(state, is_best=isBest)

    def predict_single(self, img):
        def preds(arr, denormalize=False):
            if denormalize is False:
                arr.clamp_(0, 1)
            a = im_convert(arr[0], denormalize=denormalize)
            im = Image.fromarray(np.uint8(a * 255)).convert("RGB")
            return im

        img_splitter = ImageSplitter(seg_size=192, scale_factor=4, boarder_pad_size=3)
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
        out = []
        self.generator.eval()
        for j, i in enumerate(img_patches):
            out.append(self.generator(i.to(device)))
        img_upscale = img_splitter.merge_img_tensor(out)
        result = preds(img_upscale)
        return result

    def train(self, epoch, b=0, eb=-1):

        self.generator.train()  # training mode enables batch normalization

        losses_c = AverageMeter()  # content loss

        for i, imgs in enumerate(self.train_loader):
            if i <= b and epoch == eb:
                print("skipping", i)
                continue
            lr_imgs = imgs["lr"].to(device)
            hr_imgs = imgs["hr"].to(device)

            generated = self.generator(lr_imgs)
            content_loss = self.feat_loss(generated, hr_imgs)
            self.optimizer_G.zero_grad()
            content_loss.backward()
            self.optimizer_G.step()
            losses_c.update(content_loss.detach().item(), lr_imgs.size(0))
            if i % self.sample_interval == 0:
                with torch.no_grad():
                    state = {"epoch": str(epoch), "batch": str(i)}
                    self.save_checkpoint(state)
                    self.generator.eval()
                    print(
                        "Epoch: {} [{}/{}] content loss :{}".format(
                            epoch,
                            i,
                            len(self.train_loader),
                            losses_c.avg,
                        )
                    )
                    losses_c.reset()
                    self.generator.train()
            del lr_imgs, hr_imgs, generated
