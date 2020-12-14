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
        adversarial_loss_weight=5e-3,
        opt_beta1=0.0,
        opt_beta2=0.99,
        wd=1e-3,
        lr_G=1e-4,
        lr_D=1e-4,
        top_ssim=-1,
        save_checkpoint_folder_path="./",
        save_checkpoint_file_name="checkpoint",
        save_best_file_name="best",
        load_checkpoint_file_path_generator=None,
        load_checkpoint_file_path_critic = None,
        sample_interval=100,
        feature_criterion = WassFeatureLoss().to(device)
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.feat_loss = feature_criterion
        self.val_loader = val_loader
        self.adv_loss = nn.BCEWithLogitsLoss().to(device)
        self.save_checkpoint_path = save_checkpoint_folder_path
        self.load_checkpoint_path_generator = load_checkpoint_file_path_generator
        self.load_checkpoint_path_critic = load_checkpoint_file_path_critic
        self.beta = adversarial_loss_weight
        self.top_ssim = top_ssim
        self.save_checkpoint_file_name = save_checkpoint_file_name
        self.save_best_file_name = save_best_file_name
        if not os.path.exists(self.save_checkpoint_path):
            os.mkdir(self.save_checkpoint_path)
        self.sample_interval = sample_interval
        load_generator = False
        load_critic = False
        if self.load_checkpoint_path_generator is not None:
            load_generator = True
        if self.load_checkpoint_path_critic is not None:
            load_critic = True
        self.load_checkpoint(load_generator,load_critic)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_G,
            betas=(opt_beta1, opt_beta2),
            weight_decay=wd,
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_D,
            betas=(opt_beta1, opt_beta2),
            weight_decay=wd,
        )

    def save_checkpoint(
        self,
        state={},
        is_best=False,
        checkpoint_file_name="checkpoint",
        best_file_name="best",
        writer=None
    ):
        state["generator_state_dict"] = self.generator.state_dict()
        state["discriminator_state_dict"] = self.discriminator.state_dict()
        path = self.save_checkpoint_path + "/{}".format(checkpoint_file_name)
        torch.save(state, path)
        if writer is not None:
            writer.write(f"checkpoint saved at {path}")
        if is_best:
            best_path = self.save_checkpoint_path + "/{}".format(best_file_name)
            shutil.copyfile(path, best_path)

            

    def load_checkpoint(self,load_generator=False,load_critic=False):
        if load_generator==True:
            print("loading checkpoint from ", self.load_checkpoint_path_generator)
            checkpoint = torch.load(self.load_checkpoint_path_generator)
            if "ssim" in checkpoint:
                self.top_ssim = checkpoint["ssim"]
            if "epoch" in checkpoint:
                self.epoch_start = checkpoint["epoch"]
            else:
                self.epoch_start = 0
            if "batch" in checkpoint:
                self.batch_start = checkpoint["batch"]
            else:
                self.batch_start = 0
            if "generator_state_dict" in checkpoint:
                self.generator.load_state_dict(checkpoint["generator_state_dict"])
                print("generator loaded...")
            if load_critic == True and "discriminator_state_dict" in checkpoint:
                self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
                print("critic loaded...")
            print(
                f"Info :check point details are \n epoch : {self.epoch_start} and batch : {self.batch_start} "
            )

    def train_model(self, start=0, end=100, b=0, eb=-1, isValidate=False):
        mb = master_bar(range(start, end))
        for epoch in mb:
            mb.child_comment = "epoch {}".format(epoch)
            if epoch == start:
                b = b
            else:
                b = 0
            self.train(epoch=epoch, b=b, eb=eb, parent=mb)
            if isValidate == True:
                self.validate(epoch, parent=mb)
            else:
                mb.write("No validation enabled, so saving epoch checkpoint only")
                self.save_checkpoint(
                    state = {"epoch":str(epoch),"batch":str(-1)},
                    is_best=False,
                    checkpoint_file_name=self.save_checkpoint_file_name,
                    best_file_name=self.save_best_file_name,
                    writer=mb
                )

    def validate(self, epoch, parent):
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
                ps.update(psnr_value, lr_imgs.size(0))
                parent.write(
                    "Validating Image "
                    + str(i)
                    + " psnr :"
                    + str(psnr_value)
                    + " ssim :"
                    + str(ssim_value),
                )
            parent.write("Validation Completed \n")
            parent.write("PSNR :" + str(ps.avg) + "\n")
            parent.write("SSIM :" + str(ss.avg) + "\n")
            if ss.avg > self.top_ssim:
                self.top_ssim = ss.avg
                isBest = True
                parent.write("Saving best")
            else:
                parent.write("saving checkpoint")
                isBest = False
            state = {
                "epoch": str(epoch),
                "psnr": str(ps.avg),
                "ssim": str(ss.avg),
                "batch": str(-1),
            }
            self.save_checkpoint(
                state,
                is_best=isBest,
                checkpoint_file_name=self.save_checkpoint_file_name,
                best_file_name=self.save_best_file_name,
            )

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

    def train(self, epoch, parent, b=0, eb=-1):

        self.generator.train()
        self.discriminator.train()  # training mode enables batch normalization

        losses_c = AverageMeter()  # content loss
        losses_a = AverageMeter()  # adversarial loss in the generator
        losses_d = AverageMeter()  # adversarial loss in the discriminator

        global_losses_c = AverageMeter()  # content loss
        global_losses_a = AverageMeter()  # adversarial loss in the generator
        global_losses_d = AverageMeter()  # adversarial loss in the discriminator
        pb = progress_bar(self.train_loader, parent=parent)
        for i, imgs in enumerate(pb):
            if i <= b and epoch == eb:
                parent.write("skipping " + str(i))
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

            global_losses_c.update(content_loss.detach().item(), lr_imgs.size(0))
            global_losses_a.update(adversarial_loss.detach().item(), lr_imgs.size(0))

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
            global_losses_d.update(adversarial_loss.detach().item(), hr_imgs.size(0))
            if self.sample_interval != None and i % self.sample_interval == 0 and i>=self.sample_interval:
                with torch.no_grad():
                    state = {"epoch": str(epoch), "batch": str(i)}
                    self.save_checkpoint(
                        state,
                        is_best=False,
                        checkpoint_file_name=self.save_checkpoint_file_name,
                        best_file_name=self.save_best_file_name,
                    )
                    self.generator.eval()
                    parent.write(
                        "Epoch:{} [{}/{}]  content loss :{}   advloss:{}  discLoss:{}".format(
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
        parent.write(
            "Epoch:{} ends,  Avg content loss :{}  Avg advloss:{}  Avg discLoss:{}".format(
                epoch,
                global_losses_c.avg,
                global_losses_a.avg,
                global_losses_d.avg,
            )
        )
        
class SimpleTrainer:
    def __init__(
        self,
        generator,
        train_loader,
        val_loader,
        opt_beta1=0.0,
        opt_beta2=0.99,
        wd=1e-3,
        lr=1e-4,
        top_psnr=-1,
        save_checkpoint_folder_path="./",
        save_checkpoint_file_name="checkpoint",
        save_best_file_name="best",
        criterion=nn.L1Loss(),
        load_checkpoint_file_path=None,
        sample_interval=100,
        lr_step_decay=None
    ):
        self.generator = generator.to(device)
        self.train_loader = train_loader
        self.feat_loss = criterion.to(device)
        self.val_loader = val_loader
        self.save_checkpoint_path = save_checkpoint_folder_path
        self.load_checkpoint_path = load_checkpoint_file_path
        self.top_psnr = top_psnr
        self.save_checkpoint_file_name = save_checkpoint_file_name
        self.save_best_file_name = save_best_file_name
        self.sample_interval = sample_interval
        if self.sample_interval is not None:
            if self.sample_interval <= 0:
                self.sample_interval = None
                
        self.lr_step_decay = lr_step_decay
            
        self.lr = lr
        load = False
        if load_checkpoint_file_path is not None:
            load = True
        if load == True:
            self.load_checkpoint()
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(opt_beta1, opt_beta2),
            weight_decay=wd,
        )

    def save_checkpoint(
        self,
        state={},
        is_best=False,
        checkpoint_file_name="checkpoint_mse",
        best_file_name="best_mse",
        writer=None
    ):
        state["generator_state_dict"] = self.generator.state_dict()
        path = self.save_checkpoint_path + "/{}".format(checkpoint_file_name)
        torch.save(state, path)
        if writer is not None:
            writer.write(f"checkpoint saved at {path}")
        if is_best:
            best_path = self.save_checkpoint_path + "/{}".format(best_file_name)
            shutil.copyfile(path, best_path)

    def adjust_learning_rate(self,epoch,step_size=200,gamma=0.5):
        factor = epoch // step_size
        lr = self.lr * (gamma ** factor)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

    def load_checkpoint(self):
        print("loading checkpoint from ", self.load_checkpoint_path)
        checkpoint = torch.load(self.load_checkpoint_path)
        if "psnr" in checkpoint:
            self.top_psnr = checkpoint["psnr"]
        if "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
        if "epoch" in checkpoint:
            self.epoch_start = checkpoint["epoch"]
        else:
            self.epoch_start = 0
        if "batch" in checkpoint:
            self.batch_start = checkpoint["batch"]
        else:
            self.batch_start = -1
        print(
            f"Info :check point details are \n epoch : {self.epoch_start} \n batch : {self.batch_start} "
        )

    def train_model(self, start=0, end=100, b=0, eb=-1, isValidate=False):
        mb = master_bar(range(start, end))
        for epoch in mb:
            if epoch == start:
                b = b
            else:
                b = 0
            self.train(epoch=epoch, b=b, eb=eb, parent=mb)
            if self.lr_step_decay is not None:
                self.adjust_learning_rate(epoch,step_size=self.lr_step_decay)
                mb.write('adjusting learning rate : epoch ='+str(epoch)+ ' lr = '+str(self.optimizer_G.param_groups[0]['lr']))
            if isValidate == True:
                self.validate(epoch, parent=mb)
            else:
                mb.write("No validation enabled, so saving epoch checkpoint only")
                self.save_checkpoint(
                    state = {"epoch":str(epoch),"batch":str(-1)},
                    is_best=False,
                    checkpoint_file_name=self.save_checkpoint_file_name,
                    best_file_name=self.save_best_file_name,
                    writer=mb
                )
                

    def validate(self, epoch, parent):
        with torch.no_grad():
            self.generator.eval()
            ps = AverageMeter()
            pb = progress_bar(self.val_loader, parent=parent)
            for i, imgs in enumerate(pb):
                lr_imgs, hr_imgs = imgs["lr"].to(device), imgs["hr"].to(device)
                predicted = self.generator(lr_imgs)
                psnr_value = psnr(predicted, hr_imgs)
                ps.update(psnr_value, lr_imgs.size(0))
                parent.write(
                    "Validating Image " + str(i) + " psnr : " + str(psnr_value)
                )
            parent.write("Validation Completed \n")
            parent.write("PSNR : " + str(ps.avg) + "\n")
            if ps.avg > self.top_psnr:
                self.top_psnr = ps.avg
                isBest = True
                parent.write("Saving best")
            else:
                parent.write("saving checkpoint")
                isBest = False
            state = {"epoch": str(epoch), "psnr": str(ps.avg), "batch": str(-1)}
            self.save_checkpoint(
                state,
                is_best=isBest,
                checkpoint_file_name=self.save_checkpoint_file_name,
                best_file_name=self.save_best_file_name,
            )

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
        for i in img_patches:
            out.append(self.generator(i.to(device)))
        img_upscale = img_splitter.merge_img_tensor(out)
        result = preds(img_upscale)
        return result

    def train(self, epoch, parent, b=0, eb=-1):

        self.generator.train()  # training mode enables batch normalization
        losses_c = AverageMeter()  # content loss
        global_loss_c = AverageMeter()
        pb = progress_bar(self.train_loader, parent=parent)
        for i, imgs in enumerate(pb):
            if i <= b and epoch == eb:
                parent.write("skipping" + str(i))
                continue
            lr_imgs = imgs["lr"].to(device)
            hr_imgs = imgs["hr"].to(device)

            generated = self.generator(lr_imgs)
            content_loss = self.feat_loss(generated, hr_imgs)
            self.optimizer_G.zero_grad()
            content_loss.backward()
            self.optimizer_G.step()
            losses_c.update(content_loss.detach().item(), lr_imgs.size(0))
            global_loss_c.update(content_loss.detach().item(), lr_imgs.size(0))

            if self.sample_interval != None and i % self.sample_interval == 0 and i>=self.sample_interval:
                with torch.no_grad():
                    state = {"epoch": str(epoch), "batch": str(i)}
                    self.save_checkpoint(
                        state,
                        is_best=False,
                        checkpoint_file_name=self.save_checkpoint_file_name,
                        best_file_name=self.save_best_file_name,
                        writer = parent
                    )
                    self.generator.eval()
                    parent.write(
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
        parent.write(
            "\nEpoch: {} average content loss {}".format(epoch, global_loss_c.avg)
        )