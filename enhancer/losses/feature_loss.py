from enhancer.imports.torch_imports import *


class VGG(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cuda"),
    ):
        super(VGG, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True).to(device)
        else:
            model = torchvision.models.vgg19(pretrained=True).to(device)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(
            *list(model.features.children())[: (feature_layer + 1)]
        )
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, generated):
        b, c, h, w = generated.size()
        h_tv = torch.pow(
            (generated[:, :, 1:, :] - generated[:, :, : (h - 1), :]), 2
        ).sum()
        w_tv = torch.pow(
            (generated[:, :, :, 1:] - generated[:, :, :, : (w - 1)]), 2
        ).sum()
        return (h_tv + w_tv) / (b * c * h * w)


class FeatureLoss(torch.nn.Module):
    def __init__(self):

        super(FeatureLoss, self).__init__()
        self.feature_extractor = VGG()
        self.criterion = nn.L1Loss()
#         self.tvloss = TVLoss()

    def forward(self, generated, groundtruth):
        l1_loss = self.criterion(generated, groundtruth)
#         tv_loss = self.tvloss(generated)
        generated_vgg = self.feature_extractor(generated)
        groundtruth_vgg = self.feature_extractor(groundtruth)
        groundtruth_vgg_no_grad = groundtruth_vgg.detach()
        feat_loss = self.criterion(generated_vgg, groundtruth_vgg_no_grad)
#         total_loss = 2e-8 * tv_loss + feat_loss + 1e-2 * l1_loss
        total_loss = 1e-2*feat_loss + l1_loss
        return total_loss

    
# class FeatureLoss(torch.nn.Module):
#     def __init__(self):

#         super(FeatureLoss, self).__init__()
#         self.feature_extractor = VGG()
#         self.criterion = nn.L1Loss()

#     def forward(self, generated, groundtruth):
#         l1_loss = self.criterion(generated, groundtruth)
#         generated_vgg = self.feature_extractor(generated)
#         groundtruth_vgg_no_grad =  self.feature_extractor(groundtruth).detach()
#         feat_loss =[self.criterion(generated_vgg, groundtruth_vgg_no_grad)]        
#         feat_loss +=[1e-2*l1_loss]
#         styles = self._get_style_vals(groundtruth_vgg_no_grad)
#         if styles[0] is not None:
#             feat_loss +=[ self._single_wass_loss(generated_vgg, styles)]
#         return sum(feat_loss)

#     def _calc_2_moments(self, tensor):
#         chans = tensor.shape[1]
#         tensor = tensor.view(1, chans, -1)
#         n = tensor.shape[2]
#         mu = tensor.mean(2)
#         tensor = (tensor - mu[:, :, None]).squeeze(0)
#         # Prevents nasty bug that happens very occassionally- divide by zero.  Why such things happen?
#         if n == 0:
#             return None, None
#         cov = torch.mm(tensor, tensor.t()) / float(n)
#         return mu, cov

#     def _get_style_vals(self, tensor):
#         mean, cov = self._calc_2_moments(tensor)
#         if mean is None:
#             return None, None, None
#         eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
#         eigroot_mat = torch.diag(torch.sqrt(eigvals.clamp(min=0)))
#         root_cov = torch.mm(torch.mm(eigvects, eigroot_mat), eigvects.t())
#         tr_cov = eigvals.clamp(min=0).sum()
#         return mean, tr_cov, root_cov

#     def _calc_l2wass_dist(
#         self, mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth
#     ):
#         tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
#         mean_diff_squared = (mean_stl - mean_synth).pow(2).sum()
#         cov_prod = torch.mm(torch.mm(root_cov_stl, cov_synth), root_cov_stl)
#         var_overlap = torch.sqrt(
#             torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0) + 1e-8
#         ).sum()
#         dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
#         return dist

#     def _single_wass_loss(self, pred, targ):
#         mean_test, tr_cov_test, root_cov_test = targ
#         mean_synth, cov_synth = self._calc_2_moments(pred)
#         loss = self._calc_l2wass_dist(
#             mean_test, tr_cov_test, root_cov_test, mean_synth, cov_synth
#         )
#         return loss
