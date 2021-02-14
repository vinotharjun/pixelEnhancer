from enhancer.imports.torch_imports import *
from torchvision.models import vgg16_bn
from torchvision.models.vgg import vgg19


class vgg16(nn.Module):
    def __init__(self, pre_trained=True, require_grad=False):
        super(vgg16, self).__init__()
        self.vgg_feature = models.vgg16_bn(pretrained=pre_trained).features
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        self.vgg_layer = ["block" + str(i) for i in range(0, len(self.seq_list))]

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):

        block0 = self.seq_list[0](x)
        block1 = self.seq_list[1](block0)
        block2 = self.seq_list[2](block1)
        block3 = self.seq_list[3](block2)
        block4 = self.seq_list[4](block3)
        block5 = self.seq_list[5](block4)
        block6 = self.seq_list[6](block5)
        block7 = self.seq_list[7](block6)
        block8 = self.seq_list[8](block7)
        block9 = self.seq_list[9](block8)
        block10 = self.seq_list[10](block9)
        block11 = self.seq_list[11](block10)

        block12 = self.seq_list[12](block11)
        block13 = self.seq_list[13](block12)
        block14 = self.seq_list[14](block13)
        block15 = self.seq_list[15](block14)
        block16 = self.seq_list[16](block15)
        block17 = self.seq_list[17](block16)
        block18 = self.seq_list[18](block17)
        block19 = self.seq_list[19](block18)
        block20 = self.seq_list[20](block19)
        block21 = self.seq_list[21](block20)
        block22 = self.seq_list[22](block21)

        block23 = self.seq_list[23](block22)
        block24 = self.seq_list[24](block23)
        block25 = self.seq_list[25](block24)
        block26 = self.seq_list[26](block25)
        block27 = self.seq_list[27](block26)
        block28 = self.seq_list[28](block27)
        block29 = self.seq_list[29](block28)
        block30 = self.seq_list[30](block29)
        block31 = self.seq_list[31](block30)
        block32 = self.seq_list[32](block31)
        block33 = self.seq_list[33](block32)

        block34 = self.seq_list[34](block33)
        block35 = self.seq_list[35](block34)
        block36 = self.seq_list[36](block35)
        block37 = self.seq_list[37](block36)
        block38 = self.seq_list[38](block37)
        block39 = self.seq_list[39](block38)
        block40 = self.seq_list[40](block39)
        block41 = self.seq_list[41](block40)
        block42 = self.seq_list[42](block41)
        block43 = self.seq_list[43](block42)

        vgg_output = namedtuple("vgg_output", self.vgg_layer)

        vgg_list = [
            block0,
            block1,
            block2,
            block3,
            block4,
            block5,
            block6,
            block7,
            block8,
            block9,
            block10,
            block11,
            block12,
            block13,
            block14,
            block15,
            block16,
            block17,
            block18,
            block19,
            block20,
            block21,
            block22,
            block23,
            block24,
            block25,
            block26,
            block27,
            block28,
            block29,
            block30,
            block31,
            block32,
            block33,
            block34,
            block35,
            block36,
            block37,
            block38,
            block39,
            block40,
            block42,
            block42,
            block43,
        ]

        out = vgg_output(*vgg_list)
        del vgg_list

        return out


class FeatureLossAlternate(nn.Module):
    def __init__(
        self,
        vgg,
        blocks=["block22", "block32", "block42"],
        weights=[5, 15, 2],
        gram_matrix=True,
    ):
        super(FeatureLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = vgg
        self.blocks = blocks
        self.base_loss = F.l1_loss
        self.wgts = weights
        self.gram_matrix = gram_matrix

    def forward(self, sr, hr):
        res = [self.base_loss(sr, hr)]
        hr_feat = []
        sr_feat = []
        for block in self.blocks:

            hr_feat.append(getattr(self.vgg(hr), str(block)))

        for block in self.blocks:
            sr_feat.append(getattr(self.vgg(sr), str(block)))

        res += [
            self.base_loss(inp, targ) * wgt
            for inp, targ, wgt in zip(sr_feat, hr_feat, self.wgts)
        ]
        if self.gram_matrix == True:
            res += [
                self.base_loss(gram_mat(inp), gram_mat(targ)) * wgt ** 2 * 5e3
                for inp, targ, wgt in zip(sr_feat, hr_feat, self.wgts)
            ]

        del hr_feat, sr_feat

        # hr_feat = getattr(self.vgg(hr), layer)
        # sr_feat = getattr(self.vgg(sr), layer)
        # res += [self.base_loss(hr_feat, sr_feat)]

        # res +=[self.base_loss(gram_mat(hr_feat), gram_mat(sr_feat))]
        return sum(res)


#         # return res
# vgg = vgg16().to(device)
# vgg = vgg.eval()
# criterion = FeatureLoss(vgg)
# optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)


class FeatureLoss(nn.Module):
    def __init__(
        self, m, layer_ids, layer_wgts=[5, 15.2], base_loss=F.l1_loss, gram_matix=True
    ):
        super().__init__()
        self.gram_matix = gram_matix
        self.base_loss = base_loss
        self.m, self.wgts = m, layer_wgts
        self.sfs = [SaveFeatures(m[i]) for i in layer_ids]

    def forward(self, input_im, target, sum_layers=True):

        self.m(target)
        targ_feat = [o.features.data.clone() for o in self.sfs]
        self.m(input_im)
        res = [self.base_loss(input_im, target)]
        res += [
            self.base_loss(inp.features, targ) * wgt
            for inp, targ, wgt in zip(self.sfs, targ_feat, self.wgts)
        ]
        if self.gram_matix:
            res += [
                self.base_loss(gram_matrix(inp.features), gram_matrix(targ))
                * wgt ** 2
                * 5e3
                for inp, targ, wgt in zip(self.sfs, targ_feat, self.wgts)
            ]

        if sum_layers:
            res = sum(res)
        self.close()
        return res

    def close(self):
        for o in self.sfs:
            o.remove()


def load_vgg16(cuda_enable=True, sanity_checking=False):
    def sanity_check(vgg16, blocks):
        print(blocks)
        print([list(vgg16.children())[i] for i in blocks])

    if cuda_enable:
        vgg16 = vgg16_bn(True).features.cuda().eval()
    else:
        vgg16 = vgg16_bn(True).features.cpu.eval()
    for param in vgg16.parameters():
        param.requires_grad = False
    blocks = [
        i - 1
        for i, o in enumerate(list(vgg16.children()))
        if isinstance(o, nn.MaxPool2d)
    ]
    if sanity_checking:
        sanity_check(vgg16, blocks)

    return vgg16, blocks


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(
            self.loss_network(high_resolution).detach(),
            self.loss_network(fake_high_resolution),
        )
        return perception_loss

class CharbonnierLoss(nn.Module):
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss 