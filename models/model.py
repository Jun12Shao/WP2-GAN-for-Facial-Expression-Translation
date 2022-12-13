import torch
import torch.nn as nn
import numpy as np
import functools
from pytorch_wavelets import DWTForward

class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == 'batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer


class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))     #29*128*128---> 64*128*128
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))      ## 64*128*128-> 128*64*64  -> 256*32*32
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))         # 6 layers of Residual Block

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)       ## used to generate I(yr)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)  ## used to generate an attention??

    def forward(self, x, c):            ## x is the input image and c is the condition
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        features = self.main(x)
        return self.img_reg(features), self.attetion_reg(features)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)         ## refer to the definition of residual network

class Discriminator1(NetworkBase):
    def __init__(self, image_size=64, conv_dim=64, c_dim=5, repeat_num=5):
        super(Discriminator1, self).__init__()
        self._name = 'D1'

        self.dwt= DWTForward(J=1, mode='periodization', wave='db3')  # Accepts all wave types available to PyWavelets

        layers = []
        layers.append(nn.Conv2d(12, conv_dim, kernel_size=4, stride=2, padding=1))  ## 3*128*128-->64*64*64
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):  ## here is 4 times
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))  ## 2048*2*2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)  ## 1*2*2
        self.conv2 = nn.Conv2d(curr_dim, 17, kernel_size=k_size, bias=False)  ## 17*1*1
        # self.conv3 = nn.Conv2d(curr_dim, 7, kernel_size=k_size, bias=False)             ## 4*1*1
        # self.conv4 = nn.Conv2d(curr_dim, 2, kernel_size=k_size, bias=False)             ## 2*1*1


    def forward(self, x):
            x1,xh1=self.dwt(x)                       ## xh1[0]:  bt*3*3*64*64
            x1=x1.unsqueeze(2)                         ##bt*3*1*64*64
            x1=torch.cat([x1, xh1[0]], 2)           ##bt*3*4*64*64
            x1=x1.view(-1,12,64,64)

            h = self.main(x1)
            out_real = self.conv1(h)
            out_aux = self.conv2(h)
            # out_age = self.conv3(h)
            # out_gender = self.conv4(h)

            # return out_real.squeeze(),  out_age.squeeze(), out_gender.squeeze()  # generate tensor with size 1* n
            return out_real.squeeze(), out_aux.squeeze()


class Discriminator2(NetworkBase):
    def __init__(self, image_size=32, conv_dim=64, c_dim=5, repeat_num=4):
        super(Discriminator2, self).__init__()
        self._name = 'D2'

        self.dwt= DWTForward(J=1, mode='periodization', wave='db3')  # Accepts all wave types available to PyWavelets

        layers = []
        layers.append(nn.Conv2d(48, conv_dim, kernel_size=4, stride=2, padding=1))  ## 3*128*128-->64*64*64
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):  ## here is 4 times
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))  ## 2048*2*2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)  ## 1*2*2
        self.conv2 = nn.Conv2d(curr_dim, 17, kernel_size=k_size, bias=False)  ## 17*1*1
        # self.conv3 = nn.Conv2d(curr_dim, 7, kernel_size=k_size, bias=False)             ## 4*1*1
        # self.conv4 = nn.Conv2d(curr_dim, 2, kernel_size=k_size, bias=False)             ## 2*1*1


    def forward(self, x):
            x1,xh1=self.dwt(x)                       ## xh1[0]:  bt*3*3*64*64
            x1=x1.unsqueeze(2)                         ##bt*3*1*64*64
            x1=torch.cat([x1, xh1[0]], 2)           ##bt*3*4*64*64
            x2=None
            for i in range(4):
                xm,xhm=self.dwt(x1[:,:,i,:,:])
                xm=xm.unsqueeze(2)  ##bt*3*1*64*64
                if x2 is None:
                    x2 = torch.cat([xm, xhm[0]], 2)
                else:
                    x2=torch.cat([x2, xm, xhm[0]], 2)
            x2=x2.view(-1,16*3,32,32)

            h=self.main(x2)

            out_real = self.conv1(h)
            out_aux = self.conv2(h)
            # out_age = self.conv3(h)
            # out_gender = self.conv4(h)

            # return out_real.squeeze(),  out_age.squeeze(), out_gender.squeeze()  # generate tensor with size 1* n
            return out_real.squeeze(), out_aux.squeeze()


class Discriminator0(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator0, self).__init__()
        self._name = 'D0'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))       ## 3*128*128-->64*64*64
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):  ## here is 5 times
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))          ## 2048*2*2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False) ## 1*2*2
        self.conv2 = nn.Conv2d(curr_dim, 17, kernel_size=k_size, bias=False)             ## 17*1*1
        # self.conv3 = nn.Conv2d(curr_dim, 7, kernel_size=k_size, bias=False)             ## 4*1*1
        # self.conv4 = nn.Conv2d(curr_dim, 2, kernel_size=k_size, bias=False)             ## 2*1*1


    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        # out_age=self.conv3(h)
        # out_gender=self.conv4(h)
        # return out_real.squeeze(), out_age.squeeze(), out_gender.squeeze()       # generate tensor with size 1* n
        return out_real.squeeze(), out_aux.squeeze()


