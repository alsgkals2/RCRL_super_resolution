import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from . import block as B


class DiscriminatorVGG(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(DiscriminatorVGG, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, stride=1, padding=1),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64 x 64 x 64
            nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*2, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 32 x 32
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*4, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 16 x 16
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 8 x 8
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 4 x 4
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.d*8) * self.feature_map_size * self.feature_map_size, 100),
            # nn.Linear(2048, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        # print((self.d*8) * self.feature_map_size * self.feature_map_size)
        # print(out.shape)
        out = self.classifier(out)

        return out


# VGG style Discriminator with input size 64*64
class Discriminator_VGG_64(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_64, self).__init__()
        self.fm_size = 256//base_nf
#         print(self.fm_size)
        # features
        # hxw, c
        # 64, 3
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*8 * self.fm_size * self.fm_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
#         print(x.shape)
#         print(base_nf*8 * self.fm_size * self.fm_size)
        x = self.classifier(x)
        return x

    
class DiscriminatorVGG256(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(DiscriminatorVGG256, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, stride=1, padding=1),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64 x 64 x 64
            nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*2, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 32 x 32
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*4, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 16 x 16
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 8 x 8
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 4 x 4
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32768, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        feat = self.features(x)
        out = torch.flatten(feat, 1)
        out = self.classifier(out)

        return out


class UNetDiscriminator(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat*2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat*2, num_feat*4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat*4, num_feat*8, kernel_size=4, stride=2, padding=1, bias=False))

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat*8, num_feat*4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat*4, num_feat*2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat*2, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        self.conv9 = nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
