import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)   # concatenate skip connection
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        # final output layer
        self.out_conv = nn.Conv2d(64, 1, 1)  # final 1x1 conv

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(c0)
        c1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(c1)
        c2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(c2)
        c3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(c3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        u0 = self.upsample0(b)
        d0 = F.relu(self.dec_conv0(torch.cat([u0, c3], dim=1)))
        u1 = self.upsample1(d0)
        d1 = F.relu(self.dec_conv1(torch.cat([u1, c2], dim=1)))
        u2 = self.upsample2(d1)
        d2 = F.relu(self.dec_conv2(torch.cat([u2, c1], dim=1)))
        u3 = self.upsample3(d2)
        d3 = F.relu(self.dec_conv3(torch.cat([u3, c0], dim=1)))

        logits = self.out_conv(d3)  # final output layer
        return logits


# class UNet2, we replace maxpooling by convolutions with stride 2 and upsampling by transposed convolutions with stride 2
class UNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upconv0 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64 + 64, 64, 3, padding=1)   
        # final output layer
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = F.relu(self.pool0(c0))
        c1 = F.relu(self.enc_conv1(e0))
        e1 = F.relu(self.pool1(c1))
        c2 = F.relu(self.enc_conv2(e1))
        e2 = F.relu(self.pool2(c2))
        c3 = F.relu(self.enc_conv3(e2))
        e3 = F.relu(self.pool3(c3))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        u0 = self.upconv0(b)
        d0 = F.relu(self.dec_conv0(torch.cat([u0, c3], dim=1)))
        u1 = self.upconv1(d0)
        d1 = F.relu(self.dec_conv1(torch.cat([u1, c2], dim=1)))
        u2 = self.upconv2(d1)
        d2 = F.relu(self.dec_conv2(torch.cat([u2, c1], dim=1)))
        u3 = self.upconv3(d2)
        d3 = F.relu(self.dec_conv3(torch.cat([u3, c0], dim=1)))

        logits = self.out_conv(d3)  # final output layer
        return logits



class UNet_multi(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)   # concatenate skip connection
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        # final output layer
        self.out_conv = nn.Conv2d(64, 12, 1)  # final 1x1 conv

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(c0)
        c1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(c1)
        c2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(c2)
        c3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(c3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        u0 = self.upsample0(b)
        d0 = F.relu(self.dec_conv0(torch.cat([u0, c3], dim=1)))
        u1 = self.upsample1(d0)
        d1 = F.relu(self.dec_conv1(torch.cat([u1, c2], dim=1)))
        u2 = self.upsample2(d1)
        d2 = F.relu(self.dec_conv2(torch.cat([u2, c1], dim=1)))
        u3 = self.upsample3(d2)
        d3 = F.relu(self.dec_conv3(torch.cat([u3, c0], dim=1)))

        logits = self.out_conv(d3)  # final output layer
        return logits


# class UNet_Paper(nn.Module), following the paper architecture

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class UNet_Paper(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        self.c1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = DoubleConv(512, 1024)

        # DECODER
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c6  = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c7  = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c8  = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c9  = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)  # output logits

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))
        c5 = self.c5(self.p4(c4))

        u4 = self.up4(c5)
        u4 = self.c6(torch.cat([u4, c4], dim=1))
        u3 = self.up3(u4)
        u3 = self.c7(torch.cat([u3, c3], dim=1))
        u2 = self.up2(u3)
        u2 = self.c8(torch.cat([u2, c2], dim=1))
        u1 = self.up1(u2)
        u1 = self.c9(torch.cat([u1, c1], dim=1))
        return self.out(u1)
