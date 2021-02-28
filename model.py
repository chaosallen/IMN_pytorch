# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import unetConv2, unetUp,mnetDown, unetUp_origin
from layers import ResEncoder, Decoder, AffinityAttention
from layers import deform_up, deform_down, deform_inconv
from torchvision import models
import numpy as np

class UNet(nn.Module):

    def __init__(self, in_channels, n_classes, channels=128, is_deconv=False, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes

        # downsampling
        self.conv1 = unetConv2(in_channels, self.channels, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(self.channels, self.channels, self.is_batchnorm)
        # upsampling
        self.up_concat2 = unetUp(self.channels*2, self.channels, 2, self.is_deconv)
        self.up_concat1 = unetUp(self.channels*2, self.channels, 2, self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)


        center = self.center(maxpool2)


        up2 = self.up_concat2(center, conv2)
        up1 = self.up_concat1(up2, conv1)

        output = self.outconv1(up1)

        return output
class CNN(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64, is_deconv=False, is_batchnorm=True):
        super(CNN, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes

        # downsampling
        self.conv1 = unetConv2(self.in_channels, self.channels, self.is_batchnorm)
        self.conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.conv3 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.outconv1 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        output = self.outconv1(conv3)

        return output
class IMN(nn.Module):

    def __init__(self, in_channels, n_classes, channels=128, is_maxpool=True, is_batchnorm=True):
        super(IMN, self).__init__()
        self.is_maxpool = is_maxpool
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes

        # MNET
        self.M_conv1 = unetConv2(in_channels, self.channels, self.is_batchnorm)
        self.M_up1 = nn.ConvTranspose2d(self.channels,self.channels, kernel_size=2, stride=2, padding=0)
        self.M_conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.M_up2 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, padding=0)
        self.M_center = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.M_down2 = mnetDown(self.channels*2, self.channels, 2,self.is_maxpool)
        self.M_down1 = mnetDown(self.channels*2, self.channels, 2, self.is_maxpool)

        self.outconv1 = nn.Conv2d(self.channels, self.n_classes, kernel_size=3, padding=1)

    def forward(self, inputs):
        # MNET
        M_conv1 = self.M_conv1(inputs)
        M_up1 = self.M_up1(M_conv1)
        M_conv2 = self.M_conv2(M_up1)
        M_up2 = self.M_up2(M_conv2)
        M_center = self.M_center(M_up2)
        M_down2 = self.M_down2(M_center, M_conv2)
        M_down1 = self.M_down1(M_down2, M_conv1)
        output = self.outconv1(M_down1)
        return output
class MUNet(nn.Module):

    def __init__(self, in_channels, n_classes, channels=128, is_maxpool=True,is_deconv=False, is_batchnorm=True):
        super(MUNet, self).__init__()
        self.is_maxpool = is_maxpool
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_deconv=is_deconv
        self.channels = channels
        self.n_classes=n_classes

        # MNET
        self.M_conv1 = unetConv2(in_channels, self.channels, self.is_batchnorm)
        self.M_up1 = nn.ConvTranspose2d(self.channels,self.channels, kernel_size=2, stride=2, padding=0)
        self.M_conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.M_up2 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, padding=0)
        self.M_center = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.M_down2 = mnetDown(self.channels*2, self.channels, 2,self.is_maxpool)
        self.M_down1 = mnetDown(self.channels*2, self.channels, 2, self.is_maxpool)

        # UNET
        self.U_conv1 = unetConv2(in_channels, self.channels, self.is_batchnorm)
        self.U_down1 = nn.MaxPool2d(kernel_size=2)
        self.U_conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.U_down2 = nn.MaxPool2d(kernel_size=2)
        self.U_center = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.U_up2 = unetUp(self.channels*2, self.channels, 2, self.is_deconv)
        self.U_up1 = unetUp(self.channels*2, self.channels, 2, self.is_deconv)

        #output
        self.outconv1 = nn.Conv2d(self.channels*2, self.channels, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)


    def forward(self, inputs):
        # MNET
        M_conv1 = self.M_conv1(inputs)
        M_up1 = self.M_up1(M_conv1)
        M_conv2 = self.M_conv2(M_up1)
        M_up2 = self.M_up2(M_conv2)
        M_center = self.M_center(M_up2)
        M_down2 = self.M_down2(M_center, M_conv2)
        M_down1 = self.M_down1(M_down2, M_conv1)
        #UNET
        U_conv1 = self.U_conv1(inputs)
        U_down1 = self.U_down1(U_conv1)
        U_conv2 = self.U_conv2(U_down1)
        U_down2 = self.U_down2(U_conv2)
        U_center = self.U_center(U_down2)
        U_up2 = self.U_up2(U_center, U_conv2)
        U_up1 = self.U_up1(U_up2, U_conv1)


        #output
        MU_concat = torch.cat([M_down1, U_up1], 1)
        outconv1 = self.outconv1(MU_concat)
        output=self.outconv2(outconv1)
        return output
class CSNet(nn.Module):
    def __init__(self,  in_channels, n_classes, channels=128):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet, self).__init__()
        self.enc_input = ResEncoder(in_channels, channels)
        self.encoder1 = ResEncoder(channels, channels)
        self.encoder2 = ResEncoder(channels, channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.affinity_attention = AffinityAttention(channels)
        self.attention_fuse = nn.Conv2d(channels* 2, channels, kernel_size=1)
        self.decoder2 = Decoder(channels*2, channels)
        self.decoder1 = Decoder(channels*2, channels)
        self.deconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.final = nn.Conv2d(channels, n_classes, kernel_size=1)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        input_feature = self.encoder2(down2)


        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)

        # attention_fuse = self.attention_fuse(torch.cat((input_feature, attention), dim=1))
        attention_fuse = input_feature + attention

        # Do decoder operations here

        up2 = self.deconv2(attention_fuse)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class DUNet(nn.Module):
    # downsize_nb_filters_factor=4 compare to DUNetV1V2_MM
    def __init__(self,in_channels, n_classes, channels=128):
        super(DUNet, self).__init__()
        self.channels = channels
        self.n_classes=n_classes
        self.is_batchnorm = False
        self.in_channels = in_channels
        # self.inc = deform_inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.inc = unetConv2(in_channels, self.channels, self.is_batchnorm)
        self.down1 = deform_down(self.channels, self.channels)
        self.down2 = deform_down(self.channels, self.channels)
        self.up3 = deform_up(self.channels*2, self.channels)
        self.up4 = deform_up(self.channels*2, self.channels)
        self.outc = nn.Conv2d(self.channels+1, n_classes, 1)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = torch.cat([inp, x], dim=1)
        x = self.outc(x)
        return torch.sigmoid(x)