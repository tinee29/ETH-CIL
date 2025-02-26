import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from logger import logger
import utils
import globals
from loss import WeightedBCELoss

def upconv_block(in_channels, out_channels, dropout_rate=0.5):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
    )  # (H, W, in_channels) -> (2H, 2W, out_channels)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.conv_1x1_output = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_1x1_output(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads, dropout_rate=0.5):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        
        attention = torch.matmul(query, key)
        attention = F.softmax(attention / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(attention, value.permute(0, 1, 3, 2))
        
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, width, height)
        out = self.gamma * out + x
        out = self.dropout(out)
        return out


def get_pretrained_encoder():
    encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return encoder


resnet50_encoder = get_pretrained_encoder()


class UNetEncoderASPP(nn.Module):
    def __init__(self, encoder=None, out_channels=1, dropout_rate=0.5, num_heads=8):
        super(UNetEncoderASPP, self).__init__()

        self.encoder = resnet50_encoder
        self.encoder_layers = list(self.encoder.children())

        self.enc1 = nn.Sequential(*self.encoder_layers[:3])  # (H, W, 3) -> (H/2, W/2, 64)
        self.enc2 = nn.Sequential(*self.encoder_layers[3:5])  # (H/2, W/2, 64) -> (H/4, W/4, 256)
        self.enc3 = self.encoder_layers[5]  # (H/4, W/4, 256) -> (H/8, W/8, 512)
        self.enc4 = self.encoder_layers[6]  # (H/8, W/8, 512) -> (H/16, W/16, 1024)

        self.bottleneck = self.encoder_layers[7]  # (H/16, W/16, 1024) -> (H/32, W/32, 2048)
        self.aspp = ASPP(2048, 256, dropout_rate=dropout_rate)
        self.attention = MultiHeadSelfAttention(256, num_heads=num_heads, dropout_rate=dropout_rate)

        self.latent_work = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.dec5 = upconv_block(256, 1024, dropout_rate=dropout_rate)
        self.dec4 = upconv_block(1024, 512, dropout_rate=dropout_rate)
        self.dec3 = upconv_block(512, 256, dropout_rate=dropout_rate)
        self.dec2 = upconv_block(256, 64, dropout_rate=dropout_rate)
        self.dec1 = upconv_block(64, 64, dropout_rate=dropout_rate)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        bottleneck = self.bottleneck(enc4)
        bottleneck = self.aspp(bottleneck)
        bottleneck = self.attention(bottleneck)
        bottleneck = self.latent_work(bottleneck)

        dec5 = self.dec5(bottleneck)
        dec4 = self.dec4(dec5 + enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)

        final = self.final_conv(dec1)
        return torch.sigmoid(final)

    def freeze_encoder(self):
        logger.info("Freezing encoder weights enc1, enc2")
        for param in self.enc1.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        logger.info(f"Loaded weights from {path}")