import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List, Tuple, Dict


# -----------------------------
# Utils
# -----------------------------
def make_decoder_channels(base: int = 64, n_blocks: int = 4):
    """
    Gera canais do decoder do mais profundo para o mais raso.

    Ex:
        base=64 -> [512, 256, 128, 64]
        base=32 -> [256, 128, 64, 32]
    """
    return [base * (2 ** i) for i in reversed(range(n_blocks))]


# -----------------------------
# CBAM
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# -----------------------------
# ASPP
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512, rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([b1, b2, b3, b4, gp], dim=1)
        return self.project(x)

# -----------------------------
# Encoder
# -----------------------------
class ResNeXt101Encoder(nn.Module):
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()

        weights = None
        if pretrained:
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT

        backbone = models.resnext101_32x8d(weights=weights)

        # adapta conv1 se a entrada não for RGB
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if pretrained:
                if in_channels == 1:
                    # média dos pesos RGB -> 1 canal
                    backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1   # 256
        self.layer2 = backbone.layer2   # 512
        self.layer3 = backbone.layer3   # 1024
        self.layer4 = backbone.layer4   # 2048

    def forward(self, x):
        f0 = self.relu(self.bn1(self.conv1(x)))   # ~ 1/2 resolução, 64 canais
        f1 = self.maxpool(f0)                     # ~ 1/4 resolução, 64 canais
        f2 = self.layer1(f1)                      # 256 canais
        f3 = self.layer2(f2)                      # 512 canais
        f4 = self.layer3(f3)                      # 1024 canais
        f5 = self.layer4(f4)                      # 2048 canais

        return [f0, f1, f2, f3, f4, f5]


# -----------------------------
# Decoder block
# -----------------------------
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(
                out_channels,
                reduction=cbam_reduction,
                spatial_kernel=cbam_spatial_kernel
            )

    def forward(self, x, skip=None):
        x = self.upconv(x)

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_cbam:
            x = self.cbam(x)

        return x


# -----------------------------
# Full model
# -----------------------------
class ResNeXt101SegmentationModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 1,
        pretrained: bool = True,
        in_channels: int = 1,
        base_decoder_channels: int = 64,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
        use_aspp: bool = True,
        aspp_out_channels: int = 512,
        aspp_rates: Tuple[int, int, int] = (6, 12, 18),
    ):
        super().__init__()

        self.encoder = ResNeXt101Encoder(
            pretrained=pretrained,
            in_channels=in_channels
        )

        self.use_aspp = use_aspp

        if self.use_aspp:
            self.aspp = ASPP(
                in_channels=2048,
                out_channels=aspp_out_channels,
                rates=aspp_rates
            )
            bottleneck_channels = aspp_out_channels
        else:
            bottleneck_channels = 2048

        # base=64 -> [512, 256, 128, 64]
        d4, d3, d2, d1 = make_decoder_channels(base=base_decoder_channels, n_blocks=4)

        self.decoder4 = DecoderBlock(
            in_channels=bottleneck_channels,
            skip_channels=1024,
            out_channels=d4,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder3 = DecoderBlock(
            in_channels=d4,
            skip_channels=512,
            out_channels=d3,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder2 = DecoderBlock(
            in_channels=d3,
            skip_channels=256,
            out_channels=d2,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder1 = DecoderBlock(
            in_channels=d2,
            skip_channels=64,
            out_channels=d1,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )

        self.final_up = nn.ConvTranspose2d(d1, d1, kernel_size=2, stride=2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(d1 + 64, d1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d1, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        f0, f1, f2, f3, f4, f5 = self.encoder(x)

        if self.use_aspp:
            f5 = self.aspp(f5)

        x = self.decoder4(f5, skip=f4)
        x = self.decoder3(x,  skip=f3)
        x = self.decoder2(x,  skip=f2)
        x = self.decoder1(x,  skip=f0)

        x = self.final_up(x)

        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(
                x,
                size=f0.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, f0], dim=1)
        x = self.final_conv(x)

        # garante mesma resolução da entrada/máscara
        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        return x