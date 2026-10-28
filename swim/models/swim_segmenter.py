import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swim_encoder import SwinEncoder
# from swim_encoder import SwinEncoder

# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        # Após concatenar:
        #
        # upsampled feature + skip feature

        total_channels = (
            in_channels
            + skip_channels
        )

        self.conv1 = nn.Conv2d(
            total_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x,
        skip,
    ):

        # ----------------------------------------------------
        # Upsampling para resolução do skip
        # ----------------------------------------------------

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Concatenação
        # ----------------------------------------------------

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        # ----------------------------------------------------
        # Convoluções
        # ----------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


# ============================================================
# SWIN SEGMENTER
# ============================================================

class SwinSegmenter(nn.Module):

    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = SwinEncoder(
            pretrained=pretrained,
        )

        # Swin-T:
        #
        # F1 = 96
        # F2 = 192
        # F3 = 384
        # F4 = 768

        c1, c2, c3, c4 = (
            self.encoder.channels
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder3 = DecoderBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
        )

        self.decoder2 = DecoderBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
        )

        self.decoder1 = DecoderBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
        )

        # ====================================================
        # FINAL DECODER
        # ====================================================

        self.final_conv1 = nn.Conv2d(
            c1,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.final_bn1 = nn.BatchNorm2d(
            64
        )

        self.final_conv2 = nn.Conv2d(
            64,
            32,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.final_bn2 = nn.BatchNorm2d(
            32
        )

        # ====================================================
        # SEGMENTATION HEAD
        # ====================================================

        self.segmentation_head = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    # ========================================================
    # ENCODER FREEZE / UNFREEZE
    # ========================================================

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def encoder_is_frozen(self):
        return all(
            not parameter.requires_grad
            for parameter in self.encoder.parameters()
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        input_size = x.shape[-2:]

        # ====================================================
        # SWIN ENCODER
        # ====================================================

        features = self.encoder(x)

        f1, f2, f3, f4 = features

        # ----------------------------------------------------
        # timm Swin:
        #
        # [B, H, W, C]
        #
        # Conv2d precisa:
        #
        # [B, C, H, W]
        # ----------------------------------------------------

        f1 = f1.permute(
            0, 3, 1, 2
        ).contiguous()

        f2 = f2.permute(
            0, 3, 1, 2
        ).contiguous()

        f3 = f3.permute(
            0, 3, 1, 2
        ).contiguous()

        f4 = f4.permute(
            0, 3, 1, 2
        ).contiguous()

        # ====================================================
        # DECODER
        # ====================================================

        x = self.decoder3(
            f4,
            f3,
        )

        x = self.decoder2(
            x,
            f2,
        )

        x = self.decoder1(
            x,
            f1,
        )

        # ====================================================
        # 128 × 128 -> 512 × 512
        # ====================================================

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # ====================================================
        # REFINAMENTO
        # ====================================================

        x = self.final_conv1(x)
        x = self.final_bn1(x)
        x = self.relu(x)

        x = self.final_conv2(x)
        x = self.final_bn2(x)
        x = self.relu(x)

        # ====================================================
        # LOGITS
        # ====================================================

        logits = self.segmentation_head(
            x
        )

        return logits


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda:1"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SWIN-T SEGMENTER")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    model = SwinSegmenter(
        pretrained=True,
    ).to(device)

    # --------------------------------------------------------
    # Teste com batch pequeno
    # --------------------------------------------------------

    x = torch.randn(
        1,
        3,
        512,
        512,
    ).to(device)

    print(
        f"\nInput:  {x.shape}"
    )

    model.eval()

    with torch.no_grad():

        output = model(x)

    print(
        f"Output: {output.shape}"
    )

    print(
        f"Min logit: "
        f"{output.min().item():.6f}"
    )

    print(
        f"Max logit: "
        f"{output.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Sigmoid apenas para inspeção
    # --------------------------------------------------------

    probabilities = torch.sigmoid(
        output
    )

    print(
        f"\nProbability min: "
        f"{probabilities.min().item():.6f}"
    )

    print(
        f"Probability max: "
        f"{probabilities.max().item():.6f}"
    )

    print("\nTudo OK.")