import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    """
    Swin Transformer Tiny utilizado como encoder
    para segmentação.

    Retorna features hierárquicas de múltiplos estágios
    para posteriormente conectarmos ao decoder.
    """

    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        # ====================================================
        # SWIN-T BACKBONE
        # ====================================================
        # opcoes de ecnonder
        # "swin_tiny_patch4_window7_224"
        # "swin_small_patch4_window7_224"
        # "swin_base_patch4_window7_224"
        # "swin_large_patch4_window7_224.ms_in22k"

        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=512,
        )

        # ====================================================
        # INFORMAÇÕES DOS ESTÁGIOS
        # ====================================================

        self.channels = (
            self.encoder.feature_info.channels()
        )

        self.reduction = (
            self.encoder.feature_info.reduction()
        )

    def forward(self, x):

        features = self.encoder(x)

        return features


# ============================================================
# TESTE ISOLADO
# ============================================================

if __name__ == "__main__":

    model = SwinEncoder(
        pretrained=True,
    )

    print("=" * 70)
    print("SWIN-T ENCODER")
    print("=" * 70)

    print(
        f"Channels: {model.channels}"
    )

    print(
        f"Reduction: {model.reduction}"
    )

    # Batch artificial
    x = torch.randn(
        2,
        3,
        512,
        512,
    )

    with torch.no_grad():

        features = model(x)

    print("\nFeature maps:")

    for i, feature in enumerate(features):

        print(
            f"Stage {i + 1}: "
            f"{feature.shape}"
        )