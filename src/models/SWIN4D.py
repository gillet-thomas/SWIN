import torch.nn as nn

from src.models.mlp import clf_mlpv1
from src.models.swin4d_transformer_ver7 import SwinTransformer4D


class SWIN4D(nn.Module):
    def __init__(self, config):
        super(SWIN4D, self).__init__()
        self.model = SwinTransformer4D(
            img_size=config["img_size"],
            in_chans=config["in_chans"],
            embed_dim=config["embed_dim"],
            window_size=config["window_size"],
            first_window_size=config["first_window_size"],
            patch_size=config["patch_size"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            c_multiplier=config["c_multiplier"],
            last_layer_full_MSA=config["last_layer_full_MSA"],
            attn_drop_rate=config["dropout"],
        )
        num_tokens = config["embed_dim"] * (
            config["c_multiplier"] ** (config["n_stages"] - 1)
        )
        self.output_head = clf_mlpv1(
            num_classes=config["num_classes"], num_tokens=num_tokens
        )

    def forward(self, x):
        x = self.model(
            x
        )  # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        x = self.output_head(x)  # ([8, 288, 2, 2, 2, 20]) -> ([8, 1])
        return x

    def get_embeddings(self, x):
        x = self.model(
            x
        )  # input ([8, 1, 112, 112, 112, 20]) -> ([8, 288, 2, 2, 2, 20])
        x = x.view(x.size(0), -1)  # Flattens ([8, 288, 2, 2, 2, 20]) -> ([8, 4608])
        return x
