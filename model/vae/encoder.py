import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.voxception import (
    VoxceptionBasicConv3d,
    VoxceptionResnet,
    VoxceptionDownSample,
)


class RobotConvEncoder(nn.Module):
    """
    Encoder network for converting voxels in a cubic grid to an embedding tensor.

    Args:
        f_dim (int): input voxel feature dimension.
        e_dim (int): output embedding dimension.
        grid_size (int): Input voxel cubic grid size.
        vrn_dim (int): Voxception encoder base layer dimension.
        conv_encoder_kwargs (dict): Convolution encoder layer parameters.
        n_conv_encoder_layers (int): number of convolution encoder layers.
        vrn_depth (int): Depth of the Voxception in each encoder sub layer.
    """

    def __init__(
        self,
        f_dim,
        e_dim,
        grid_size,
        vrn_dim=64,
        conv_encoder_kwargs=None,
        n_conv_encoder_layers=3,
        vrn_depth=3,
    ):
        super().__init__()
        if n_conv_encoder_layers < 1:
            raise ValueError("The minimum number of encoder layers is 1")
        max_levels = int(np.log2(grid_size))
        if 2**max_levels != grid_size:
            raise ValueError("grid_size must be a power of 2")
        if max_levels < n_conv_encoder_layers:
            raise ValueError("too many encoder layers")

        self.f_dim = f_dim
        self.e_dim = e_dim
        self.grid_size = grid_size
        output_resolution = grid_size // (2**n_conv_encoder_layers)
        output_dim = vrn_dim * (2**n_conv_encoder_layers)
        encoder = [VoxceptionResnet(f_dim, vrn_dim, **(conv_encoder_kwargs or {}))]

        for i in range(n_conv_encoder_layers):
            in_dim = vrn_dim * (2**i)
            out_dim = vrn_dim * (2 ** (i + 1))
            for _ in range(vrn_depth):
                encoder.append(
                    VoxceptionResnet(in_dim, in_dim, **(conv_encoder_kwargs or {}))
                )
            encoder.append(
                VoxceptionDownSample(in_dim, out_dim, **(conv_encoder_kwargs or {}))
            )
        encoder.append(
            VoxceptionBasicConv3d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        encoder.append(nn.MaxPool3d(output_resolution))
        self.encoder = nn.Sequential(*encoder)
        self.fc_out = nn.Linear(output_dim, e_dim * 2)

        self.bn_out = nn.BatchNorm1d(e_dim * 2)
        self.mu_out = nn.Linear(e_dim * 2, e_dim)
        self.log_var_out = nn.Linear(e_dim * 2, e_dim)

    def forward(self, feature_grid):
        """
        Args:
            feature_grid: feature tensor of shape [N, f_dim, grid_size, grid_size, grid_size].

        Returns:
            Embedding tensor of shape [N, e_dim].

        Note:
            N: batch size.
            P: number of points.
        """
        enc = self.encoder(feature_grid)
        out = self.fc_out(enc.view(enc.shape[0], -1))
        out = F.gelu(self.bn_out(out))
        mu = self.mu_out(out)
        log_var = self.log_var_out(out)
        return mu, log_var
