import torch.nn as nn
import numpy as np
from model.voxception import VoxceptionResnet


class RobotConvDecoder(nn.Module):
    """
    Decoder for generating voxels in a cubic grid from embedding

    Args:
        f_dim (int): dimension of output voxel features.
        e_dim (int): dimension of embedding.
        grid_size (int): Output voxel cubic grid size.
        vrn_dim (int): Voxception decoder base layer dimension.
        conv_decoder_kwargs (dict): Convolution decoder layer parameters.
        n_conv_t_decoder_layers (int): number of transposed convolution decoder layers.
        vrn_depth (int): Depth of the Voxception in each decoder sub layer.
    """

    def __init__(
        self,
        f_dim,
        e_dim,
        grid_size,
        vrn_dim=128,
        conv_decoder_kwargs=None,
        n_conv_t_decoder_layers=3,
        vrn_depth=3,
    ):
        super().__init__()
        if n_conv_t_decoder_layers < 1:
            raise ValueError("The minimum number of decoder layers is 1")
        max_levels = int(np.log2(grid_size))
        if 2**max_levels != grid_size:
            raise ValueError("grid_size must be a power of 2")
        if max_levels < n_conv_t_decoder_layers:
            raise ValueError("too many decoder layers")

        self.f_dim = f_dim
        self.e_dim = e_dim
        self.grid_size = grid_size
        self.input_resolution = grid_size // (2**n_conv_t_decoder_layers)
        input_out_size = (
            vrn_dim * (2**n_conv_t_decoder_layers) * self.input_resolution**3
        )
        self.fc_input = nn.Linear(e_dim, input_out_size)
        self.bn_input = nn.BatchNorm1d(input_out_size)

        upsampling = []
        for i in range(n_conv_t_decoder_layers):
            in_dim = 2 ** (n_conv_t_decoder_layers - i) * vrn_dim
            out_dim = 2 ** (n_conv_t_decoder_layers - i - 1) * vrn_dim
            upsampling.append(
                VoxceptionResnet(in_dim, out_dim, **(conv_decoder_kwargs or {}))
            )
            for _ in range(vrn_depth - 1):
                upsampling.append(
                    VoxceptionResnet(out_dim, out_dim, **(conv_decoder_kwargs or {}))
                )
            upsampling.append(
                nn.ConvTranspose3d(
                    out_dim,
                    out_dim if i < n_conv_t_decoder_layers - 1 else f_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )

        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, embedding):
        feature_grid = self.bn_input(self.fc_input(embedding))
        feature_grid = self.upsampling(
            feature_grid.view(
                feature_grid.shape[0],
                -1,
                self.input_resolution,
                self.input_resolution,
                self.input_resolution,
            )
        ).view(
            feature_grid.shape[0],
            self.f_dim,
            self.grid_size,
            self.grid_size,
            self.grid_size,
        )
        return feature_grid
