import torch as t
import torch.nn as nn
from model.resnet import ResnetBlockFC
from model.common import pad_and_crop_coordinate3d, coordinate3d_to_index


class RLDecoder(nn.Module):
    """
    Decoder for generating point features from latent code

    Args:
        f_dim (int): dimension of output point features.
        c_dim (int): dimension of latent conditioned code c.
        hidden_dim (int): hidden size of Decoder network.
        grid_resolution (int): resolution of the internal latent conditioned code c.
        sample_mode (str): sampling feature strategy, "bilinear" / "nearest".
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55].
    """

    def __init__(
        self,
        f_dim,
        c_dim,
        hidden_dim=128,
        grid_resolution=16,
        sample_mode="bilinear",
        padding=0.02,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.sample_mode = sample_mode
        self.padding = padding
        self.grid_resolution = grid_resolution
        self.conv_in = nn.Conv3d(c_dim, hidden_dim, kernel_size=1)
        self.resnet_out = ResnetBlockFC(hidden_dim, f_dim)

    def forward(self, normalized_points, condition_code):
        """
        Args:
            normalized_points: normalized point location tensor of shape [N, P, 3],
                roughly in range [0, 1].
            condition_code: [N, c_dim, D, H, W] Tensor.

        Returns:
            point feature tensor of shape [N, P, f_dim].

        Note:
            P: number of sample points.
            D: depth (X size).
            H: height (Y size).
            W: width (Z size).
        """
        coord = pad_and_crop_coordinate3d(
            normalized_points.clone(), padding=self.padding
        )
        index = coordinate3d_to_index(coord, self.grid_resolution)
        condition_code = self.conv_in(condition_code)
        condition_code = condition_code.flatten(start_dim=2)

        c = t.gather(
            condition_code,
            index=index.expand(index.shape[0], condition_code.shape[1], index.shape[2]),
            dim=-1,
        ).permute(0, 2, 1)
        out = self.resnet_out(c)
        return out
