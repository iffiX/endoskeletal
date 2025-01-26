import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from model.resnet import ResnetBlockFC
from model.unet3d import UNet3D
from model.common import pad_and_crop_coordinate3d, coordinate3d_to_index


class LocalPoolBaseEncoder(nn.Module):
    """
    Base model for all encoder network that convert points.

    Args:
        f_dim (int): input point feature dimension.
        c_dim (int): latent conditioned code c dimension.
        hidden_dim (int): hidden dimension of the network.
        scatter_type (str): feature aggregation when doing local pooling, "mean"/"max".
        grid_resolution (int): defined resolution for grid feature.
        padding (float): conventional padding parameter of ONet for unit cube,
            so [-0.5, 0.5] -> [-0.55, 0.55] with 0.1 padding.
    """

    def __init__(
        self,
        f_dim,
        c_dim,
        hidden_dim=128,
        scatter_type="max",
        grid_resolution=16,
        padding=0.02,
    ):
        super().__init__()
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.grid_resolution = grid_resolution
        self.padding = padding

        self.resnet_in = ResnetBlockFC(f_dim, hidden_dim)
        self.conv_out = nn.Conv3d(hidden_dim, c_dim, kernel_size=1)
        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def pool_local(self, index, hidden):
        """
        Pools the feature to the latent grid.

        Args:
            index: Index tensor of shape [N, 1, P].
            hidden: Hidden tensor of shape [N, P, hidden_dim].

        Returns:
            Pooled feature tensor of shape [N, hidden_dim, reso, reso, reso].
        """
        # permuted feature shape: [N, hidden_dim, P]
        # pooled shape: [N, hidden_dim, reso ** 3]
        pooled = self.scatter(
            hidden.permute(0, 2, 1), index, dim_size=self.grid_resolution**3
        )
        if self.scatter == scatter_max:
            pooled = pooled[0]
        return pooled.view(
            pooled.shape[0],
            pooled.shape[1],
            self.grid_resolution,
            self.grid_resolution,
            self.grid_resolution,
        )

    def forward(self, normalized_points, feature):
        """
        Args:
            normalized_points: normalized point location tensor of shape [N, P, 3],
                roughly in range [0, 1].
            feature: feature tensor of shape [N, P, f_dim].

        Returns:
            Index tensor of shape [N, 1, P]
            Latent feature grid of shape [N, c_dim, reso, reso, reso].

        Note:
            N: batch size.
            P: number of points.
            reso: grid_resolution.
        """

        # acquire the index for each point
        coord = pad_and_crop_coordinate3d(
            normalized_points.clone(), padding=self.padding
        )
        index = coordinate3d_to_index(coord, self.grid_resolution)

        # h shape: [N, P, hidden_dim]
        h = self.resnet_in(feature)

        h = self.pool_local(index, h)

        feature_grid = self.conv_out(h)

        return index, feature_grid


class RLEncoder(LocalPoolBaseEncoder):
    """
    Encoder network for convert points to latent feature grid and postprocess with U-Net.

    Args:
        f_dim (int): input point feature dimension.
        c_dim (int): latent conditioned code c dimension.
        hidden_dim (int): hidden dimension of the network.
        scatter_type (str): feature aggregation when doing local pooling, "mean"/"max".
        grid_resolution (int): defined resolution for grid feature.
        padding (float): conventional padding parameter of ONet for unit cube,
            so [-0.5, 0.5] -> [-0.55, 0.55] with 0.1 padding.
        unet3d_kwargs (dict): UNet3D parameters
    """

    def __init__(
        self,
        f_dim,
        c_dim,
        hidden_dim=128,
        scatter_type="max",
        grid_resolution=16,
        padding=0.02,
        unet3d_kwargs=None,
    ):
        super().__init__(
            f_dim=f_dim,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
            scatter_type=scatter_type,
            grid_resolution=grid_resolution,
            padding=padding,
        )
        self.unet3d = UNet3D(
            c_dim,
            c_dim,
            is_segmentation=False,
            final_sigmoid=False,
            **(unet3d_kwargs or {})
        )

    def forward(self, normalized_points, feature):
        """
        Args:
            normalized_points: normalized point location tensor of shape [N, P, 3],
                roughly in range [0, 1].
            feature: feature tensor of shape [N, P, f_dim].

        Returns:
            Latent code tensor of shape [N, c_dim, reso, reso, reso].

        Note:
            N: batch size.
            P: number of points.
            reso: grid_resolution.
        """
        _, feature_grid = super().forward(normalized_points, feature)
        return self.unet3d(feature_grid)
