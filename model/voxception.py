import torch as t
import torch.nn as nn


class VoxceptionBasicConv3d(nn.Module):
    def __init__(
        self,
        size_in: int,
        size_out: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation=nn.ReLU,
        batch_norm=True,
    ):
        super(VoxceptionBasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            size_in, size_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm3d(size_out, eps=0.001) if batch_norm else lambda x: x
        self.activation = (
            activation(inplace=True) if activation is not None else lambda x: x
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Voxception(nn.Module):
    def __init__(
        self, size_in: int, size_out: int, activation=nn.ReLU, batch_norm=True
    ):
        super().__init__()
        self.conv1 = VoxceptionBasicConv3d(
            size_in,
            size_in,
            kernel_size=1,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.conv2 = VoxceptionBasicConv3d(
            size_in,
            size_in,
            kernel_size=3,
            padding=1,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.conv_out = nn.Conv3d(2 * size_in, size_out, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input of shape [N, size_in, D, H, W]

        Returns:
            Output of shape [N, size_out, D, H, W]
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.conv_out(t.cat((x1, x2), dim=1))
        return out


class VoxceptionResnet(nn.Module):
    def __init__(
        self, size_in: int, size_out: int, activation=nn.ReLU, batch_norm=True
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            VoxceptionBasicConv3d(
                size_in,
                size_in // 4,
                kernel_size=3,
                padding=1,
                activation=activation,
                batch_norm=batch_norm,
            ),
            VoxceptionBasicConv3d(
                size_in // 4,
                size_in // 2,
                kernel_size=3,
                padding=1,
                activation=activation,
                batch_norm=batch_norm,
            ),
        )
        self.conv2 = nn.Sequential(
            VoxceptionBasicConv3d(
                size_in,
                size_in // 4,
                kernel_size=1,
                activation=activation,
                batch_norm=batch_norm,
            ),
            VoxceptionBasicConv3d(
                size_in // 4,
                size_in // 4,
                kernel_size=3,
                padding=1,
                activation=activation,
                batch_norm=batch_norm,
            ),
            VoxceptionBasicConv3d(
                size_in // 4,
                size_in - size_in // 2,
                kernel_size=1,
                activation=activation,
                batch_norm=batch_norm,
            ),
        )
        self.conv_out = nn.Conv3d(size_in, size_out, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input of shape [N, size_in, D, H, W]

        Returns:
            Output of shape [N, size_out, D, H, W]
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x + t.cat((x1, x2), dim=1)
        out = self.conv_out(x)
        return out


class VoxceptionDownSample(nn.Module):
    def __init__(
        self, size_in: int, size_out: int, activation=nn.ReLU, batch_norm=True
    ):
        super().__init__()
        self.conv1 = VoxceptionBasicConv3d(
            size_in,
            size_in // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.conv2 = VoxceptionBasicConv3d(
            size_in,
            size_in // 2,
            kernel_size=1,
            stride=2,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.conv3 = nn.Sequential(
            *(
                [
                    nn.Conv3d(
                        size_in,
                        size_in // 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.AvgPool3d(3, stride=2, padding=1),
                ]
                + [nn.BatchNorm3d(size_in // 2, eps=0.001)]
                if batch_norm
                else [] + [activation()] if activation is not None else []
            )
        )
        self.conv4 = nn.Sequential(
            *(
                [
                    nn.Conv3d(
                        size_in,
                        size_in // 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.MaxPool3d(3, stride=2, padding=1),
                ]
                + [nn.BatchNorm3d(size_in // 2, eps=0.001)]
                if batch_norm
                else [] + [activation()] if activation is not None else []
            )
        )
        self.conv_out = nn.Conv3d(4 * (size_in // 2), size_out, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input of shape [N, size_in, D, H, W]

        Returns:
            Output of shape [N, size_out, D // 2, H // 2, W // 2]
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = t.cat((x1, x2, x3, x4), dim=1)
        out = self.conv_out(x)
        return out
