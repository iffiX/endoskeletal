import torch


def pad_and_crop_coordinate3d(p, padding=0.1):
    """
    Apply padding to point coordinates, normalize and crop to [0, 1].
    Args:
        p (tensor): point of shape [N, P, 3], roughly in range [0, 1], may have outliers
        padding (float): conventional padding parameter of ONet for unit cube,
            so [-0.5, 0.5] -> [-0.55, 0.55] with 0.1 padding.
    """

    p_nor = (p - 0.5) / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def coordinate3d_to_index(x, reso):
    """
    Converts normalized coordinates in range [0, 1] to 3D integer indices defined
    by the cube of shape [reso, reso, reso].

    Args:
        x (tensor): normalized coordinates of shape [N, P, 3] in range [0, 1],
            last dim is ordered in XYZ.
        reso (int): defined resolution.

    Returns:
        Index tensor of shape [N, 1, P].

    Note:
        N: batch size.
        P: number of points.
    """
    x = (x * (reso - 1)).long()
    index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def coordinate3d_to_relative_position(x, reso):
    """
    Converts normalized coordinates in range [0, 1] to 3D integer indices defined
    by the cube of shape [reso, reso, reso].

    Args:
        x (tensor): normalized coordinates of shape [N, P, 3] in range [0, 1],
            last dim is ordered in XYZ.
        reso (int): defined resolution.

    Returns:
        Relative position of shape [N, 1, P].

    Note:
        N: batch size.
        P: number of points.
    """
    xx = torch.floor(x * reso) / reso
    relative_position = (x - xx) * reso
    return relative_position
