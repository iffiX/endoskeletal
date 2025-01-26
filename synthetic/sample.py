import torch as t
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple


class BaseSample:
    def __init__(self, resolution: int):
        self.resolution = resolution

    def get(self) -> List[t.Tensor]:
        return []


class StarNode:
    def __init__(
        self,
        id: int,
        max_children_num: int,
        attach_offset_length: float,
        rigid_radius: float,
        soft_radius: float,
        rand_blend_weight: float,
        rand_unit_offsets: Tuple[np.ndarray, np.ndarray],
    ):
        self.id = id
        self.center = np.zeros(3)
        self.max_children_num = max_children_num
        self.attach_offset_length = attach_offset_length
        self.rigid_radius = rigid_radius
        self.soft_radius = soft_radius
        self.rand_blend_weight = rand_blend_weight
        self.rand_unit_offsets = rand_unit_offsets

        self.attach_offset_length_degrade_ratio = 1

        self.children = []  # type: List[StarNode]
        self.children_attach_offset = []  # type: List[np.ndarray]
        self.parent = None  # type: Union[None, StarNode]
        # Shape [2, 3] for each part, start (0) and finish (1)
        self.parts = None  # type: Union[None, Tuple[np.ndarray, np.ndarray]]
        # Shape [2] for each part, rigid (0) and soft (1)
        self.parts_radius = None  # type: Union[None, Tuple[np.ndarray, np.ndarray]]
        self.parts_ids = None  # type: Union[None, Tuple[int, int]]
        self.ax = None  # type: Union[None, np.ndarray]
        self.level = np.inf

    def attach_child(
        self,
        child,
        child_attach_offset_length_degrade_ratio_multiplier: float = 0.7,
    ):
        if len(self.children) > 0:
            opposite_mean = -np.mean(np.stack(self.children_attach_offset), axis=0)
            opposite_mean_norm = np.linalg.norm(opposite_mean)

            if opposite_mean_norm > 0:
                opposite_mean_unit = opposite_mean / opposite_mean_norm
                offset = child.rand_unit_offsets[
                    0
                ] * child.rand_blend_weight + opposite_mean_unit * (
                    1 - child.rand_blend_weight
                )
                child_offset_unit = offset / np.linalg.norm(offset)
            else:
                child_offset_unit = child.rand_unit_offsets[0]
        else:
            child_offset_unit = child.rand_unit_offsets[0]

        child_offset_unit2 = child.rand_unit_offsets[
            1
        ] * child.rand_blend_weight + child_offset_unit * (1 - child.rand_blend_weight)
        child_offset_unit2 = child_offset_unit2 / np.linalg.norm(child_offset_unit2)

        self.children.append(child)
        self.children_attach_offset.append(child_offset_unit)

        child.attach_offset_length_degrade_ratio = (
            self.attach_offset_length_degrade_ratio
            * child_attach_offset_length_degrade_ratio_multiplier
        )
        child.parent = self
        child.level = self.level + 1
        offset_from_center_to_child_joint = (
            child_offset_unit
            * self.attach_offset_length
            * self.attach_offset_length_degrade_ratio
        )
        offset_from_child_joint_to_child_center = (
            child_offset_unit2
            * child.attach_offset_length
            * child.attach_offset_length_degrade_ratio
        )
        child.center = (
            self.center
            + offset_from_center_to_child_joint
            + offset_from_child_joint_to_child_center
        )
        child.parts = (
            np.array([self.center, self.center + offset_from_center_to_child_joint]),
            np.array(
                [
                    self.center + offset_from_center_to_child_joint,
                    self.center
                    + offset_from_center_to_child_joint
                    + offset_from_child_joint_to_child_center,
                ]
            ),
        )
        child.parts_radius = (
            np.array([self.rigid_radius, self.soft_radius]),
            np.array([child.rigid_radius, child.soft_radius]),
        )
        child.parts_ids = (self.id, child.id)
        child.ax = np.cross(
            offset_from_center_to_child_joint, offset_from_child_joint_to_child_center
        )
        child.ax = child.ax / np.linalg.norm(child.ax)

    def is_attachable(self):
        return self.max_children_num > len(self.children)


class StarRobot(BaseSample):
    def __init__(
        self,
        min_num_nodes: int,
        max_num_nodes: int,
        batch_size: int,
        device: int,
        seed: int = None,
        resolution: int = 64,
    ):
        """
        Return a batch of synthetic robot segmentation samples.
        Note:
            Returned data: [batch_size,
        """
        super().__init__(resolution)
        if seed is not None:
            self.random = np.random.RandomState(seed)
        else:
            self.random = np.random
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.batch_size = batch_size
        self.device = device
        self.max_children = 2
        self.resolution = resolution

        self.is_not_empty = t.zeros(
            [batch_size, 1] + [resolution] * 3, dtype=t.float32, device=device
        )
        self.is_rigid = t.zeros(
            [batch_size, 1] + [resolution] * 3, dtype=t.float32, device=device
        )
        self.segment_ids = t.zeros(
            [batch_size, 1] + [resolution] * 3, dtype=t.long, device=device
        )
        # Generated after build
        self.segments = None
        self.robot_topologies = (
            self.generate_robot_topology()
        )  # type: List[List[StarNode]]
        self.build()
        self.postprocess()

    def generate_robot_topology(self):
        robot_topologies = []
        for robot in range(self.batch_size):
            num_nodes = self.random.randint(self.min_num_nodes, self.max_num_nodes + 1)
            start_center = np.zeros(3)
            robot_params = self.random.rand(num_nodes, 11)
            robot_topology = []
            for node in range(num_nodes):
                attach_offset_length = 0.1 + 0.2 * robot_params[node, 0]
                rigid_radius = 0.02 + 0.03 * robot_params[node, 1]
                soft_radius = 0.02 + 0.05 * robot_params[node, 2]
                max_children_num = 2
                if robot_params[node, 3] > 0.66:
                    max_children_num += 2
                elif 0.33 < robot_params[node, 3] < 0.66:
                    max_children_num += 1
                blend_weight = 0.5 * robot_params[node, 4]
                child_offset = np.array(robot_params[node, 5:8]) * 2 - 1
                child_offset /= np.linalg.norm(child_offset)
                parent_offset = np.array(robot_params[node, 8:11]) * 2 - 1
                parent_offset /= np.linalg.norm(parent_offset)
                s_node = StarNode(
                    node + 1,
                    max_children_num,
                    attach_offset_length,
                    rigid_radius,
                    soft_radius,
                    blend_weight,
                    (child_offset, parent_offset),
                )
                robot_topology.append(s_node)

            robot_topology[0].level = 0
            robot_topology[0].center = start_center

            # attach to parents with lower level value first
            for s_node in robot_topology[1:]:
                best_parent = None
                for parent in robot_topology:
                    if (
                        parent is not s_node
                        and parent.is_attachable()
                        and (best_parent is None or parent.level < best_parent.level)
                    ):
                        best_parent = parent
                best_parent.attach_child(
                    s_node, child_attach_offset_length_degrade_ratio_multiplier=0.9
                )
            robot_topologies.append(robot_topology)
        return robot_topologies

    def build(self):
        # 2 positions for every segment, 1 position for root node
        positions = np.zeros((self.batch_size, self.max_num_nodes * 3, 3))
        for robot in range(self.batch_size):
            for node in range(len(self.robot_topologies[robot])):
                # print(self.robot_topologies[robot][node].parts)
                positions[robot, node] = self.robot_topologies[robot][node].center

        # shape: [N, 3]
        lower_bound = np.min(positions, axis=1) - np.ones(3) * 0.05
        upper_bound = np.max(positions, axis=1)

        # shape: [N]
        dx = (np.max(upper_bound - lower_bound, axis=1) + 0.1) / self.resolution

        # shape: [N, 3]
        center_offset = (
            np.array([[self.resolution] * 3])
            - (upper_bound - lower_bound) / dx[:, None]
        ) / 2

        lower_bound = t.from_numpy(lower_bound).to(device=self.device)
        dx = t.from_numpy(dx).to(device=self.device)
        center_offset = t.from_numpy(center_offset).to(device=self.device)

        # shape: [N, resolution**3, 3]
        coords = (
            t.tensor(
                list(np.ndindex(self.resolution, self.resolution, self.resolution)),
                dtype=t.long,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(self.batch_size, self.resolution**3, 3)
        )
        for node in range(1, self.max_num_nodes):
            # shape: [2 * N, 3]
            start = t.tensor(
                np.array(
                    [
                        (
                            self.robot_topologies[robot][node].parts[0][0]
                            if node < len(self.robot_topologies[robot])
                            else np.zeros((3,))
                        )
                        for robot in range(self.batch_size)
                    ]
                    + [
                        (
                            self.robot_topologies[robot][node].parts[1][0]
                            if node < len(self.robot_topologies[robot])
                            else np.zeros((3,))
                        )
                        for robot in range(self.batch_size)
                    ]
                ),
                dtype=t.float32,
                device=self.device,
            )
            # shape: [2 * N, 3]
            end = t.tensor(
                np.array(
                    [
                        (
                            self.robot_topologies[robot][node].parts[0][1]
                            if node < len(self.robot_topologies[robot])
                            else np.zeros((3,))
                        )
                        for robot in range(self.batch_size)
                    ]
                    + [
                        (
                            self.robot_topologies[robot][node].parts[1][1]
                            if node < len(self.robot_topologies[robot])
                            else np.zeros((3,))
                        )
                        for robot in range(self.batch_size)
                    ]
                ),
                dtype=t.float32,
                device=self.device,
            )
            # shape: [2 * N]
            soft_radius = t.tensor(
                [
                    (
                        self.robot_topologies[robot][node].parts_radius[0][1]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ]
                + [
                    (
                        self.robot_topologies[robot][node].parts_radius[1][1]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ],
                dtype=t.float32,
                device=self.device,
            )
            # shape: [2 * N]
            rigid_radius = t.tensor(
                [
                    (
                        self.robot_topologies[robot][node].parts_radius[0][0]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ]
                + [
                    (
                        self.robot_topologies[robot][node].parts_radius[1][0]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ],
                dtype=t.float32,
                device=self.device,
            )
            # shape: [2 * N]
            # For root node, segment id is 1, max is num_nodes
            segment = t.tensor(
                [
                    (
                        self.robot_topologies[robot][node].parts_ids[0]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ]
                + [
                    (
                        self.robot_topologies[robot][node].parts_ids[1]
                        if node < len(self.robot_topologies[robot])
                        else 0
                    )
                    for robot in range(self.batch_size)
                ],
                dtype=t.float32,
                device=self.device,
            )

            self.add_component_batch(
                self.is_not_empty,
                self.is_rigid,
                self.segment_ids,
                coords,
                segment[: self.batch_size],
                start[: self.batch_size],
                end[: self.batch_size],
                soft_radius[: self.batch_size],
                rigid_radius[: self.batch_size],
                lower_bound,
                dx,
                center_offset,
            )
            self.add_component_batch(
                self.is_not_empty,
                self.is_rigid,
                self.segment_ids,
                coords,
                segment[self.batch_size :],
                start[self.batch_size :],
                end[self.batch_size :],
                soft_radius[self.batch_size :],
                rigid_radius[self.batch_size :],
                lower_bound,
                dx,
                center_offset,
            )

    def add_component_batch(
        self,
        is_not_empty: t.Tensor,
        is_rigid: t.Tensor,
        segment_ids: t.Tensor,
        coords,
        idx,
        start,
        end,
        soft_radius,
        rigid_radius,
        lower_bound,
        dx,
        center_offset,
    ):
        start = self.position_to_coords(start, lower_bound, dx, center_offset)
        end = self.position_to_coords(end, lower_bound, dx, center_offset)
        soft_radius = soft_radius / dx
        rigid_radius = rigid_radius / dx
        length = t.norm(end - start, dim=1)
        # shape: [N, 3]
        u = end - start

        # shape: [N, 3, 3]
        proj_mat = (
            u.unsqueeze(1)
            * u.unsqueeze(-1)
            / t.sum(u * u, dim=1).view(self.batch_size, 1, 1)
        )
        # shape: [N, P, 3]
        proj_ps = t.sum(
            proj_mat.view(self.batch_size, 1, 3, 3)
            * (
                (coords - start.view(self.batch_size, 1, 3)).view(
                    self.batch_size, -1, 1, 3
                )
            ),
            dim=-1,
        ) + start.view(self.batch_size, 1, 3)
        # shape: [N, P]
        proj_dist = t.norm(proj_ps - coords, dim=2)

        # sum of distance from projected point to both ends
        # if projected point is on the cylinder component axis, the summed distance
        # should be equal to the length of the cylinder component
        # shape: [N, P]
        end_dist = t.norm(proj_ps - start.view(self.batch_size, 1, 3), dim=2) + t.norm(
            proj_ps - end.view(self.batch_size, 1, 3), dim=2
        )

        # shape: [N, P]
        m_in_seg = t.isclose(end_dist, length.view(self.batch_size, 1), rtol=1e-2)
        m_is_not_empty = t.logical_and(
            proj_dist <= (soft_radius + rigid_radius).view(self.batch_size, 1), m_in_seg
        )
        m_is_rigid = t.logical_and(
            proj_dist <= rigid_radius.view(self.batch_size, 1), m_in_seg
        )

        dist_start = t.norm(start.view(self.batch_size, 1, 3) - coords, dim=2)
        dist_end = t.norm(end.view(self.batch_size, 1, 3) - coords, dim=2)
        m_is_near_start = dist_start <= (soft_radius + rigid_radius).view(
            self.batch_size, 1
        )
        m_is_near_end = dist_end <= (soft_radius + rigid_radius).view(
            self.batch_size, 1
        )

        self.select_fill(
            is_not_empty,
            m_is_not_empty,
            1,
        )
        self.select_fill(is_not_empty, m_is_near_start, 1)
        self.select_fill(is_not_empty, m_is_near_end, 1)
        self.select_fill(
            is_rigid,
            m_is_rigid,
            1,
        )
        self.select_fill(
            segment_ids,
            m_is_rigid,
            idx,
        )

    @staticmethod
    def select_fill(target_tensor, mask, fill):
        """
        Args:
            target_tensor: shape [N, C, resolution (X), resolution (Y), resolution (Z)]
            mask: shape [N, resolution ** 3 (P)], flattened in XYZ order
            fill: shape [N, C] or [N] or constant
        """
        if t.is_tensor(fill):
            fill = fill.view(fill.shape[0], -1)
        target_tensor.copy_(
            t.where(
                mask.view(
                    target_tensor.shape[0],
                    1,
                    target_tensor.shape[2],
                    target_tensor.shape[3],
                    target_tensor.shape[4],
                ).expand(target_tensor.shape),
                fill[:, :, None, None, None] if t.is_tensor(fill) else fill,
                target_tensor,
            )
        )

    @staticmethod
    def select_find(target_tensor, mask):
        """
        Args:
            target_tensor: shape [N, C, resolution (X), resolution (Y), resolution (Z)]
            mask: shape [N, resolution ** 3 (P)], flattened in XYZ order
        """
        return t.sum(
            t.where(
                mask.view(
                    target_tensor.shape[0],
                    1,
                    target_tensor.shape[2],
                    target_tensor.shape[3],
                    target_tensor.shape[4],
                ).expand(target_tensor.shape),
                target_tensor,
                0,
            ).flatten(start_dim=1)
            != 0,
            dim=1,
        )

    @staticmethod
    def position_to_coords(position, lower_bound, dx, center_offset):
        return ((position - lower_bound) / dx.unsqueeze(1)) + center_offset

    def postprocess(self):
        self.segments = (
            F.one_hot(self.segment_ids, self.max_num_nodes + 1)
            .squeeze(1)[:, :, :, :, 1:]
            .permute(0, 4, 1, 2, 3)
            .to(device=self.device)
        )

    def get(self, adjust=True, *_):
        samples = t.cat(
            [
                (~self.is_not_empty.bool()).to(t.float32),
                t.logical_and(self.is_not_empty.bool(), ~self.is_rigid.bool()).to(
                    t.float32
                ),
                self.segments,
            ],
            dim=1,
        )
        if adjust:
            grid = (
                t.stack(
                    t.meshgrid(
                        t.arange(self.resolution),
                        t.arange(self.resolution),
                        t.arange(self.resolution),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .float()
                .to(self.device)
            )
            center = t.tensor(
                [
                    (self.resolution - 1) / 2,
                    (self.resolution - 1) / 2,
                    (self.resolution - 1) / 2,
                ],
                device=self.device,
            )
            shifted_grids = t.zeros(
                (self.batch_size, self.resolution, self.resolution, self.resolution, 3),
                device=self.device,
            )
            for i in range(self.batch_size):
                non_empty = t.nonzero(self.is_not_empty[i, 0])
                bbox_min = non_empty.min(dim=0).values
                bbox_max = non_empty.max(dim=0).values
                centroid = (bbox_min + bbox_max) / 2.0
                shift_vector = t.floor(center - centroid)
                shifted_grids[i] = grid - shift_vector
            shifted_grids = 2.0 * shifted_grids / (self.resolution - 1) - 1.0
            shifted_samples = F.grid_sample(
                samples,
                shifted_grids,
                mode="nearest",
                padding_mode="border",
                align_corners=True,
            )
            samples = shifted_samples
        return samples


class StarRobotReweighted(StarRobot):
    def generate_robot_topology(self):
        robot_topologies = []
        for robot in range(self.batch_size):
            node_range = list(range(self.min_num_nodes, self.max_num_nodes + 1))
            weights = np.array([2**node for node in node_range])
            probabilities = weights / weights.sum()
            num_nodes = np.random.choice(node_range, p=probabilities)
            start_center = np.zeros(3)
            robot_params = self.random.rand(num_nodes, 11)
            robot_topology = []
            for node in range(num_nodes):
                attach_offset_length = 0.1 + 0.2 * robot_params[node, 0]
                rigid_radius = 0.02 + 0.03 * robot_params[node, 1]
                soft_radius = 0.02 + 0.05 * robot_params[node, 2]
                max_children_num = 2
                if robot_params[node, 3] > 0.66:
                    max_children_num += 2
                elif 0.33 < robot_params[node, 3] < 0.66:
                    max_children_num += 1
                blend_weight = 0.5 * robot_params[node, 4]
                child_offset = np.array(robot_params[node, 5:8]) * 2 - 1
                child_offset /= np.linalg.norm(child_offset)
                parent_offset = np.array(robot_params[node, 8:11]) * 2 - 1
                parent_offset /= np.linalg.norm(parent_offset)
                s_node = StarNode(
                    node + 1,
                    max_children_num,
                    attach_offset_length,
                    rigid_radius,
                    soft_radius,
                    blend_weight,
                    (child_offset, parent_offset),
                )
                robot_topology.append(s_node)

            robot_topology[0].level = 0
            robot_topology[0].center = start_center

            # attach to parents with lower level value first
            for s_node in robot_topology[1:]:
                best_parent = None
                for parent in robot_topology:
                    if (
                        parent is not s_node
                        and parent.is_attachable()
                        and (best_parent is None or parent.level < best_parent.level)
                    ):
                        best_parent = parent
                best_parent.attach_child(
                    s_node, child_attach_offset_length_degrade_ratio_multiplier=0.9
                )
            robot_topologies.append(robot_topology)
        return robot_topologies
