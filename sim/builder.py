import os
import itertools
import cc3d
import numpy as np
import torch as t
import lxml.etree as etree
from typing import Tuple, List, Union
from sklearn.decomposition import PCA
from scipy.ndimage import (
    binary_dilation,
    label,
)
from utils.plot import plt, plot_binary, plot_colored_rigid_by_id, plot_hinge_joints


class SimBuilder:
    def __init__(
        self,
        voxel_size: float = 0.01,
        hinge_axis_prediction_method: str = "pca",
        valid_min_rigid_ratio: float = 0.2,
        valid_min_joint_num: int = 1,
        valid_max_connected_components: int = 2,
        min_rigid_volume: int = 64,
        min_rigid_volume_to_contact_surface_ratio: float = 2.0,
        min_cycle_contact_size: int = 5,
        max_joint_adjacent_distance: int = 2,
        min_joint_adjacent_num: int = 10,
        hinge_limit: float = 1.5,
        hinge_torque: float = 5,
    ):
        self.voxel_size = voxel_size
        self.hinge_axis_prediction_method = hinge_axis_prediction_method
        self.valid_min_rigid_ratio = valid_min_rigid_ratio
        self.valid_min_joint_num = valid_min_joint_num
        self.valid_max_connected_components = valid_max_connected_components
        self.min_rigid_segment_size = min_rigid_volume
        self.min_rigid_volume_to_contact_surface_ratio = (
            min_rigid_volume_to_contact_surface_ratio
        )
        self.min_cycle_contact_size = min_cycle_contact_size
        self.max_joint_adjacent_distance = max_joint_adjacent_distance
        self.min_joint_adjacent_num = min_joint_adjacent_num
        self.hinge_limit = hinge_limit
        self.hinge_torque = hinge_torque
        # For recording intermediate processing data for visualization etc.
        self.log = {}
        self.stats = {}

    def is_valid(self):
        return self.log["is_valid"]

    def structure(self):
        """
        Returns:
            A dictionary with following keys:
                is_not_empty: A binary 3d array of shape [X, Y, Z]
                is_rigid: A binary 3d array of shape [X, Y, Z]
                segment_id: An id 3d array of shape [X, Y, Z], voxels labeled as 0 is
                    non-rigid.
                connections: A list of dicts with two keys, "position" and "axis",
                    both has value of a numpy array of shape [3]
        """
        if not self.is_valid():
            raise ValueError("Robot is invalid")
        return {
            "is_not_empty": self.log["is_not_empty"],
            "is_rigid": self.log["is_rigid_final"],
            "segment_id": self.log[
                "segment_ids_separate_rigid_segments_and_get_joints"
            ],
            "connections": self.log["connections"],
        }

    def statistics(self):
        """
        Returns:
            Dictionary with following keys:
            is_not_empty_accuracy:
                indicating the ratio of soft shells after
                fixing exposed rigid parts and before fixing,
                float, range (0~1)
            joint_num:
                number of joints, int, range (0~N)
            is_rigid_volume:
                total rigid volume, int
            total_volume:
                total volume, int
            cc3d:
                A dictionary, cc3d statistics of segment ids
            size: x_size, y_size, z_size, tuple (int, int, int)
        """
        return self.stats

    def build(
        self,
        logits: Union[t.Tensor, np.ndarray],
        sim_name: str = "robot",
        result_path: str = "",
        record_path: str = "",
        save_history: bool = True,
        save_h5_history: bool = True,
        print_summary: bool = False,
    ):
        """
        Args:


        Returns:
            A rsc config string for simulation
        """

        ROBOT_RSC = """
        <RSC>
            <Structure>
                <Bodies>
                    {}
                </Bodies>
                <Constraints>
                    {}
                </Constraints>
            </Structure>
            <Simulator>
                <Signal>
                    <ExpansionNum>0</ExpansionNum>
                    <RotationNum>{}</RotationNum>
                </Signal>
                <RecordHistory>
                    <RecordStepSize>200</RecordStepSize>
                    <RecordVoxel>1</RecordVoxel>
                    <RecordLink>0</RecordLink>
                    <RecordRigidBody>1</RecordRigidBody>
                    <RecordJoint>1</RecordJoint>
                </RecordHistory>
            </Simulator>
            <Save>
                {}
            </Save>
        </RSC>
        """
        self.log["is_valid"] = False
        result = self.parse(
            logits=logits.cpu().numpy() if t.is_tensor(logits) else logits,
        )
        if result is None:
            if print_summary:
                print(
                    f"[{sim_name}] Build failed, invalid reason: {self.log['invalid_reason']}"
                )
            return ""
        else:
            is_not_empty, is_rigid, rigid_segments, joints = result

        body_config, z_offset = self.get_body_config(
            is_not_empty, is_rigid, rigid_segments
        )

        self.log["is_valid"] = True
        if print_summary:
            print(
                f"[{sim_name}] "
                f"Voxel num: {np.sum(is_not_empty)} "
                f"rigid segment num: "
                f"{self.log['segment_id_num_separate_rigid_segments_and_get_joints']} "
                f"joint num: {len(joints)}"
            )
        constraint_configs = self.get_constraint_configs(joints, z_offset)
        save_config = self.get_save_config(
            sim_name=sim_name,
            result_path=result_path,
            record_path=record_path,
            save_history=save_history,
            save_h5_history=save_h5_history,
        )
        robot_rsc = ROBOT_RSC.format(
            body_config,
            "\n".join(constraint_configs),
            len(constraint_configs),
            save_config,
        )
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(robot_rsc, parser=parser)
        return etree.tostring(tree, pretty_print=True, encoding=str)

    def visualize(
        self,
        is_not_empty_ax=None,
        is_rigid_ax=None,
        segmentation_ax=None,
        plot_joints=True,
    ):
        if is_not_empty_ax is None and is_rigid_ax is None and segmentation_ax is None:
            fig = plt.figure(figsize=(12, 4))
            axs = fig.subplots(1, 3, subplot_kw={"projection": "3d"})

            if self.log["is_valid"]:
                plot_binary(
                    axs[0],
                    self.log["is_not_empty"],
                    (0, 1, 0, 1),
                )
                plot_binary(
                    axs[1],
                    self.log["is_rigid_final"],
                    (1, 0, 0, 1),
                )
                plot_colored_rigid_by_id(
                    axs[2],
                    self.log["is_rigid_final"],
                    self.log["segment_ids_separate_rigid_segments_and_get_joints"],
                    [
                        (1, 0, 0, 0.5),  # red
                        (0, 1, 0, 0.5),  # blue
                        (1, 1, 0, 0.5),  # yellow
                        (0, 0, 1, 0.5),  # green
                        (1, 0, 1, 0.5),  # purple
                        (0, 1, 1, 0.5),  # cyan
                        (1, 1, 1, 0.5),  # white
                        (0.5, 0.5, 0.5, 0.5),  # grey
                    ],
                )
                plot_hinge_joints(axs[2], self.log["connections"])
                plt.pause(10000)
        elif self.log["is_valid"]:
            if is_not_empty_ax is not None:
                plot_binary(
                    is_not_empty_ax,
                    self.log["is_not_empty"],
                    (0, 1, 0, 1),
                )
            if is_rigid_ax is not None:
                plot_binary(
                    is_rigid_ax,
                    self.log["is_rigid_final"],
                    (1, 0, 0, 1),
                )
            if segmentation_ax is not None:
                plot_colored_rigid_by_id(
                    segmentation_ax,
                    self.log["is_rigid_final"],
                    self.log["segment_ids_separate_rigid_segments_and_get_joints"],
                    [
                        (1, 0, 0, 0.5),  # red
                        (0, 1, 0, 0.5),  # blue
                        (1, 1, 0, 0.5),  # yellow
                        (0, 0, 1, 0.5),  # green
                        (1, 0, 1, 0.5),  # purple
                        (0, 1, 1, 0.5),  # cyan
                        (1, 1, 1, 0.5),  # white
                        (0.5, 0.5, 0.5, 0.5),  # grey
                    ],
                )
                if plot_joints:
                    plot_hinge_joints(segmentation_ax, self.log["connections"])

    def parse(
        self,
        logits: np.ndarray,
    ):
        assert logits.ndim == 4
        raw = logits.argmax(axis=0)
        nc, rx, ry, rz = logits.shape

        # find the largest connected component
        labels, num_labels = label((raw))
        component_sizes = np.bincount(labels.flatten())
        component_sizes[0] = 0
        largest_component_label = component_sizes.argmax()
        largest_components = labels == largest_component_label
        modified = np.where(largest_components, raw, 0)

        # add the soft surface if it is not already there
        boundary_mask = np.zeros((rx, ry, rz), dtype=bool)
        boundary_mask[2 : rx - 2, 2 : ry - 2, 2 : rz - 2] = True
        inner_mask = modified > 1 & boundary_mask
        surfaces = binary_dilation(modified > 1, iterations=2) & ~inner_mask
        modified = np.where(surfaces, 1, modified)

        # calculate the statistics
        non_empty = modified > 0
        rigidity = modified > 1
        rigid_ratio = (np.sum(rigidity) + 1) / (np.sum(non_empty) + 1)
        if rigid_ratio < self.valid_min_rigid_ratio:
            self.log["invalid_reason"] = (
                f"Below min rigid ratio {self.valid_min_rigid_ratio}, Ratio: {rigid_ratio}"
            )
            return None

        self.stats["is_not_empty_accuracy"] = min(1, (modified == raw).mean())
        self.log["is_not_empty"] = non_empty
        self.log["is_rigid"] = rigidity

        # Convert from logit segmentation to one hot segmentation
        segmentation = np.zeros(logits.shape, dtype=int)
        np.put_along_axis(segmentation, np.expand_dims(modified, axis=0), 1, axis=0)
        self.log["segmentation"] = segmentation

        # Convert from one hot segmentation to ids within is_rigid region
        segment_ids = np.zeros((rx, ry, rz), dtype=int)
        for channel in range(2, nc):
            segment_ids[segmentation[channel] == 1] = channel - 1
        self.log["segment_ids"] = segment_ids

        # Processing
        segment_ids = self.eliminate_small_regions(
            segment_ids, min_volume=self.min_rigid_segment_size
        )
        self.log["segment_ids_eliminate_small_region"] = segment_ids

        segment_ids = self.eliminate_low_volume_to_contact_ratio_regions(
            segment_ids,
            min_volume_to_contact_ratio=self.min_rigid_volume_to_contact_surface_ratio,
        )
        self.log["segment_ids_eliminate_low_volume_to_contact_ratio_regions"] = (
            segment_ids
        )

        segment_ids = self.eliminate_large_cycles(
            segment_ids, min_contact_size=self.min_cycle_contact_size
        )
        self.log["segment_ids_eliminate_large_cycles"] = segment_ids

        segment_ids, segment_id_num, connections, pruned_connections = (
            self.separate_rigid_segments_and_get_joints(
                segment_ids,
                max_adjacent_distance=self.max_joint_adjacent_distance,
                min_adjacent_num=self.min_joint_adjacent_num,
                hinge_method=self.hinge_axis_prediction_method,
            )
        )

        # Check number of connected components
        connected_components = self.find_connected_components(
            [connection["components"] for connection in connections], segment_id_num
        )
        if len(connected_components) > self.valid_max_connected_components:
            self.log["invalid_reason"] = (
                f"Exceeds max connected component num {self.valid_max_connected_components}, "
                f"connected components: {connected_components}"
            )
            return None

        if len(connections) < self.valid_min_joint_num:
            self.log["invalid_reason"] = (
                f"Below min joint num {self.valid_min_joint_num}, Joints: {connections}"
            )
            return None

        self.log["segment_ids_separate_rigid_segments_and_get_joints"] = segment_ids
        self.log["segment_id_num_separate_rigid_segments_and_get_joints"] = (
            segment_id_num
        )
        self.stats["cc3d"] = cc3d.statistics(segment_ids)
        # Add 1 since cc3d won't compute surface area between background 0 and non_empty 1
        self.stats["surface_area"] = cc3d.contacts(
            non_empty + 1,
            connectivity=6,
            surface_area=True,
        )[(1, 2)]
        self.stats["is_rigid_volume"] = np.sum(rigidity)
        self.stats["total_volume"] = np.sum(non_empty)
        self.log["connections"] = connections
        self.log["pruned_connections"] = pruned_connections
        rigidity = segment_ids > 0
        self.log["is_rigid_final"] = rigidity
        self.stats["joint_num"] = len(connections)

        return non_empty, rigidity, segment_ids, connections

    def eliminate_small_regions(self, ids, min_volume: int):
        """
        Args:
            ids: 3D numpy int array of shape [X, Y, Z]
            min_volume: Minimum volume of a region to not be eliminated

        Returns:
            New ids array with small regions merged to the connected largest
            region above min_volume, if the small region is not connected to
            any valid region, it is removed.
        """
        labels, num = cc3d.connected_components(
            ids,
            connectivity=6,
            return_N=True,
            out_dtype=np.uint32,
        )
        connection_graph = cc3d.contacts(labels, connectivity=6, surface_area=True)
        statistics = cc3d.statistics(labels)
        # First element is label, second element is contact size
        # For label 0 (background), pad with None at start
        contact = [None] + [[None, 0] for _ in range(num)]
        for connection, contact_size in connection_graph.items():
            # Save the neighbor with max contact region, and the neighbor
            # must be larger than threshold
            if (
                contact[connection[0]][1] < contact_size
                and statistics["voxel_counts"][connection[1]] >= min_volume
            ):
                contact[connection[0]][0] = connection[1]
                contact[connection[0]][1] = contact_size

            if (
                contact[connection[1]][1] < contact_size
                and statistics["voxel_counts"][connection[0]] >= min_volume
            ):
                contact[connection[1]][0] = connection[0]
                contact[connection[1]][1] = contact_size

        new_ids = np.copy(labels)
        for label in range(1, num + 1):
            if statistics["voxel_counts"][label] < min_volume:
                if contact[label][0] is not None:
                    new_ids[labels == label] = contact[label][0]
                else:
                    # Just remove it
                    new_ids[labels == label] = 0

        return new_ids

    def eliminate_low_volume_to_contact_ratio_regions(
        self, ids, min_volume_to_contact_ratio: float
    ):
        """
        Args:
            ids: 3D numpy int array of shape [X, Y, Z]
            min_volume_to_contact_ratio: Minimum volume to contact ratio of a region
                not to be eliminated

        Returns:
            New ids array with low volume to contact ratio regions merged to
            the connected largest region above.
        """
        labels, num = cc3d.connected_components(
            ids,
            connectivity=6,
            return_N=True,
            out_dtype=np.uint32,
        )
        connection_graph = cc3d.contacts(labels, connectivity=6, surface_area=True)
        statistics = cc3d.statistics(labels)
        # First element is label, second element is contact size
        # For label 0 (background), pad with None at start
        contact = [None] + [[None, 0] for _ in range(num)]
        for connection, contact_size in connection_graph.items():
            # Save the neighbor with max contact region
            if contact[connection[0]][1] < contact_size:
                contact[connection[0]][0] = connection[1]
                contact[connection[0]][1] = contact_size

            if contact[connection[1]][1] < contact_size:
                contact[connection[1]][0] = connection[0]
                contact[connection[1]][1] = contact_size

        new_ids = np.copy(labels)
        for label in range(1, num + 1):
            if contact[label][0] is not None:
                if (
                    statistics["voxel_counts"][label] / contact[label][1]
                    < min_volume_to_contact_ratio
                ):
                    new_ids[labels == label] = contact[label][0]

        return new_ids

    def eliminate_large_cycles(self, ids, min_contact_size: int):
        """

        Args:
            ids: 3D numpy int array of shape [X, Y, Z]
            min_contact_size: Minimum contact size to consider two regions have an edge between them

        Returns:
            New ids array with regions connected forming a cycle merged as one part.
        """
        while True:
            labels, num = cc3d.connected_components(
                ids,
                connectivity=6,
                return_N=True,
                out_dtype=np.uint32,
            )
            connection_graph = cc3d.contacts(labels, connectivity=6, surface_area=True)
            filtered_connection_graph = [
                connection
                for connection, contact_size in connection_graph.items()
                if contact_size >= min_contact_size
            ]
            cycle = self.find_cycle(filtered_connection_graph, num)
            if cycle is None:
                return labels
            new_ids = np.copy(labels)
            for label in cycle:
                # Assign a new label (original labels are 0 and 1 to n+1)
                new_ids[labels == label] = num + 2
            ids = new_ids

    def separate_rigid_segments_and_get_joints(
        self,
        segment_ids: np.ndarray,
        max_adjacent_distance: int,
        min_adjacent_num: int,
        hinge_method: str,
    ):
        segment_labels, label_num = cc3d.connected_components(
            segment_ids,
            connectivity=6,
            return_N=True,
            out_dtype=np.uint32,
        )
        statistics = cc3d.statistics(segment_labels)

        l_indices = np.indices(segment_labels.shape).reshape(3, -1).T

        structure = np.ones([max_adjacent_distance * 2 + 1] * 3, dtype=bool)
        s_indices = np.indices(structure.shape)
        structure[s_indices[0], s_indices[1], s_indices[2]] = (
            np.linalg.norm(s_indices - max_adjacent_distance, axis=0)
            <= max_adjacent_distance
        )
        neighbor_num = np.sum(structure)
        neighbor_labels = self.convolve_gather(segment_labels, structure).reshape(
            -1, neighbor_num
        )

        current_labels = segment_labels.reshape(-1)
        adjacency = {}
        for i in range(2, label_num + 1):
            for j in range(1, i):
                # First list for positions of smaller label voxels,
                # second list for positions of larger label voxels.
                adjacency[(i, j)] = adjacency[(j, i)] = [[], []]

        for c_label, c_pos, n_labels in zip(current_labels, l_indices, neighbor_labels):
            for n_label in n_labels:
                if n_label != c_label and c_label != 0 and n_label != 0:
                    small_list, large_list = adjacency[(c_label, n_label)]
                    if n_label < c_label:
                        large_list.append(c_pos)
                    else:
                        small_list.append(c_pos)

        connections = []
        pruned_connections = []
        for i in range(2, label_num + 1):
            for j in range(1, i):
                if (
                    len(adjacency[(i, j)][0]) > min_adjacent_num
                    and len(adjacency[(i, j)][1]) > min_adjacent_num
                ):
                    points = np.stack(adjacency[(i, j)][0] + adjacency[(i, j)][1])
                    position = np.mean(points, axis=0)
                    if hinge_method == "pca":
                        pca = PCA(n_components=3)
                        pca.fit(points)
                        hinge_axis = pca.components_[0]
                        hinge_axis /= np.linalg.norm(hinge_axis)
                    elif hinge_method == "cross_product":
                        v1 = statistics["centroids"][i] - position
                        v2 = statistics["centroids"][j] - position
                        hinge_axis = np.cross(v1, v2)
                        if np.linalg.norm(hinge_axis) != 0:
                            hinge_axis /= np.linalg.norm(hinge_axis)
                    else:
                        raise ValueError(f"Invalid hinge method {hinge_method}")
                    connections.append(
                        {
                            "components": (i, j),
                            "position": position,
                            "axis": hinge_axis,
                            "size": (
                                len(adjacency[(i, j)][0]) + len(adjacency[(i, j)][1])
                            )
                            / 2,
                        }
                    )

            # Prune connections within 2 threshold distance of each other
            while True:
                prune = None
                for conn_1, conn_2 in itertools.combinations(connections, 2):
                    if (
                        np.linalg.norm(conn_1["position"] - conn_2["position"])
                        < max_adjacent_distance * 2
                    ):
                        prune = conn_1
                        break
                if prune is None:
                    break
                connections.remove(prune)
                pruned_connections.append({"connection": prune, "reason": "too close"})

            # Find cycles caused by adding joints and prune joint with the smallest size
            while True:
                connection_graph = [
                    connection["components"] for connection in connections
                ]
                connection_lut = {
                    connection["components"]: idx
                    for idx, connection in enumerate(connections)
                }
                cycle = self.find_cycle(connection_graph, label_num)
                if cycle is None:
                    break
                # Find all corresponding edges
                cycles_edges = []
                for i in range(len(cycle)):
                    end_1 = cycle[i - 1]
                    end_2 = cycle[i]
                    if (end_1, end_2) in connection_lut:
                        cycles_edges.append(connection_lut[(end_1, end_2)])
                    else:
                        cycles_edges.append(connection_lut[(end_2, end_1)])
                min_idx = None
                min_size = np.inf
                for idx in cycles_edges:
                    size = connections[idx]["size"]
                    if min_size > size:
                        min_idx = idx
                        min_size = size
                pruned_connections.append(
                    {
                        "connection": connections[min_idx],
                        "reason": f"forming a cycle {cycle}",
                    }
                )
                connections.pop(min_idx)
        return segment_labels, label_num, connections, pruned_connections

    @staticmethod
    def find_cycle(connection_graph: List[Tuple[int, int]], node_num: int):
        """
        Args:
            connection_graph: Edges of a graph, node index start from 1
            node_num: Number of nodes in the graph, from 1 to N

        Returns:
            None if cycle is not found, else a list containing every node
            in the cycle.
        """
        # DFS algorithm
        adjacency = [set() for _ in range(node_num)]
        # 0: not visited, 1: visited but not finished cycle checking
        # 2: visited and finished cycle checking
        state = [0 for _ in range(node_num)]
        parent = [None for _ in range(node_num)]
        cycle_start, cycle_end = None, None
        for connection in connection_graph:
            adjacency[connection[0] - 1].add(connection[1] - 1)
            adjacency[connection[1] - 1].add(connection[0] - 1)

        def dfs_find_cycle(node: int, parent_node: int):
            nonlocal cycle_start, cycle_end
            state[node] = 1
            for adj_node in adjacency[node]:
                if adj_node == parent_node:
                    continue
                if state[adj_node] == 0:
                    parent[adj_node] = node
                    if dfs_find_cycle(adj_node, node):
                        return True
                else:
                    cycle_end = node
                    cycle_start = adj_node
                    return True
            state[node] = 2
            return False

        for node in range(node_num):
            if state[node] == 0 and dfs_find_cycle(node, parent[node]):
                break

        if cycle_end is None:
            return None
        else:
            cycle = []
            while cycle_end != cycle_start:
                cycle.append(cycle_end + 1)
                cycle_end = parent[cycle_end]
            cycle.append(cycle_start + 1)
            cycle = list(reversed(cycle))
            return cycle

    @staticmethod
    def find_connected_components(
        connection_graph: List[Tuple[int, int]], node_num: int
    ):
        """
        Args:
            connection_graph: Edges of a graph, node index start from 1
            node_num: Number of nodes in the graph, from 1 to N

        Returns:
            A list of every connected component in the graph.
        """
        # DFS algorithm
        adjacency = [set() for _ in range(node_num)]
        # 0: not visited, 1: visited
        state = [0 for _ in range(node_num)]
        for connection in connection_graph:
            adjacency[connection[0] - 1].add(connection[1] - 1)
            adjacency[connection[1] - 1].add(connection[0] - 1)

        def dfs_find_connected_component(node: int, connected_component: List[int]):
            state[node] = 1
            connected_component.append(node)
            for adj_node in adjacency[node]:
                if state[adj_node] == 0:
                    dfs_find_connected_component(adj_node, connected_component)
            return connected_component

        connected_components = []
        for node in range(node_num):
            if state[node] == 0:
                connected_components.append(dfs_find_connected_component(node, []))

        return connected_components

    @staticmethod
    def convolve_gather(input: np.ndarray, structure: np.ndarray):
        """
        Args:
            input: 3D numpy int array of shape [X, Y, Z]
            structure: 3D numpy bool mask array

        Returns:
            Neighboring values selected by structure at every voxel
            Eg: suppose there is a 3x3x3 mask with 6 connectivity,
            7 elements will be selected. so output size would be
            [X, Y, Z, 7]
        """
        padded_input = np.zeros(
            [input.shape[i] + structure.shape[i] - 1 for i in range(3)],
            dtype=input.dtype,
        )
        pad_neg = [structure.shape[i] // 2 for i in range(3)]
        padded_input[
            pad_neg[0] : pad_neg[0] + input.shape[0],
            pad_neg[1] : pad_neg[1] + input.shape[1],
            pad_neg[2] : pad_neg[2] + input.shape[2],
        ] = input
        indices = np.indices(input.shape).reshape(3, -1, 1)
        offsets = np.indices(structure.shape).reshape(3, 1, -1)
        full_indices = indices + offsets
        all_elements = padded_input[full_indices[0], full_indices[1], full_indices[2]]
        selected_elements = all_elements[:, structure.flatten()]
        return selected_elements.reshape(list(input.shape) + [-1])

    def get_body_config(
        self, is_not_empty: np.ndarray, is_rigid: np.ndarray, rigid_segments: np.ndarray
    ):
        BODY_CONFIG = """
        <Body ID="1">
            <Orientation>0,0,0,1</Orientation>
            <OriginPosition>0,0,0</OriginPosition>
            {}
            <MaterialID>
                {}
            </MaterialID>
            <SegmentID>
                {}
            </SegmentID>
            <SegmentType>
                {}
            </SegmentType>
        </Body>
        """
        SHAPE_TEMPLATE = """
            <X_Voxels>{}</X_Voxels>
            <Y_Voxels>{}</Y_Voxels>
            <Z_Voxels>{}</Z_Voxels>
        """
        LAYER_TEMPLATE = "<Layer>{}</Layer>"

        non_empty_x = np.sum(is_not_empty.astype(int), axis=(1, 2)) > 0
        start_x = np.argmax(non_empty_x)
        end_x = len(non_empty_x) - np.argmax(np.flip(non_empty_x))
        non_empty_y = np.sum(is_not_empty.astype(int), axis=(0, 2)) > 0
        start_y = np.argmax(non_empty_y)
        end_y = len(non_empty_y) - np.argmax(np.flip(non_empty_y))

        non_empty_z = np.sum(is_not_empty.astype(int), axis=(0, 1)) > 0
        start_layer = np.argmax(non_empty_z)
        end_layer = len(non_empty_z) - np.argmax(np.flip(non_empty_z))
        x_size = end_x - start_x
        y_size = end_y - start_y
        z_size = end_layer - start_layer

        x_layer_size = is_not_empty.shape[0]
        y_layer_size = is_not_empty.shape[1]
        layer_size = x_layer_size * y_layer_size

        self.stats["size"] = (x_size, y_size, z_size)

        material_id, segment_id, segment_type = [], [], []
        m_id = np.zeros(is_not_empty.shape, dtype=int)
        s_id = np.zeros(is_not_empty.shape, dtype=int)
        s_type = np.zeros(is_not_empty.shape, dtype=int)
        m_id[is_not_empty] = 1
        m_id[is_rigid] = 2
        s_id[is_not_empty] = 1
        s_id = np.where(is_rigid, rigid_segments + 1, s_id)
        s_type[is_rigid] = 1
        # Transform from X, Y, Z to ZYX
        m_id = m_id[:, :, start_layer:end_layer].transpose(2, 1, 0).flatten()
        s_id = s_id[:, :, start_layer:end_layer].transpose(2, 1, 0).flatten()
        s_type = s_type[:, :, start_layer:end_layer].transpose(2, 1, 0).flatten()

        for z in range(z_size):
            offset = z * layer_size
            material_id.append(
                LAYER_TEMPLATE.format(
                    ",".join(map(str, m_id[offset : offset + layer_size]))
                )
            )
            segment_id.append(
                LAYER_TEMPLATE.format(
                    ",".join(map(str, s_id[offset : offset + layer_size]))
                )
            )
            segment_type.append(
                LAYER_TEMPLATE.format(
                    ",".join(map(str, s_type[offset : offset + layer_size]))
                )
            )
        shape = SHAPE_TEMPLATE.format(x_layer_size, y_layer_size, z_size)

        return (
            BODY_CONFIG.format(
                shape,
                "\n".join(material_id),
                "\n".join(segment_id),
                "\n".join(segment_type),
            ),
            start_layer,
        )

    def get_constraint_configs(self, joints: List[dict], z_offset):
        HINGE_CONSTRAINT_TEMPLATE = """
        <Constraint>
            <Type>HINGE_JOINT</Type>
            <RigidBodyA>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyA>
            <RigidBodyB>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyB>
            <HingeRotationSignalID>{}</HingeRotationSignalID>
            <HingeAAxis>{}, {}, {}</HingeAAxis>
            <HingeBAxis>{}, {}, {}</HingeBAxis>
            <HingeMin>{}</HingeMin>
            <HingeMax>{}</HingeMax>
            <HingeTorque>{}</HingeTorque>
        </Constraint>
        """
        constraints = []
        for idx, joint in enumerate(joints):
            constraints.append(
                HINGE_CONSTRAINT_TEMPLATE.format(
                    joint["components"][0] + 1,
                    joint["position"][0] * self.voxel_size,
                    joint["position"][1] * self.voxel_size,
                    (joint["position"][2] - z_offset) * self.voxel_size,
                    joint["components"][1] + 1,
                    joint["position"][0] * self.voxel_size,
                    joint["position"][1] * self.voxel_size,
                    (joint["position"][2] - z_offset) * self.voxel_size,
                    idx,
                    joint["axis"][0],
                    joint["axis"][1],
                    joint["axis"][2],
                    -joint["axis"][0],
                    -joint["axis"][1],
                    -joint["axis"][2],
                    -self.hinge_limit,
                    self.hinge_limit,
                    self.hinge_torque,
                )
            )
        return constraints

    def get_save_config(
        self,
        sim_name: str = "robot",
        result_path: str = "",
        record_path: str = "",
        save_history: bool = True,
        save_h5_history: bool = True,
    ):
        SAVE_CONFIG = """
        <ResultPath>{}</ResultPath>
        <Record>
            {}
        </Record>
        """
        TXT_RECORD_CONFIG = """
        <Text>
            <Rescale>0.001</Rescale>
            <Path>{}</Path>
        </Text>
        """
        H5_RECORD_CONFIG = """
        <HDF5>
            <Path>{}</Path>
        </HDF5>
        """
        records = []
        if save_history:
            records.append(
                TXT_RECORD_CONFIG.format(
                    os.path.join(record_path, f"{sim_name}.history")
                )
            )
        if save_h5_history:
            records.append(
                H5_RECORD_CONFIG.format(
                    os.path.join(record_path, f"{sim_name}.h5_history")
                )
            )
        return SAVE_CONFIG.format(
            os.path.join(result_path, f"{sim_name}.result"), "\n".join(records)
        )
