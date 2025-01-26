# rise.pyi
from typing import List, Optional, Protocol, Union
import numpy as np

###############################################################################
# NumPy Data Types
###############################################################################

#: 3D Vector (x, y, z) with float32 components.
Vec3f = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
])

#: Quaternion (x, y, z, w) with float32 components.
Quat3f = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("w", np.float32),
])

#: Record describing a single voxel in the simulation.
#:
#: Fields:
#:     index (np.int32): Global voxel index (unique ID within this simulation).
#:     body (np.int32): Rigid body index to which this voxel belongs.
#:     index_x (np.int32): 3D lattice X index of this voxel in the soft body.
#:     index_y (np.int32): 3D lattice Y index of this voxel in the soft body.
#:     index_z (np.int32): 3D lattice Z index of this voxel in the soft body.
#:     connectivity (np.int32): The connectivity mask/level of this voxel.
#:     is_surface (np.int32): Boolean indicator (0 or 1) if the voxel is on the surface.
#:     is_rigid (np.int32): Boolean indicator (0 or 1) if the voxel is fully rigid.
#:     material (np.int32): An integer denoting the material or material ID of this voxel.
#:     expansion (np.float32): Current expansion coefficient of this voxel.
#:     orientation (Quat3f): Orientation of this voxel (as a quaternion).
#:     position (Vec3f): Position of this voxel in world coordinates.
#:     ppp_offset (Vec3f): Additional offset in the +x,+y,+z direction.
#:     nnn_offset (Vec3f): Additional offset in the -x,-y,-z direction.
#:     linear_velocity (Vec3f): Linear velocity in x, y, z directions.
#:     angular_velocity (Vec3f): Angular velocity around x, y, z axes.
#:     poissons_strain (Vec3f): Poisson's ratio strain in x, y, z directions.
RS_SimulationVoxelRecord = np.dtype([
    ("index", np.int32),
    ("body", np.int32),
    ("index_x", np.int32),
    ("index_y", np.int32),
    ("index_z", np.int32),
    ("connectivity", np.int32),
    ("is_surface", np.int32),
    ("is_rigid", np.int32),
    ("material", np.int32),
    ("expansion", np.float32),
    ("orientation", Quat3f),
    ("position", Vec3f),
    ("ppp_offset", Vec3f),
    ("nnn_offset", Vec3f),
    ("linear_velocity", Vec3f),
    ("angular_velocity", Vec3f),
    ("poissons_strain", Vec3f),
])

#: Record describing a link (edge) between two voxels or a tendon link.
#:
#: Fields:
#:     pos_voxel (np.int32): Index of the positive voxel involved in the link.
#:     neg_voxel (np.int32): Index of the negative voxel involved in the link.
#:     is_tendon (np.int32): Boolean indicator (0 or 1) if this is a tendon link.
#:     is_failed (np.int32): Boolean indicator (0 or 1) if the link has failed (ruptured).
#:     strain (np.float32): Current strain of this link.
#:     stress (np.float32): Current stress of this link.
#:     pos_position (Vec3f): Position of the positive voxel's side in world coordinates.
#:     neg_position (Vec3f): Position of the negative voxel's side in world coordinates.
RS_SimulationLinkRecord = np.dtype([
    ("pos_voxel", np.int32),
    ("neg_voxel", np.int32),
    ("is_tendon", np.int32),
    ("is_failed", np.int32),
    ("strain", np.float32),
    ("stress", np.float32),
    ("pos_position", Vec3f),
    ("neg_position", Vec3f),
])

#: Record describing a rigid body in the simulation.
#:
#: Fields:
#:     index (np.int32): Internal unique index of this rigid body instance.
#:     body (np.int32): Global ID for which body this is (if multiple exist).
#:     body_segment_id (np.int32): ID within the body for a particular segment.
#:     mass (np.float32): Total mass of the rigid body.
#:     orientation (Quat3f): Orientation of the rigid body (as a quaternion).
#:     com (Vec3f): Center of mass in world coordinates.
#:     linear_velocity (Vec3f): Linear velocity of the rigid body in x, y, z directions.
#:     angular_velocity (Vec3f): Angular velocity of the rigid body around x, y, z axes.
RS_SimulationRigidBodyRecord = np.dtype([
    ("index", np.int32),
    ("body", np.int32),
    ("body_segment_id", np.int32),
    ("mass", np.float32),
    ("orientation", Quat3f),
    ("com", Vec3f),
    ("linear_velocity", Vec3f),
    ("angular_velocity", Vec3f),
])

#: Record describing a joint between two rigid bodies.
#:
#: Fields:
#:     config_constraint_index (np.int32): Internal config-based constraint index.
#:     rigid_body_a (np.int32): Index of the first rigid body.
#:     rigid_body_b (np.int32): Index of the second rigid body.
#:     hinge_rotation_signal_id (np.int32): If this hinge uses a rotation signal, its ID.
#:     type (np.int32): Integer representing the joint type (e.g., hinge).
#:     position (Vec3f): Position of the joint in world coordinates.
#:     axis (Vec3f): Axis of the hinge or rotational axis in world coordinates.
#:     hinge_min (np.float32): Minimum hinge angle limit (if applicable).
#:     hinge_max (np.float32): Maximum hinge angle limit (if applicable).
#:     angle (np.float32): Current hinge angle (if applicable).
#:     torque (np.float32): Current torque (if applicable).
RS_SimulationJointRecord = np.dtype([
    ("config_constraint_index", np.int32),
    ("rigid_body_a", np.int32),
    ("rigid_body_b", np.int32),
    ("hinge_rotation_signal_id", np.int32),
    ("type", np.int32),
    ("position", Vec3f),
    ("axis", Vec3f),
    ("hinge_min", np.float32),
    ("hinge_max", np.float32),
    ("angle", np.float32),
    ("torque", np.float32),
])

###############################################################################
# Callback Protocol
###############################################################################

class RiseKernelCallback(Protocol):
    """Protocol for a Python callback that can be used during simulation.

    The callback is invoked during simulation steps to allow closed-loop control
    or data collection.

    Example usage:
        def my_callback(
            kernel_ids, time_points, frames, expansion_signals, rotation_signals
        ):
            # kernel_ids: list of integer IDs for each kernel being processed.
            # time_points: list of simulation times corresponding to each kernel.
            # frames: list of RiseFrame objects, each containing simulation data.
            # expansion_signals: list of float32 arrays or None, to modify expansions.
            # rotation_signals: list of float32 arrays or None, to modify rotations.

            # Implement your custom control logic here.
    """

    def __call__(
        self,
        kernel_ids: List[int],
        time_points: List[float],
        frames: List["RiseFrame"],
        expansion_signals: List[Optional[np.ndarray]],
        rotation_signals: List[Optional[np.ndarray]],
    ) -> None:
        """
        Args:
            kernel_ids (List[int]): A list of kernel IDs that map 1:1 with the other lists.
            time_points (List[float]): Current simulation times for each corresponding kernel.
            frames (List[RiseFrame]): List of current simulation frames for each kernel.
            expansion_signals (List[Optional[np.ndarray]]):
                List of float32 arrays (shape: [num_expansion_signals]) or None for each kernel.
                Modify in-place to control expansions.
            rotation_signals (List[Optional[np.ndarray]]):
                List of float32 arrays (shape: [num_rotation_signals]) or None for each kernel.
                Modify in-place to control rotations.
        """
        ...

###############################################################################
# Frame Class
###############################################################################

class RiseFrame:
    """Class representing a frame (snapshot) in a Rise simulation."""

    def step(self) -> int:
        """
        Returns:
            int: The current simulation step index for this frame.
        """

    def time_point(self) -> float:
        """
        Returns:
            float: The current simulation time (in seconds) for this frame.
        """

    def com(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (3,) with dtype float32 representing the
                global center of mass for the entire object or system in this frame.
        """

    def voxels(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (voxel_num,) with dtype RS_SimulationVoxelRecord
                representing all voxel data for this frame.
        """

    def links(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (link_num,) with dtype RS_SimulationLinkRecord
                representing all link (edge/tendon) data for this frame.
        """

    def rigid_bodies(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (rigid_body_num,) with dtype RS_SimulationRigidBodyRecord
                representing all rigid body data for this frame.
        """

    def joints(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (joint_num,) with dtype RS_SimulationJointRecord
                representing all joint data for this frame.
        """

    def voxel_positions(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (voxel_num, 3) with dtype float32,
                where each row is the (x, y, z) position of a voxel.
        """

    def voxel_linear_velocities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (voxel_num, 3) with dtype float32,
                where each row is the linear velocity (vx, vy, vz) of a voxel.
        """

    def voxel_angular_velocities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (voxel_num, 3) with dtype float32,
                where each row is the angular velocity (wx, wy, wz) of a voxel.
        """

    def voxel_poissons_strains(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (voxel_num, 3) with dtype float32,
                where each row is the Poisson's ratio strain in x, y, z directions for a voxel.
        """

    def rigid_body_mass(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 1D array of shape (rigid_body_num,) with dtype float32,
                where each element is the mass of a rigid body.
        """

    def rigid_body_center_of_mass(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (rigid_body_num, 3) with dtype float32,
                representing the center of mass of each rigid body.
        """

    def rigid_body_orientations(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (rigid_body_num, 4) with dtype float32,
                representing the orientation quaternion (w, x, y, z) of each rigid body.
        """

    def rigid_body_linear_velocities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (rigid_body_num, 3) with dtype float32,
                where each row is the linear velocity (vx, vy, vz) of a rigid body.
        """

    def rigid_body_angular_velocities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray:
                A 2D array of shape (rigid_body_num, 3) with dtype float32,
                where each row is the angular velocity (wx, wy, wz) of a rigid body.
        """

###############################################################################
# Main Rise Class
###############################################################################

class Rise:
    """Python interface for the Rise simulation engine.

    This class wraps the underlying C++ simulator and provides methods to
    initialize and run simulations on one or more GPU devices.
    """

    def __init__(self, devices: List[int] = ..., batch_size_per_device: int = ...) -> None:
        """
        Args:
            devices (List[int], optional):
                List of CUDA device indices on which to run simulations.
                By default, an empty list means run on CPU or pick a default device.
            batch_size_per_device (int, optional):
                Number of simulations to batch together per device for efficiency.
        """

    def run_sims(
        self,
        configs: List[List[str]],
        ids: List[int],
        callback: Optional[RiseKernelCallback] = ...,
        dt_update_interval: int = ...,
        collision_update_interval: int = ...,
        constraint_update_interval: int = ...,
        divergence_check_interval: int = ...,
        record_buffer_size: int = ...,
        max_steps: int = ...,
        save_result: bool = ...,
        save_record: bool = ...,
        policy: str = ...,
        log_level: str = ...,
    ) -> List[bool]:
        """
        Runs multiple simulations according to given configurations.

        Args:
            configs (List[List[str]]):
                A list of simulation configurations, each being a list of string parameters.
                Each inner list corresponds to one simulation instance.
            ids (List[int]):
                A list of unique integer IDs for each simulation to run. Must match
                the length of `configs`.
            callback (Optional[RiseKernelCallback], optional):
                A Python callable conforming to `RiseKernelCallback` that will be
                invoked during simulations. Default is None (no callback).
            dt_update_interval (int, optional):
                Simulation steps between time step (dt) updates. Default is 10.
            collision_update_interval (int, optional):
                Simulation steps between collision checks. Default is 10.
            constraint_update_interval (int, optional):
                Simulation steps between constraint solver updates. Default is 2.
            divergence_check_interval (int, optional):
                Simulation steps between divergence checks (for stability). Default is 100.
            record_buffer_size (int, optional):
                Internal size of the buffer for storing simulation records (frames). Default is 500.
            max_steps (int, optional):
                Maximum number of simulation steps to run before stopping. Default is 1_000_000.
            save_result (bool, optional):
                Whether to save final simulation results. Default is True.
            save_record (bool, optional):
                Whether to store time-series frames/records of the simulation. Default is True.
            policy (str, optional):
                Execution policy, either "batched" or "sequential". Default is "batched".
            log_level (str, optional):
                Log level for console/file output. Can be "err", "warn", "info", "debug", or "trace".
                Default is "info".

        Returns:
            List[bool]:
                A list of booleans, indicating success or failure for each simulation in order.
        """
