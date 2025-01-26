import h5py
import msgpack
import numpy as np
from typing import List


class RiseSimulationDataReader:
    def __init__(self, file_path):
        """
        Initialize the data reader by opening the HDF5 file.

        :param file_path: Path to the HDF5 file generated by RS_SimulationRecorderHdf5ForVis.
        """
        self.file_path = file_path
        self.file = h5py.File(self.file_path, "r")

        # Access groups
        self.config_group = self.file["/config"]
        self.time_group = self.file["/frames/time"]
        self.voxel_group = self.file["/frames/voxel"]
        self.link_group = self.file["/frames/link"]
        self.rigid_body_group = self.file["/frames/rigid_body"]
        self.joint_group = self.file["/frames/joint"]

    def close(self):
        """
        Close the HDF5 file.
        """
        self.file.close()

    def read_config(self):
        """
        Read the configuration data.

        :return: Byte array containing the serialized configuration.
        """
        config_dataset = self.config_group["config"]
        config = msgpack.unpackb(config_dataset[:], raw=False)
        return config

    def read_time_steps(self):
        """
        Read the simulation time steps and time points.

        :return: Tuple of numpy arrays (steps, time_points)
        """
        steps = self.time_group["step"][:]
        time_points = self.time_group["time_point"][:]
        return steps, time_points

    def read_voxel_data(self, frame_index=None):
        """
        Read voxel data for a specific frame or all frames.

        :param frame_index: Index of the frame to read. If None, reads all frames.
        :return: Dictionary containing voxel data arrays.
        """
        voxel_num = self.voxel_group["voxel_num"][:]
        voxel_offset = self.voxel_group["voxel_offset"][:]

        # Determine the range of indices to read
        if frame_index is not None:
            start = voxel_offset[frame_index]
            end = start + voxel_num[frame_index]
        else:
            start = 0
            end = voxel_offset[-1] + voxel_num[-1]

        # Reshape multi-dimensional data appropriately
        voxel_data = {
            "index": self.voxel_group["index"][start:end],
            "body": self.voxel_group["body"][start:end],
            "body_segment_id": self.voxel_group["body_segment_id"][start:end],
            "index_xyz": self.voxel_group["index_xyz"][start:end].reshape(-1, 3),
            "connectivity": self.voxel_group["connectivity"][start:end],
            "is_surface": self.voxel_group["is_surface"][start:end],
            "is_rigid": self.voxel_group["is_rigid"][start:end],
            "material": self.voxel_group["material"][start:end],
            "expansion": self.voxel_group["expansion"][start:end],
            "orientation": self.voxel_group["orientation"][start:end].reshape(-1, 4),
            "position": self.voxel_group["position"][start:end].reshape(-1, 3),
            "ppp_offset": self.voxel_group["ppp_offset"][start:end].reshape(-1, 3),
            "nnn_offset": self.voxel_group["nnn_offset"][start:end].reshape(-1, 3),
            "linear_velocity": self.voxel_group["linear_velocity"][start:end].reshape(
                -1, 3
            ),
            "angular_velocity": self.voxel_group["angular_velocity"][start:end].reshape(
                -1, 3
            ),
            "poissons_strain": self.voxel_group["poissons_strain"][start:end].reshape(
                -1, 3
            ),
        }
        return voxel_data

    def read_link_data(self, frame_index=None):
        """
        Read link data for a specific frame or all frames.

        :param frame_index: Index of the frame to read. If None, reads all frames.
        :return: Dictionary containing link data arrays.
        """
        link_num = self.link_group["link_num"][:]
        link_offset = self.link_group["link_offset"][:]

        if frame_index is not None:
            start = link_offset[frame_index]
            end = start + link_num[frame_index]
        else:
            start = 0
            end = link_offset[-1] + link_num[-1]

        link_data = {
            "pos_voxel": self.link_group["pos_voxel"][start:end],
            "neg_voxel": self.link_group["neg_voxel"][start:end],
            "is_tendon": self.link_group["is_tendon"][start:end],
            "is_failed": self.link_group["is_failed"][start:end],
            "strain": self.link_group["strain"][start:end],
            "stress": self.link_group["stress"][start:end],
            "pos_position": self.link_group["pos_position"][start:end].reshape(-1, 3),
            "neg_position": self.link_group["neg_position"][start:end].reshape(-1, 3),
        }
        return link_data

    def read_rigid_body_data(self, frame_index=None):
        """
        Read rigid body data for a specific frame or all frames.

        :param frame_index: Index of the frame to read. If None, reads all frames.
        :return: Dictionary containing rigid body data arrays.
        """
        rb_num = self.rigid_body_group["rigid_body_num"][:]
        rb_offset = self.rigid_body_group["rigid_body_offset"][:]

        if frame_index is not None:
            start = rb_offset[frame_index]
            end = start + rb_num[frame_index]
        else:
            start = 0
            end = rb_offset[-1] + rb_num[-1]

        rb_data = {
            "index": self.rigid_body_group["index"][start:end],
            "body": self.rigid_body_group["body"][start:end],
            "body_segment_id": self.rigid_body_group["body_segment_id"][start:end],
            "mass": self.rigid_body_group["mass"][start:end],
            "orientation": self.rigid_body_group["orientation"][start:end].reshape(
                -1, 4
            ),
            "com": self.rigid_body_group["com"][start:end].reshape(-1, 3),
            "linear_velocity": self.rigid_body_group["linear_velocity"][
                start:end
            ].reshape(-1, 3),
            "angular_velocity": self.rigid_body_group["angular_velocity"][
                start:end
            ].reshape(-1, 3),
        }
        return rb_data

    def read_joint_data(self, frame_index=None):
        """
        Read joint data for a specific frame or all frames.

        :param frame_index: Index of the frame to read. If None, reads all frames.
        :return: Dictionary containing joint data arrays.
        """
        joint_num = self.joint_group["joint_num"][:]
        joint_offset = self.joint_group["joint_offset"][:]

        if frame_index is not None:
            start = joint_offset[frame_index]
            end = start + joint_num[frame_index]
        else:
            start = 0
            end = joint_offset[-1] + joint_num[-1]

        joint_data = {
            "config_constraint_index": self.joint_group["config_constraint_index"][
                start:end
            ],
            "rigid_body_a": self.joint_group["rigid_body_a"][start:end],
            "rigid_body_b": self.joint_group["rigid_body_b"][start:end],
            "hinge_rotation_signal_id": self.joint_group["hinge_rotation_signal_id"][
                start:end
            ],
            "type": self.joint_group["type"][start:end],
            "position": self.joint_group["position"][start:end].reshape(-1, 3),
            "axis": self.joint_group["axis"][start:end].reshape(-1, 3),
            "hinge_min": self.joint_group["hinge_min"][start:end],
            "hinge_max": self.joint_group["hinge_max"][start:end],
            "angle": self.joint_group["angle"][start:end],
            "torque": self.joint_group["torque"][start:end],
        }
        return joint_data

    def get_number_of_frames(self):
        """
        Get the total number of frames in the simulation.

        :return: Integer representing the number of frames.
        """
        return len(self.time_group["step"])

    def read_frame(self, frame_index, bodies: List[int] = None):
        """
        Read all data for a specific frame.

        :param frame_index: Index of the frame to read.
        :param bodies: List of rigid body indices to include in the frame data.
        :return: Dictionary containing time, voxel, link, rigid body, and joint data.
        """
        steps, time_points = self.read_time_steps()

        # Read all data for the frame
        voxel_data = self.read_voxel_data(frame_index)
        link_data = self.read_link_data(frame_index)
        rigid_body_data = self.read_rigid_body_data(frame_index)
        joint_data = self.read_joint_data(frame_index)

        if bodies is not None:
            # Filter voxels based on specified bodies
            voxel_mask = np.isin(voxel_data["body"], bodies)
            voxel_data = {key: value[voxel_mask] for key, value in voxel_data.items()}

            # Filter rigid bodies based on specified bodies
            rigid_body_mask = np.isin(rigid_body_data["index"], bodies)
            rigid_body_data = {
                key: value[rigid_body_mask] for key, value in rigid_body_data.items()
            }

            # Filter links based on filtered voxels
            link_mask = np.isin(link_data["pos_voxel"], voxel_data["index"]) & np.isin(
                link_data["neg_voxel"], voxel_data["index"]
            )
            link_data = {key: value[link_mask] for key, value in link_data.items()}

            # Filter joints based on filtered rigid bodies
            joint_mask = np.isin(
                joint_data["rigid_body_a"], rigid_body_data["index"]
            ) & np.isin(joint_data["rigid_body_b"], rigid_body_data["index"])
            joint_data = {key: value[joint_mask] for key, value in joint_data.items()}

        frame_data = {
            "step": steps[frame_index],
            "time_point": time_points[frame_index],
            "voxels": voxel_data,
            "links": link_data,
            "rigid_bodies": rigid_body_data,
            "joints": joint_data,
        }
        return frame_data

    def __len__(self):
        return len(self.time_group["step"])

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and close the HDF5 file.
        """
        self.close()
