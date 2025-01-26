import bpy
import numpy as np
import result_process.blender.generate as generate
import result_process.blender.reader as reader
from math import cos, sin, sqrt
from mathutils import Vector
from typing import Callable, Tuple
from sim.builder import SimBuilder


def setup_scene(
    scene_blend_file_path=None, output_file_path=None, use_gpu: bool = True
):
    if scene_blend_file_path is not None:
        bpy.ops.wm.open_mainfile(filepath=scene_blend_file_path)
    else:
        # Get the object by name (default cube is named "Cube")
        cube = bpy.data.objects.get("Cube")

        # Check if the cube exists and delete it
        if cube:
            bpy.data.objects.remove(cube)

    # Set the unit system to 'METRIC'
    bpy.context.scene.unit_settings.system = "METRIC"

    # Set the unit scale to 1 (1 Blender unit = 1 meter)
    bpy.context.scene.unit_settings.scale_length = 1.0

    ## Set up rendering settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.image_settings.file_format = "PNG"
    if output_file_path is not None:
        bpy.context.scene.render.filepath = output_file_path

    if use_gpu:
        # Set to use GPU instead of CPU
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
            "CUDA"  # Use 'CUDA' for NVIDIA, 'OPTIX' for OptiX (NVIDIA RTX), or 'OPENCL' for AMD
        )
        bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
        devices = bpy.context.preferences.addons["cycles"].preferences.devices

        if not devices:
            raise RuntimeError("Unsupported device type")

        # Enable all available GPU devices
        bpy.context.scene.cycles.device = "GPU"
        devices[0].use = True  # Enable all detected GPUs


def setup_robot_scene(
    h5_file_path: str,
    frame: int,
    color_domain: str = "poissons_strain",
    color_map: Callable[[np.ndarray], np.ndarray] = None,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    mode: str = "voxels",
    opacity: float = 1.0,
    voxel_size: float = 0.01,
):
    # Use the RS_SimulationDataReader to read the data
    with reader.RiseSimulationDataReader(h5_file_path) as r:
        frame_collection = bpy.data.collections.new(f"frame_{frame}")
        bpy.context.scene.collection.children.link(frame_collection)

        generate.generate_robot(
            frame_collection,
            r,
            frame,
            color_domain=color_domain,
            color_map=color_map,
            transform=transform,
            mode=mode,
            opacity=opacity,
            voxel_size=voxel_size,
        )


def setup_robot_abstract_scene(
    h5_file_path: str,
    frame: int,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    opacity: float = 1.0,
    voxel_size: float = 0.01,
    draw_shell: bool = True,
    draw_bone: bool = True,
    draw_node: bool = True,
    draw_edge: bool = True,
):
    # Use the RS_SimulationDataReader to read the data
    with reader.RiseSimulationDataReader(h5_file_path) as r:
        frame_collection = bpy.data.collections.new(f"frame_{frame}")
        bpy.context.scene.collection.children.link(frame_collection)

        generate.generate_robot_abstract(
            frame_collection,
            r,
            frame,
            transform=transform,
            opacity=opacity,
            voxel_size=voxel_size,
            draw_shell=draw_shell,
            draw_bone=draw_bone,
            draw_node=draw_node,
            draw_edge=draw_edge,
        )


def setup_robot_abstract_scene_with_joint(
    h5_file_path: str,
    frame: int,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    opacity: float = 1.0,
    voxel_size: float = 0.01,
):
    # Use the RS_SimulationDataReader to read the data
    with reader.RiseSimulationDataReader(h5_file_path) as r:
        frame_collection = bpy.data.collections.new(f"frame_{frame}")
        bpy.context.scene.collection.children.link(frame_collection)

        generate.generate_robot_abstract_with_joint(
            frame_collection,
            r,
            frame,
            transform=transform,
            opacity=opacity,
            voxel_size=voxel_size,
        )


def setup_robot_structure_scene(
    builder: SimBuilder,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    shell_color=(0, 0, 1),
    voxel_size: float = 0.01,
):
    robot_structure_collection = bpy.data.collections.new(f"robot_structure")
    bpy.context.scene.collection.children.link(robot_structure_collection)
    generate.generate_robot_structure(
        robot_structure_collection,
        builder,
        transform=transform,
        shell_color=shell_color,
        voxel_size=voxel_size,
    )


def setup_diagonal_camera(center: Tuple[float, float, float], distance: float):
    # Ensure center is a Vector
    center_vector = Vector(center)

    # Calculate camera position at a diagonal direction from the center
    camera_position = center_vector + Vector((distance, distance, distance)) / sqrt(3)

    # Add a new camera object
    bpy.ops.object.camera_add(location=camera_position)

    # Get reference to the newly added camera
    camera = bpy.context.object

    # Set camera to look at the center point
    direction = camera_position - center_vector
    rot_quat = direction.to_track_quat("Z", "Y")

    # Apply the rotation to the camera
    camera.rotation_euler = rot_quat.to_euler()

    # Set the camera as the active camera
    bpy.context.scene.camera = camera


def setup_azimuth_camera(
    center: Tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
):
    # Ensure center is a Vector
    center_vector = Vector(center)

    # Calculate the camera position based on azimuth and elevation angles
    x = center_vector.x + distance * cos(elevation) * cos(azimuth)
    y = center_vector.y + distance * cos(elevation) * sin(azimuth)
    z = center_vector.z + distance * sin(elevation)

    camera_position = Vector((x, y, z))

    # Add a new camera object at the calculated position
    bpy.ops.object.camera_add(location=camera_position)

    # Get reference to the newly added camera
    camera = bpy.context.object

    # Set the camera to look at the center point
    direction = camera_position - center_vector
    rot_quat = direction.to_track_quat("Z", "Y")

    # Apply the rotation to the camera
    camera.rotation_euler = rot_quat.to_euler()

    # Set the camera as the active camera
    bpy.context.scene.camera = camera


def main_collection():
    return bpy.context.scene.collection


def render():
    bpy.ops.render.render(write_still=True)


def save(scene_blend_file_path):
    bpy.ops.wm.save_as_mainfile(filepath=scene_blend_file_path)
