import math
import bpy
import bmesh
import mathutils
import colorsys
import numpy as np
from typing import Callable, Tuple, List
from scipy.interpolate import RegularGridInterpolator
from mathutils import Vector, Quaternion
from .reader import RiseSimulationDataReader
from sim.builder import SimBuilder


def bilinear_interpolation(x, y, x1, x2, y1, y2, Q11, Q12, Q21, Q22):
    """
    Perform bilinear interpolation for a point (x, y) given four surrounding points
    (x1, y1), (x1, y2), (x2, y1), (x2, y2) with their corresponding values
    Q11, Q12, Q21, Q22.

    Args:
        x (float): x-coordinate of the target point.
        y (float): y-coordinate of the target point.
        x1, x2 (float): x-coordinates of the bounding points.
        y1, y2 (float): y-coordinates of the bounding points.
        Q11, Q12, Q21, Q22 (float): Values at the bounding points.
            Q11: value at (x1, y1)
            Q12: value at (x1, y2)
            Q21: value at (x2, y1)
            Q22: value at (x2, y2)

    Returns:
        float: Interpolated value at (x, y).
    """
    if x1 == x2:
        R1 = Q11
        R2 = Q12
    else:
        # Interpolate in the x-direction
        R1 = ((x2 - x) * Q11 + (x - x1) * Q21) / (x2 - x1)
        R2 = ((x2 - x) * Q12 + (x - x1) * Q22) / (x2 - x1)

    if y1 == y2:
        return R1  # No y interpolation needed if y1 == y2

        # Interpolate in the y-direction
    return ((y2 - y) * R1 + (y - y1) * R2) / (y2 - y1)


def get_object_override(active_object, objects: list = None):
    if objects is None:
        objects = []
    else:
        objects = list(objects)

    if not active_object in objects:
        objects.append(active_object)

    assert all(isinstance(obj, bpy.types.Object) for obj in objects)

    return dict(
        selectable_objects=objects,
        selected_objects=objects,
        selected_editable_objects=objects,
        editable_objects=objects,
        visible_objects=objects,
        active_object=active_object,
        object=active_object,
    )


def do_auto_smooth(object: bpy.types.Object, angle=30):
    """
    Applies auto-smooth to the object in Blender using either the new geometry nodes or fallback to modifiers.

    Args:
        object (bpy.types.Object): The object to smooth.
    """
    try:
        object.data.use_auto_smooth = True
    except AttributeError:
        with bpy.context.temp_override(**get_object_override(object)):
            result = bpy.ops.object.modifier_add_node_group(
                asset_library_type="ESSENTIALS",
                asset_library_identifier="",
                relative_asset_identifier="geometry_nodes\\smooth_by_angle.blend\\NodeTree\\Smooth by Angle",
            )
            if "CANCELLED" in result:
                return

            modifier = object.modifiers[-1]
            modifier["Socket_1"] = True
            modifier["Input_1"] = math.radians(angle)
            object.update_tag()


def get_or_generate_material(name, color):
    """Get existing or create a new material with the specified color."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color  # For Blender 2.8 and above
        return mat


def get_or_generate_voxel_material(voxel_mat_name="VoxelMaterial"):
    if voxel_mat_name in bpy.data.materials:
        voxel_mat = bpy.data.materials[voxel_mat_name]
    else:
        voxel_mat = bpy.data.materials.new(name=voxel_mat_name)

        # Enable 'Use Nodes' for the material
        voxel_mat.use_nodes = True
        nodes = voxel_mat.node_tree.nodes
        links = voxel_mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        vertex_color_node = nodes.new(type="ShaderNodeVertexColor")

        vertex_color_node.layer_name = "Color"

        # Link nodes
        links.new(
            vertex_color_node.outputs["Color"], principled_node.inputs["Base Color"]
        )
        links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
    return voxel_mat


def get_or_generate_shell_material(shell_mat_name="ShellMaterial"):
    if shell_mat_name in bpy.data.materials:
        shell_mat = bpy.data.materials[shell_mat_name]
    else:
        shell_mat = bpy.data.materials.new(name=shell_mat_name)

        # Enable 'Use Nodes' for the shell material
        shell_mat.use_nodes = True
        nodes = shell_mat.node_tree.nodes
        links = shell_mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes for shell material
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        vertex_color_node = nodes.new(type="ShaderNodeVertexColor")

        vertex_color_node.layer_name = "Color"

        # Link nodes
        links.new(
            vertex_color_node.outputs["Color"], principled_node.inputs["Base Color"]
        )
        links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
    return shell_mat


def get_or_generate_bone_material(bones_mat_name="BoneMaterial"):
    bones_mat_name = "BoneMaterial"
    if bones_mat_name in bpy.data.materials:
        bones_mat = bpy.data.materials[bones_mat_name]
    else:
        bones_mat = bpy.data.materials.new(name=bones_mat_name)

        # Enable 'Use Nodes' for the bones material
        bones_mat.use_nodes = True
        nodes = bones_mat.node_tree.nodes
        links = bones_mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes for bones material
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")

        # Set base color to black
        principled_node.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)

        # Link nodes
        links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
    return bones_mat


def generate_cylinder_between(
    collection,
    cylinder_name: str,
    point1: tuple,
    point2: tuple,
    radius: float,
    material,
):
    """Create a cylinder between two points with specified radius and material."""
    vec = Vector(point2) - Vector(point1)
    length = vec.length

    # Create cylinder at origin pointing up
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=(0, 0, 0))
    cylinder = bpy.context.active_object
    cylinder.name = cylinder_name

    # Store the current cursor location
    cursor_location = bpy.context.scene.cursor.location.copy()

    # Set cylinder origin to the bottom end
    bpy.context.scene.cursor.location = Vector((0, 0, -length / 2))
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    # Restore the cursor location
    bpy.context.scene.cursor.location = cursor_location

    # Move cylinder to point1
    cylinder.location = point1

    # Rotate cylinder to align with vector
    vec.normalize()
    up = Vector((0, 0, 1))
    axis = up.cross(vec)
    angle = up.angle(vec)
    if axis.length == 0:
        # Vectors are parallel
        if vec.dot(up) > 0:
            rot_quat = Quaternion()
        else:
            rot_quat = Quaternion((0, 0, 1), np.pi)
    else:
        axis.normalize()
        rot_quat = Quaternion(axis, angle)
    cylinder.rotation_mode = "QUATERNION"
    cylinder.rotation_quaternion = rot_quat

    # Remove from current collection(s)
    for coll in cylinder.users_collection:
        coll.objects.unlink(cylinder)
    # Add to our collection
    collection.objects.link(cylinder)

    if material is not None:
        # Assign material
        if cylinder.data.materials:
            cylinder.data.materials[0] = material
        else:
            cylinder.data.materials.append(material)

    return cylinder


def generate_arrow_between(
    collection,
    arrow_name: str,
    start_point: tuple,
    end_point: tuple,
    shaft_radius: float,
    cone_length: float,
    cone_radius: float,
    material=None,
):
    """Create an arrow (cylinder shaft and cone tip) between two points with specified radius and material."""
    # Calculate the vector between the points
    vec = Vector(end_point) - Vector(start_point)
    length = vec.length
    shaft_length = length

    # Create shaft (cylinder) at origin pointing up
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius, depth=shaft_length, location=(0, 0, 0)
    )
    shaft = bpy.context.active_object
    shaft.name = f"{arrow_name}_shaft"

    # Store the current cursor location
    cursor_location = bpy.context.scene.cursor.location.copy()

    # Set shaft origin to the bottom end
    bpy.context.scene.cursor.location = Vector((0, 0, -shaft_length / 2))
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    # Restore the cursor location
    bpy.context.scene.cursor.location = cursor_location

    # Move shaft to point1
    shaft.location = start_point

    # Rotate shaft to align with vector
    vec.normalize()
    up = Vector((0, 0, 1))
    axis = up.cross(vec)
    angle = up.angle(vec)
    if axis.length == 0:
        # Vectors are parallel
        if vec.dot(up) > 0:
            rot_quat = Quaternion()
        else:
            rot_quat = Quaternion((1, 0, 0), np.pi)
    else:
        axis.normalize()
        rot_quat = Quaternion(axis, angle)
    shaft.rotation_mode = "QUATERNION"
    shaft.rotation_quaternion = rot_quat

    # Create cone (arrow tip) at the top of the shaft
    bpy.ops.mesh.primitive_cone_add(
        radius1=cone_radius,
        depth=cone_length,
        location=(0, 0, 0),
    )
    cone = bpy.context.active_object
    cone.name = f"{arrow_name}_cone"

    # Set cone origin to the bottom (pointy end)
    bpy.context.scene.cursor.location = Vector((0, 0, -cone_length / 2))
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    # Restore the cursor location again
    bpy.context.scene.cursor.location = cursor_location

    # Move cone to the top of the shaft
    cone.location = Vector(start_point) + vec * shaft_length

    # Rotate cone to align with the same vector
    cone.rotation_mode = "QUATERNION"
    cone.rotation_quaternion = rot_quat

    # Remove shaft and cone from current collection(s)
    for coll in shaft.users_collection:
        coll.objects.unlink(shaft)
    for coll in cone.users_collection:
        coll.objects.unlink(cone)

    # Add both objects (shaft and cone) to the target collection
    collection.objects.link(shaft)
    collection.objects.link(cone)

    # Assign material if provided
    if material is not None:
        if shaft.data.materials:
            shaft.data.materials[0] = material
        else:
            shaft.data.materials.append(material)

        if cone.data.materials:
            cone.data.materials[0] = material
        else:
            cone.data.materials.append(material)

    return shaft, cone


def generate_trace_curve(collection, com_positions, arrow_interval=10):
    """
    Create a curve in Blender passing through the given COM positions and add arrows along the curve.

    Args:
        collection: The Blender collection to which the curve and arrows should be added.
        com_positions: List of (x, y, z) tuples or numpy arrays.
        arrow_interval: Interval between arrows along the curve (number of frames between arrows).
    """
    # Create a new curve data block
    curve_data = bpy.data.curves.new(name="COM_Curve", type="CURVE")
    curve_data.dimensions = "3D"

    # Create a new polyline in the curve
    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(com_positions) - 1)  # Add points (already has 1 point)

    for i, com in enumerate(com_positions):
        x, y, z = com
        polyline.points[i].co = (x, y, z, 1)  # The fourth value is the weight (w)

    # Create a new curve object
    curve_obj = bpy.data.objects.new("COM_Curve_Object", curve_data)

    collection.objects.link(curve_obj)

    # Optionally, adjust curve properties (e.g., bevel depth for thickness)
    curve_data.bevel_depth = 0.005  # Adjust the thickness as needed

    # Set a material (make the curve black)
    mat = get_or_generate_material(name="COM_Curve_Material", color=(0, 0, 0, 1))
    curve_obj.data.materials.append(mat)

    # Generate arrows along the curve
    # Get the spline from the curve
    spline = curve_obj.data.splines[0]
    points = spline.points

    # Ensure we have enough points
    num_points = len(points)
    if num_points < 2:
        print("Not enough points in the spline to generate arrows.")
        return

    # Create arrows along the curve at specified intervals
    for i in range(0, num_points - 1, arrow_interval):
        # Get the start and end points of the segment
        p0 = mathutils.Vector(points[i].co[:3])
        p1 = mathutils.Vector(points[i + 1].co[:3])

        # Calculate the position for the arrow (midpoint of the segment)
        position = (p0 + p1) / 2.0

        # Calculate the tangent (direction of the segment)
        tangent_vec = (p0 - p1).normalized()

        # Create an arrow object
        # Create a new mesh data block
        arrow_mesh = bpy.data.meshes.new(f"ArrowMesh_{i}")

        # Create a bmesh to build the mesh
        bm = bmesh.new()

        # Create the arrowhead (cone)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            radius1=0.03,  # Base radius
            radius2=0.0,  # Tip radius
            depth=0.1,  # Height of the cone
            segments=32,
        )

        # Move the cone so that its base is at the origin and points along the +Z axis
        bmesh.ops.translate(
            bm,
            verts=bm.verts,
            vec=mathutils.Vector((0, 0, 0.05)),
        )

        # Finish up
        bm.to_mesh(arrow_mesh)
        bm.free()

        arrow_obj = bpy.data.objects.new(f"Arrow_{i}", arrow_mesh)

        # Position the arrow at the position
        arrow_obj.location = position

        # Orient the arrow along the tangent
        rotation = tangent_vec.to_track_quat("-Z", "Y").to_euler()
        arrow_obj.rotation_euler = rotation

        # Add the arrow to the collection
        collection.objects.link(arrow_obj)

        # Set the arrow's scale (optional)
        # arrow_obj.scale = (0.05, 0.05, 0.05)  # Adjust the size as needed

        # Set the arrow's material (black)
        if "Arrow_Material" in bpy.data.materials:
            arrow_mat = bpy.data.materials["Arrow_Material"]
        else:
            arrow_mat = bpy.data.materials.new(name="Arrow_Material")
            arrow_mat.diffuse_color = (0, 0, 0, 1)  # Black color
        arrow_obj.data.materials.append(arrow_mat)


def generate_part(
    collection,
    part_name: str,
    positions: np.ndarray,
    orientations: np.ndarray,
    ppp_offsets: np.ndarray,
    nnn_offsets: np.ndarray,
    connectivity: np.ndarray,
    colors: np.ndarray = None,
    remove_connected_faces: bool = True,
    smooth: bool = False,
    voxel_size: float = 0.01,
):
    if not remove_connected_faces and smooth:
        raise ValueError("smooth cannot be True when remove_connected_faces is False")
    # Create a new mesh and object for the part
    part_mesh = bpy.data.meshes.new(f"{part_name}_Mesh")
    part_object = bpy.data.objects.new(f"{part_name}", part_mesh)
    collection.objects.link(part_object)

    # Create a BMesh for the part
    bm_part = bmesh.new()

    # Add a vertex color layer for the part
    color_layer_part = bm_part.loops.layers.color.new("Color")

    # Direction to bit mapping
    direction_bits = {
        "+X": 0,
        "-X": 1,
        "+Y": 2,
        "-Y": 3,
        "+Z": 4,
        "-Z": 5,
    }

    # Create a dictionary to store the mapping from grid positions to vertices
    vertex_dict = {}

    # Create a set to keep track of created faces
    face_set = set()

    # Helper function to create or get a vertex at a given position
    resolution = voxel_size / 100

    def get_vertex(pos):
        key = tuple((p // resolution) * resolution for p in pos)
        if key in vertex_dict:
            return vertex_dict[key]
        else:
            index = len(bm_part.verts)
            v = bm_part.verts.new(pos)
            v.index = index
            vertex_dict[key] = v
            return v

    # Build the mesh by creating faces between shared vertices
    for i in range(len(positions)):
        pos = positions[i]
        orientation_quat = orientations[i]  # x, y, z, w

        # Compute the scale vector (size along each axis)
        ppp_offset = ppp_offsets[i]
        nnn_offset = nnn_offsets[i]
        center_offset = (ppp_offset + nnn_offset) / 2
        scale_vector = np.abs(ppp_offset - nnn_offset)
        scale_vector = mathutils.Vector(scale_vector)

        # Convert the quaternion from numpy array to mathutils.Quaternion
        quat = mathutils.Quaternion(
            (
                orientation_quat[3],
                orientation_quat[0],
                orientation_quat[1],
                orientation_quat[2],
            )
        )
        rotation_matrix = quat.to_matrix().to_4x4()

        # Create scaling matrix with different scales along each axis
        scale_matrix = mathutils.Matrix.Diagonal(scale_vector).to_4x4()

        # Combine rotation and scaling
        transform_matrix = rotation_matrix @ scale_matrix

        # Apply translation to the transformation matrix
        transform_matrix.translation = mathutils.Vector(pos + center_offset)

        # Compute the eight corner positions of the voxel
        corners = []
        for dx in [-0.5, 0.5]:
            for dy in [-0.5, 0.5]:
                for dz in [-0.5, 0.5]:
                    corners.append((dx, dy, dz))

        # Create or get vertices at the corner positions
        verts = [
            get_vertex(transform_matrix @ mathutils.Vector(corner))
            for corner in corners
        ]

        # Define the faces of the cube (voxel)
        face_indices = [
            (0, 1, 3, 2),  # -X face
            (4, 6, 7, 5),  # +X face
            (0, 4, 5, 1),  # -Y face
            (2, 3, 7, 6),  # +Y face
            (0, 2, 6, 4),  # -Z face
            (1, 5, 7, 3),  # +Z face
        ]

        # Map faces to directions
        direction_faces = {
            "-X": face_indices[0],
            "+X": face_indices[1],
            "-Y": face_indices[2],
            "+Y": face_indices[3],
            "-Z": face_indices[4],
            "+Z": face_indices[5],
        }

        # Get the connectivity of the current voxel
        conn = connectivity[i]  # Should be an integer

        # For each direction, check if the voxel is NOT connected, and create the face
        for direction, face_idx in direction_faces.items():
            bit = direction_bits[direction]
            if not remove_connected_faces or not (conn & (1 << bit)):
                # The voxel is not connected in this direction, create the face
                face_verts = [verts[idx] for idx in face_idx]
                # Create a key for the face to check for duplicates
                face_key = tuple(sorted([v.index for v in face_verts]))
                if face_key not in face_set:
                    try:
                        face = bm_part.faces.new(face_verts)
                        for loop in face.loops:
                            loop[color_layer_part] = (
                                colors[i]
                                if colors is not None
                                else (1.0, 1.0, 1.0, 1.0)
                            )
                    except ValueError:
                        # Face already exists
                        pass
                    face_set.add(face_key)

    # Finish up, write the bmesh into the mesh
    bm_part.to_mesh(part_mesh)
    bm_part.free()

    if smooth:
        # Duplicate the original object before remeshing
        original_part_object = part_object.copy()
        original_part_object.data = part_object.data.copy()
        collection.objects.link(original_part_object)

        # Apply Remesh modifier to smooth out the mesh
        voxel_size = 0.02  # Smaller values result in higher resolution meshes

        remesh_modifier = part_object.modifiers.new(name="Remesh", type="REMESH")
        remesh_modifier.mode = "VOXEL"
        remesh_modifier.voxel_size = voxel_size
        remesh_modifier.use_smooth_shade = True

        # Apply the Remesh modifier
        bpy.context.view_layer.objects.active = part_object
        bpy.ops.object.modifier_apply(modifier=remesh_modifier.name)

        # Remesh will remove the data layer, so we need to create a new one, then do data transfer
        bpy.context.view_layer.objects.active = part_object
        bpy.ops.geometry.color_attribute_add(
            name="Color", domain="CORNER", data_type="BYTE_COLOR"
        )

        # Add Data Transfer modifier to transfer vertex colors
        data_transfer_modifier = part_object.modifiers.new(
            name="DataTransfer", type="DATA_TRANSFER"
        )
        data_transfer_modifier.object = original_part_object
        data_transfer_modifier.use_loop_data = True
        data_transfer_modifier.data_types_loops = {"COLOR_CORNER"}
        data_transfer_modifier.loop_mapping = (
            "NEAREST_POLYNOR"  # You can choose other mappings as needed
        )
        data_transfer_modifier.layers_vcol_loop_select_src = "Color"

        # Apply the Data Transfer modifier
        bpy.ops.object.modifier_apply(modifier=data_transfer_modifier.name)

        # Remove the original object if not needed
        bpy.data.objects.remove(original_part_object, do_unlink=True)

    return part_object


def generate_ground(
    collection,
    plane_size: float = 10,
    plane_mat_name: str = "GroundMaterial",
):
    # Create or reuse ground material
    plane_mat = get_or_generate_material(plane_mat_name, (0.5, 0.5, 0.5, 1.0))

    # Create a rectangular plane under the voxels
    bpy.ops.mesh.primitive_plane_add(
        size=plane_size,
        location=(0, 0, 0),
    )
    plane = bpy.context.object
    plane.name = "GroundPlane"

    plane.data.materials.append(plane_mat)
    collection.objects.link(plane)
    bpy.context.scene.collection.objects.unlink(plane)


def generate_elevated_ground(
    collection,
    height: np.ndarray,
    x_size: float,
    y_size: float,
    subdivide=1,
    uv_scale=1,
    display_region: Tuple[float, float, float, float] = None,
    plane_mat_name: str = "ElevatedGroundMaterial",
):
    """
    Generate an elevated ground mesh from a height map.

    Args:
        collection (bpy.types.Collection): Blender collection to which the elevated ground mesh will be added.
        height (np.ndarray): 2D array of height values, y dimension first.
        x_size (float): X size of the elevated ground mesh.
        y_size (float): Y size of the elevated ground mesh.
        subdivide (int, optional): Factor to increase mesh resolution. Defaults to 1.
        uv_scale (float, optional): Real-world distance in meters that maps to a UV range of 1. Defaults to 1.
        display_region (tuple of float, optional): The region to display (min_x, max_x, min_y, max_y).
            Only vertices and faces within this region will be displayed. Defaults to None (no truncation).
        plane_mat_name (str, optional): Name of the material for the elevated ground plane. Defaults to "ElevatedGroundMaterial".
    """
    original_x_values = x_values = height.shape[1]
    original_y_values = y_values = height.shape[0]

    # Apply subdivision
    x_values *= subdivide
    y_values *= subdivide

    # Generate x and y coordinates
    x_coords = np.linspace(-x_size / 2.0, x_size / 2.0, x_values)
    y_coords = np.linspace(-y_size / 2.0, y_size / 2.0, y_values)

    # Define the original grid coordinates
    original_x = np.linspace(-x_size / 2.0, x_size / 2.0, original_x_values)
    original_y = np.linspace(-y_size / 2.0, y_size / 2.0, original_y_values)

    # Generate new height grid using bilinear interpolation
    X, Y = np.meshgrid(x_coords, y_coords)
    height_array = np.zeros_like(X.flatten())

    for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
        # Find the bounding indices
        x1_idx = np.searchsorted(original_x, x) - 1
        x2_idx = x1_idx + 1
        y1_idx = np.searchsorted(original_y, y) - 1
        y2_idx = y1_idx + 1

        # Handle edge cases where indices might go out of bounds or are degenerate
        if x1_idx < 0:
            x1_idx = 0
        if x2_idx >= original_x_values:
            x2_idx = original_x_values - 1
        if y1_idx < 0:
            y1_idx = 0
        if y2_idx >= original_y_values:
            y2_idx = original_y_values - 1

        # Bounding x and y values
        x1, x2 = original_x[x1_idx], original_x[x2_idx]
        y1, y2 = original_y[y1_idx], original_y[y2_idx]

        # Corresponding height values at bounding points
        Q11 = height[y1_idx, x1_idx]
        Q12 = height[y2_idx, x1_idx]
        Q21 = height[y1_idx, x2_idx]
        Q22 = height[y2_idx, x2_idx]

        # Perform bilinear interpolation
        height_array[i] = bilinear_interpolation(
            x, y, x1, x2, y1, y2, Q11, Q12, Q21, Q22
        )

    # Create mesh grid
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = height_array.flatten()

    # Apply truncation based on display_region
    if display_region:
        min_x, max_x, min_y, max_y = display_region
        # Filter the vertices based on the region
        mask = (
            (X_flat >= min_x)
            & (X_flat <= max_x)
            & (Y_flat >= min_y)
            & (Y_flat <= max_y)
        )
        X_flat = X_flat[mask]
        Y_flat = Y_flat[mask]
        Z_flat = Z_flat[mask]
        x_values = len(np.unique(X_flat))
        y_values = len(np.unique(Y_flat))

    # Create list of vertices
    vertices = list(zip(X_flat, Y_flat, Z_flat))

    # Create faces (quads)
    faces = []
    for j in range(y_values - 1):
        for i in range(x_values - 1):
            idx = j * x_values + i
            face = (idx, idx + 1, idx + x_values + 1, idx + x_values)
            faces.append(face)

    # Create mesh and object
    mesh = bpy.data.meshes.new("GroundMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)

    # Compute normals correctly (use do_auto_smooth)
    obj = bpy.data.objects.new("Floor", mesh)
    do_auto_smooth(obj)

    # Create UV map
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    # Calculate UVs
    for face_idx, face in enumerate(mesh.polygons):
        for loop_idx in face.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            x = mesh.vertices[vert_idx].co.x
            y = mesh.vertices[vert_idx].co.y
            uv = (x / uv_scale % 1, y / uv_scale % 1)
            uv_layer[loop_idx].uv = uv

    # Assign material
    material = get_or_generate_material(plane_mat_name, (0.5, 0.5, 0.5, 1.0))
    if len(mesh.materials):
        mesh.materials[0] = material
    else:
        mesh.materials.append(material)

    # Link object to collection
    collection.objects.link(obj)

    return obj


def generate_floor(
    collection,
    reader: RiseSimulationDataReader,
    subdivide=1,
    uv_scale=1,
    display_region: Tuple[float, float, float, float] = None,
    floor_mat_name: str = "FloorMaterial",
):
    """
    Generates the floor mesh in Blender based on the floor elevation configuration,
    with options to subdivide the mesh and control UV mapping scale.

    Args:
        collection (bpy.types.Collection): Blender collection to which the floor mesh will be added.
        reader (RiseSimulationDataReader): RiseSimulationDataReader instance.
        subdivide (int, optional): Factor to increase mesh resolution. Defaults to 1.
        uv_scale (float, optional): Real-world distance in meters that maps to a UV range of 1. Defaults to 1.
        display_region (tuple of float, optional): The region to display (min_x, max_x, min_y, max_y).
            Only vertices and faces within this region will be displayed. Defaults to None (no truncation).

    Returns:
        bpy.types.Object: The created floor mesh object.
    """
    # Parse the configuration
    config = reader.read_config()
    floor_config = config.get("floor_elevation_config", {})

    h_min = floor_config.get("h_min", 0.0)
    h_max = floor_config.get("h_max", 0.0)
    height_array = floor_config.get("height", [])
    x_size = floor_config.get("x_size", 1.0)
    y_size = floor_config.get("y_size", 1.0)
    x_values = floor_config.get("x_values", 2)
    y_values = floor_config.get("y_values", 2)

    # Apply subdivision
    x_values *= subdivide
    y_values *= subdivide

    # Generate x and y coordinates
    x_coords = np.linspace(-x_size / 2.0, x_size / 2.0, x_values)
    y_coords = np.linspace(-y_size / 2.0, y_size / 2.0, y_values)

    if not height_array:
        return

    original_height_array = np.array(height_array, dtype=np.uint16)
    original_x_values = floor_config.get("x_values", 2)
    original_y_values = floor_config.get("y_values", 2)

    # Reshape to 2D grid
    height_grid = original_height_array.reshape((original_y_values, original_x_values))

    # Define the original grid coordinates
    original_x = np.linspace(-x_size / 2.0, x_size / 2.0, original_x_values)
    original_y = np.linspace(-y_size / 2.0, y_size / 2.0, original_y_values)

    # Generate new height grid using bilinear interpolation
    X, Y = np.meshgrid(x_coords, y_coords)
    height_array = np.zeros_like(X.flatten())

    for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
        # Find the bounding indices
        x1_idx = np.searchsorted(original_x, x) - 1
        x2_idx = x1_idx + 1
        y1_idx = np.searchsorted(original_y, y) - 1
        y2_idx = y1_idx + 1

        # Handle edge cases where indices might go out of bounds or are degenerate
        if x1_idx < 0:
            x1_idx = 0
        if x2_idx >= original_x_values:
            x2_idx = original_x_values - 1
        if y1_idx < 0:
            y1_idx = 0
        if y2_idx >= original_y_values:
            y2_idx = original_y_values - 1

        # Bounding x and y values
        x1, x2 = original_x[x1_idx], original_x[x2_idx]
        y1, y2 = original_y[y1_idx], original_y[y2_idx]

        # Corresponding height values at bounding points
        Q11 = height_grid[y1_idx, x1_idx]
        Q12 = height_grid[y2_idx, x1_idx]
        Q21 = height_grid[y1_idx, x2_idx]
        Q22 = height_grid[y2_idx, x2_idx]

        # Perform bilinear interpolation
        height_array[i] = bilinear_interpolation(
            x, y, x1, x2, y1, y2, Q11, Q12, Q21, Q22
        )

    # Scale the height array to actual heights in meters
    heights_in_meters = h_min + (height_array / 65536.0) * (h_max - h_min)

    # Create mesh grid
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = heights_in_meters.flatten()

    # Apply truncation based on display_region
    if display_region:
        min_x, max_x, min_y, max_y = display_region
        # Filter the vertices based on the region
        mask = (
            (X_flat >= min_x)
            & (X_flat <= max_x)
            & (Y_flat >= min_y)
            & (Y_flat <= max_y)
        )
        X_flat = X_flat[mask]
        Y_flat = Y_flat[mask]
        Z_flat = Z_flat[mask]
        x_values = len(np.unique(X_flat))
        y_values = len(np.unique(Y_flat))

    # Create list of vertices
    vertices = list(zip(X_flat, Y_flat, Z_flat))

    # Create faces (quads)
    faces = []
    for j in range(y_values - 1):
        for i in range(x_values - 1):
            idx = j * x_values + i
            face = (idx, idx + 1, idx + x_values + 1, idx + x_values)
            faces.append(face)

    # Create mesh and object
    mesh = bpy.data.meshes.new("FloorMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)

    # Compute normals correctly (use do_auto_smooth)
    obj = bpy.data.objects.new("Floor", mesh)
    do_auto_smooth(obj)

    # Create UV map
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    # Calculate UVs
    for face_idx, face in enumerate(mesh.polygons):
        for loop_idx in face.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            x = mesh.vertices[vert_idx].co.x
            y = mesh.vertices[vert_idx].co.y
            uv = (x / uv_scale % 1, y / uv_scale % 1)
            uv_layer[loop_idx].uv = uv

    # Assign material
    material = get_or_generate_material(floor_mat_name, (0.5, 0.5, 0.5, 1.0))
    if len(mesh.materials):
        mesh.materials[0] = material
    else:
        mesh.materials.append(material)

    # Link object to collection
    collection.objects.link(obj)

    return obj


def generate_robot(
    collection,
    reader: RiseSimulationDataReader,
    frame: int,
    color_domain: str = "poissons_strain",
    color_map: Callable[[np.ndarray], np.ndarray] = None,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    bodies: List[int] = None,
    mode: str = "voxels",
    opacity: float = 1.0,
    voxel_size: float = 0.01,
):
    if mode not in ("voxels", "transparent", "simple_transparent"):
        raise ValueError(
            "mode must be either voxels or transparent or simple_transparent"
        )

    # Read data for the specified frame
    frame_data = reader.read_frame(frame, bodies=bodies)
    voxel_data = frame_data["voxels"]
    is_rigid = voxel_data["is_rigid"]
    positions = voxel_data["position"]  # Shape: (num_voxels, 3)
    orientations = voxel_data["orientation"]  # Shape: (num_voxels, 4)

    if transform is not None:
        positions, orientations = transform(positions, orientations)

    poissons_strain = voxel_data["poissons_strain"]  # Shape: (num_voxels, 3)
    linear_velocity = voxel_data["linear_velocity"]  # Shape: (num_voxels, 3)
    angular_velocity = voxel_data["angular_velocity"]  # Shape: (num_voxels, 3)
    ppp_offsets = voxel_data["ppp_offset"]  # Shape: (num_voxels, 3)
    nnn_offsets = voxel_data["nnn_offset"]  # Shape: (num_voxels, 3)
    connectivity = voxel_data["connectivity"]  # Shape: (num_voxels,)

    if color_domain == "poissons_strain":
        color_scale = np.clip(np.abs(poissons_strain).max(axis=1) * 10, 0, 1)
    elif color_domain == "linear_velocity":
        color_scale = np.clip(np.abs(linear_velocity).max(axis=1), 0, 1)
    elif color_domain == "angular_velocity":
        color_scale = np.clip(np.abs(angular_velocity).max(axis=1), 0, 1)
    elif color_domain == "none":
        pass
    else:
        raise ValueError("Invalid color domain")

    non_rigid_mask = is_rigid == False
    outside_colors = np.zeros([np.sum(non_rigid_mask), 4], dtype=np.float32)
    if color_domain != "none":
        if color_map is None:
            for idx, strain_value in enumerate(color_scale[non_rigid_mask]):
                color = colorsys.hsv_to_rgb((1.0 - strain_value) * 0.66, 1.0, 1.0)
                outside_colors[idx] = (*color, opacity)
        else:
            outside_colors[:, :3] = color_map(color_scale[non_rigid_mask])
            outside_colors[:, 3] = opacity
    else:
        # blue
        outside_colors[:, 2:] = 1

    if mode == "voxels":
        voxel_object = generate_part(
            collection,
            "Voxel",
            positions[non_rigid_mask],
            orientations[non_rigid_mask],
            ppp_offsets[non_rigid_mask],
            nnn_offsets[non_rigid_mask],
            connectivity[non_rigid_mask],
            colors=outside_colors,
            smooth=False,
            remove_connected_faces=False,
            voxel_size=voxel_size,
        )

        rigid_voxel_object = generate_part(
            collection,
            "RigidVoxel",
            positions[~non_rigid_mask],
            orientations[~non_rigid_mask],
            ppp_offsets[~non_rigid_mask],
            nnn_offsets[~non_rigid_mask],
            connectivity[~non_rigid_mask],
            colors=np.ones([np.sum(~non_rigid_mask), 4], dtype=np.float32)
            * np.array([[0.808, 0.807, 0.671, 1]]),
            smooth=False,
            remove_connected_faces=False,
            voxel_size=voxel_size,
        )

        # Assign the material to the voxel object
        voxel_object.data.materials.append(get_or_generate_voxel_material())
        rigid_voxel_object.data.materials.append(get_or_generate_voxel_material())
    else:
        shell_object = generate_part(
            collection,
            "Shell",
            positions[non_rigid_mask],
            orientations[non_rigid_mask],
            ppp_offsets[non_rigid_mask],
            nnn_offsets[non_rigid_mask],
            connectivity[non_rigid_mask],
            colors=outside_colors,
            smooth=True,
            voxel_size=voxel_size,
        )
        shell_object.data.materials.append(get_or_generate_shell_material())

        if mode == "transparent":
            shell_object2 = generate_part(
                collection,
                "Shell2",
                positions[non_rigid_mask],
                orientations[non_rigid_mask],
                ppp_offsets[non_rigid_mask],
                nnn_offsets[non_rigid_mask],
                connectivity[non_rigid_mask],
                colors=outside_colors,
                smooth=False,
                voxel_size=voxel_size,
            )
            shell_object2.data.materials.append(get_or_generate_shell_material())

        body_segment_ids = voxel_data["body_segment_id"]
        unique_ids = np.unique(body_segment_ids)
        for idx, bsid in enumerate(unique_ids):
            if bsid <= 1:
                continue
            mask = body_segment_ids == bsid
            bone_colors = np.ones([np.sum(mask), 4], dtype=np.float32)
            bone_colors[:, 3] = opacity
            bone_object = generate_part(
                collection,
                f"Bone_{idx}",
                positions[mask],
                orientations[mask],
                ppp_offsets[mask],
                nnn_offsets[mask],
                connectivity[mask],  # TODO: use correct connectivity
                colors=bone_colors,
                smooth=True,
                voxel_size=voxel_size,
            )
            bone_object.data.materials.append(get_or_generate_bone_material())


def generate_robot_abstract(
    collection,
    reader: RiseSimulationDataReader,
    frame: int,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    opacity: float = 1.0,
    voxel_size: float = 0.01,
    draw_shell: bool = True,
    draw_bone: bool = True,
    draw_node: bool = True,
    draw_edge: bool = True,
):
    # Read data for the specified frame
    frame_data = reader.read_frame(frame)
    voxel_data = frame_data["voxels"]
    is_rigid = voxel_data["is_rigid"]
    positions = voxel_data["position"]  # Shape: (num_voxels, 3)
    orientations = voxel_data["orientation"]  # Shape: (num_voxels, 4)

    if transform is not None:
        positions, orientations = transform(positions, orientations)

    ppp_offsets = voxel_data["ppp_offset"]  # Shape: (num_voxels, 3)
    nnn_offsets = voxel_data["nnn_offset"]  # Shape: (num_voxels, 3)
    connectivity = voxel_data["connectivity"]  # Shape: (num_voxels,)

    non_rigid_mask = is_rigid == False
    outside_colors = np.zeros([np.sum(non_rigid_mask), 4], dtype=np.float32)
    outside_colors[:, 3] = opacity

    if draw_shell:
        shell_object = generate_part(
            collection,
            "Shell",
            positions[non_rigid_mask],
            orientations[non_rigid_mask],
            ppp_offsets[non_rigid_mask],
            nnn_offsets[non_rigid_mask],
            connectivity[non_rigid_mask],
            colors=outside_colors,
            smooth=True,
            voxel_size=voxel_size,
        )

        # Assign the material to the shell object
        shell_object.data.materials.append(get_or_generate_shell_material())

    if draw_bone:
        body_segment_ids = voxel_data["body_segment_id"]
        unique_ids = np.unique(body_segment_ids)
        for idx, bsid in enumerate(unique_ids):
            if bsid <= 1:
                continue
            mask = body_segment_ids == bsid
            bone_colors = np.zeros([np.sum(mask), 4], dtype=np.float32)
            bone_colors[:, 3] = opacity
            bone_object = generate_part(
                collection,
                f"Bone_{idx}",
                positions[mask],
                orientations[mask],
                ppp_offsets[mask],
                nnn_offsets[mask],
                connectivity[mask],  # TODO: use correct connectivity
                colors=bone_colors,
                smooth=True,
                voxel_size=voxel_size,
            )
            # bone object also transparent
            bone_object.data.materials.append(get_or_generate_shell_material())

    # Read rigid body and joint data
    rigid_body_data = frame_data["rigid_bodies"]
    joint_data = frame_data["joints"]

    # Extract COM positions and orientations
    com_positions = rigid_body_data["com"]  # Shape: (num_rigid_bodies, 3)
    rb_orientations = rigid_body_data["orientation"]  # Shape: (num_rigid_bodies, 4)

    # Apply transform if provided
    if transform is not None:
        com_positions, rb_orientations = transform(com_positions, rb_orientations)

    # Create a mapping from rigid body index to array index
    rigid_body_indices = rigid_body_data["index"]  # Shape: (num_rigid_bodies,)
    index_to_array_idx = {rb_index: i for i, rb_index in enumerate(rigid_body_indices)}

    # Get materials
    red_material = get_or_generate_material("RedMaterial", (1.0, 0.0, 0.0, 1.0))  # RGBA
    blue_material = get_or_generate_material("BlueMaterial", (0.0, 0.0, 1.0, 1.0))

    # Define sizes
    ball_radius = voxel_size * 2
    cylinder_radius = voxel_size * 0.5

    if draw_node:
        # Create red spheres at COM positions
        for i, com in enumerate(com_positions):
            # Convert numpy array to tuple
            com_tuple = tuple(com.tolist())
            # Create a sphere at the COM position
            bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_radius, location=com_tuple)
            sphere = bpy.context.active_object
            sphere.name = f"RigidBody_{i}_COM"
            # Remove sphere from current collection(s)
            for coll in sphere.users_collection:
                coll.objects.unlink(sphere)
            # Add sphere to our collection
            collection.objects.link(sphere)
            sphere.data.materials.append(red_material)

    if draw_edge:
        # Create blue cylinders between connected rigid bodies
        rigid_body_a_indices = joint_data["rigid_body_a"]
        rigid_body_b_indices = joint_data["rigid_body_b"]

        for i in range(len(rigid_body_a_indices)):
            rb_a_index = rigid_body_a_indices[i]
            rb_b_index = rigid_body_b_indices[i]
            # Map to array indices
            idx_a = index_to_array_idx.get(rb_a_index)
            idx_b = index_to_array_idx.get(rb_b_index)
            if idx_a is None or idx_b is None:
                continue  # Rigid body index not found, skip
            com_a = com_positions[idx_a]
            com_b = com_positions[idx_b]
            # Convert numpy arrays to tuples
            com_a_tuple = tuple(com_a.tolist())
            com_b_tuple = tuple(com_b.tolist())
            # Create cylinder between COMs
            generate_cylinder_between(
                collection,
                f"Joint_{i}",
                com_a_tuple,
                com_b_tuple,
                radius=cylinder_radius,
                material=blue_material,
            )


def generate_robot_abstract_with_joint(
    collection,
    reader: RiseSimulationDataReader,
    frame: int,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    opacity: float = 1.0,
    voxel_size: float = 0.01,
):
    # Read data for the specified frame
    frame_data = reader.read_frame(frame)
    voxel_data = frame_data["voxels"]
    is_rigid = voxel_data["is_rigid"]
    positions = voxel_data["position"]  # Shape: (num_voxels, 3)
    orientations = voxel_data["orientation"]  # Shape: (num_voxels, 4)

    if transform is not None:
        positions, orientations = transform(positions, orientations)

    ppp_offsets = voxel_data["ppp_offset"]  # Shape: (num_voxels, 3)
    nnn_offsets = voxel_data["nnn_offset"]  # Shape: (num_voxels, 3)
    connectivity = voxel_data["connectivity"]  # Shape: (num_voxels,)

    non_rigid_mask = is_rigid == False
    outside_colors = np.zeros([np.sum(non_rigid_mask), 4], dtype=np.float32)
    outside_colors[:, 3] = opacity

    shell_object = generate_part(
        collection,
        "Shell",
        positions[non_rigid_mask],
        orientations[non_rigid_mask],
        ppp_offsets[non_rigid_mask],
        nnn_offsets[non_rigid_mask],
        connectivity[non_rigid_mask],
        colors=outside_colors,
        smooth=True,
        voxel_size=voxel_size,
    )

    # Assign the material to the shell object
    shell_object.data.materials.append(get_or_generate_shell_material())

    body_segment_ids = voxel_data["body_segment_id"]
    unique_ids = np.unique(body_segment_ids)
    for idx, bsid in enumerate(unique_ids):
        if bsid <= 1:
            continue
        mask = body_segment_ids == bsid
        bone_colors = np.zeros([np.sum(mask), 4], dtype=np.float32)
        bone_colors[:, 3] = opacity
        bone_object = generate_part(
            collection,
            f"Bone_{idx}",
            positions[mask],
            orientations[mask],
            ppp_offsets[mask],
            nnn_offsets[mask],
            connectivity[mask],  # TODO: use correct connectivity
            colors=bone_colors,
            smooth=True,
            voxel_size=voxel_size,
        )
        # bone object also transparent
        bone_object.data.materials.append(get_or_generate_shell_material())

    # Read rigid body and joint data
    rigid_body_data = frame_data["rigid_bodies"]
    joint_data = frame_data["joints"]

    # Extract COM positions and orientations
    com_positions = rigid_body_data["com"]  # Shape: (num_rigid_bodies, 3)
    rb_orientations = rigid_body_data["orientation"]  # Shape: (num_rigid_bodies, 4)

    # Apply transform if provided
    if transform is not None:
        com_positions, rb_orientations = transform(com_positions, rb_orientations)

    # Create a mapping from rigid body index to array index
    rigid_body_indices = rigid_body_data["index"]  # Shape: (num_rigid_bodies,)
    index_to_array_idx = {rb_index: i for i, rb_index in enumerate(rigid_body_indices)}

    # Get materials
    red_material = get_or_generate_material("RedMaterial", (1.0, 0.0, 0.0, 1.0))  # RGBA
    blue_material = get_or_generate_material("BlueMaterial", (0.0, 0.0, 1.0, 1.0))
    green_material = get_or_generate_material("GreenMaterial", (0.0, 1.0, 0.0, 1.0))

    # Define sizes
    ball_radius = voxel_size * 2
    arrow_length = voxel_size * 6
    arrow_shaft_radius = voxel_size * 0.6
    arrow_cone_length = voxel_size * 1.2
    arrow_cone_radius = voxel_size * 1.2

    # # Create red spheres at COM positions
    # for i, com in enumerate(com_positions):
    #     # Convert numpy array to tuple
    #     com_tuple = tuple(com.tolist())
    #     # Create a sphere at the COM position
    #     bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_radius, location=com_tuple)
    #     sphere = bpy.context.active_object
    #     sphere.name = f"RigidBody_{i}_COM"
    #     # Remove sphere from current collection(s)
    #     for coll in sphere.users_collection:
    #         coll.objects.unlink(sphere)
    #     # Add sphere to our collection
    #     collection.objects.link(sphere)
    #     # Assign the red material
    #     if sphere.data.materials:
    #         sphere.data.materials[0] = red_material
    #     else:
    #         sphere.data.materials.append(red_material)

    # Create arrows at joint positions and along joint axes
    joint_positions = joint_data["position"]  # Shape: (num_joints, 3)
    joint_axes = joint_data["axis"]  # Shape: (num_joints, 3)

    # Apply transform if provided
    if transform is not None:
        joint_positions, _ = transform(joint_positions, np.zeros_like(joint_positions))
        # For axes, we need to rotate them using orientations if the transform includes rotation
        # Here we assume the transform function handles positions and orientations
        # For simplicity, we'll assume axes are transformed as vectors
        _, joint_axes_transformed = transform(np.zeros_like(joint_axes), joint_axes)
        joint_axes = joint_axes_transformed
    else:
        joint_axes = joint_axes

    # # Create joint arrows
    # for i in range(len(joint_positions)):
    #     start_point = tuple(joint_positions[i].tolist())
    #     axis_vector = joint_axes[i]
    #     axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)
    #     # Define end point based on some length
    #     end_point = joint_positions[i] + axis_vector_normalized * arrow_length
    #     end_point = tuple(end_point.tolist())
    #
    #     # Create arrow
    #     generate_arrow_between(
    #         collection,
    #         arrow_name=f"JointArrow_{i}",
    #         start_point=start_point,
    #         end_point=end_point,
    #         shaft_radius=arrow_shaft_radius,
    #         cone_length=arrow_cone_length,
    #         cone_radius=arrow_cone_radius,
    #         material=blue_material,
    #     )
    # Create joint cubes
    for i in range(len(joint_positions)):
        joint_pos = joint_positions[i]
        joint_pos_tuple = tuple(joint_pos.tolist())
        # Create cube at joint position
        bpy.ops.mesh.primitive_cube_add(size=voxel_size * 2, location=joint_pos_tuple)
        cube = bpy.context.active_object
        cube.name = f"JointCube_{i}"
        # Remove cube from current collection(s)
        for coll in cube.users_collection:
            coll.objects.unlink(cube)
        # Add cube to our collection
        collection.objects.link(cube)
        # Assign the green material
        if cube.data.materials:
            cube.data.materials[0] = blue_material
        else:
            cube.data.materials.append(blue_material)

    # # Create blue cylinders between connected rigid bodies
    # rigid_body_a_indices = joint_data["rigid_body_a"]
    # rigid_body_b_indices = joint_data["rigid_body_b"]
    #
    # for i in range(len(rigid_body_a_indices)):
    #     rb_a_index = rigid_body_a_indices[i]
    #     rb_b_index = rigid_body_b_indices[i]
    #     # Map to array indices
    #     idx_a = index_to_array_idx.get(rb_a_index)
    #     idx_b = index_to_array_idx.get(rb_b_index)
    #     if idx_a is None or idx_b is None:
    #         continue  # Rigid body index not found, skip
    #     com_a = com_positions[idx_a]
    #     com_b = com_positions[idx_b]
    #     # Convert numpy arrays to tuples
    #     com_a_tuple = tuple(com_a.tolist())
    #     com_b_tuple = tuple(com_b.tolist())
    #     # Create cylinder between COMs
    #     generate_cylinder_between(
    #         collection,
    #         f"Joint_{i}",
    #         com_a_tuple,
    #         com_b_tuple,
    #         radius=voxel_size * 0.5,
    #         material=blue_material,
    #     )


def generate_robot_structure(
    collection,
    builder: SimBuilder,
    transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
    shell_color=(0, 0, 1),
    voxel_size: float = 0.01,
):
    if shell_color is None:
        shell_color = [0, 0, 1]
    is_not_empty = builder.structure()["is_not_empty"]
    segment_id = builder.structure()["segment_id"]

    # Find surface voxels
    # Create an empty 3x3x3 array of boolean type
    connectivity_mask = np.zeros((3, 3, 3), dtype=bool)

    # Set the six neighbors (6-connectivity) to True
    # Direct neighbors in x, y, and z directions from the center voxel [1, 1, 1]
    connectivity_mask[1, 1, 0] = True  # Front
    connectivity_mask[1, 1, 2] = True  # Back
    connectivity_mask[1, 0, 1] = True  # Up
    connectivity_mask[1, 2, 1] = True  # Down
    connectivity_mask[0, 1, 1] = True  # Left
    connectivity_mask[2, 1, 1] = True  # Right

    all_positions = (
        np.indices(is_not_empty.shape, dtype=np.float32) + 0.5
    ) * voxel_size
    non_empty_positions = all_positions.transpose((1, 2, 3, 0))[is_not_empty]
    min_z = np.min(non_empty_positions[:, 2])
    center_offset = (
        np.min(non_empty_positions, axis=0) + np.max(non_empty_positions, axis=0)
    ) / 2
    center_offset[2] = min_z - voxel_size

    for part_idx, segment in enumerate(
        [is_not_empty] + [segment_id == i for i in range(1, np.max(segment_id) + 1)]
    ):
        neighbor_labels = SimBuilder.convolve_gather(segment, connectivity_mask)
        is_surface = np.logical_and(segment != 0, np.any(neighbor_labels == 0, axis=-1))
        positions = (
            all_positions.transpose((1, 2, 3, 0))[is_surface] - center_offset[None]
        )
        orientations = np.zeros([positions.shape[0], 4], dtype=np.float32)
        orientations[:, -1] = 1

        if transform is not None:
            positions, orientations = transform(positions, orientations)

        nnn_offsets = np.full(positions.shape, -voxel_size / 2, dtype=np.float32)
        ppp_offsets = np.full(positions.shape, voxel_size / 2, dtype=np.float32)
        colors = np.zeros([positions.shape[0], 4], dtype=np.float32)

        if part_idx == 0:
            colors[:] = np.array([list(shell_color) + [0.5]], dtype=np.float32)
        else:
            colors[:] = np.array([[1, 1, 1, 1]], dtype=np.float32)
        # neighbor labels order: [-X, -Y, -Z, +Z, +Y, +X]
        connectivity = (
            neighbor_labels[is_surface][:, 5]
            & (neighbor_labels[is_surface][:, 0] << 1)
            & (neighbor_labels[is_surface][:, 4] << 2)
            & (neighbor_labels[is_surface][:, 1] << 3)
            & (neighbor_labels[is_surface][:, 3] << 4)
            & (neighbor_labels[is_surface][:, 2] << 5)
        )
        part_object = generate_part(
            collection,
            "Shell" if part_idx == 0 else f"Bone_{part_idx - 1}",
            positions,
            orientations,
            ppp_offsets,
            nnn_offsets,
            connectivity,
            colors=colors,
            smooth=True,
        )
        part_object.data.materials.append(
            get_or_generate_shell_material()
            if part_idx == 0
            else get_or_generate_bone_material()
        )


def compute_transform_for_put_robot_to_ground(
    reader: RiseSimulationDataReader,
    frame: int,
    target_xy_position: Tuple[float, float] = (0.0, 0.0),
):
    voxel_positions = reader.read_voxel_data(frame)["position"]
    z_offset = float(np.min(voxel_positions[:, 2]))
    xy_offset = (
        np.min(voxel_positions[:, :2], axis=0, keepdims=True)
        + np.max(voxel_positions[:, :2], axis=0, keepdims=True)
    ) / 2 - np.array([target_xy_position])

    def transform_frame(positions, orientations):
        new_positions = np.copy(positions)
        new_positions[:, 2] -= z_offset
        new_positions[:, :2] -= xy_offset
        return new_positions, orientations

    return transform_frame


def compute_transform_for_put_robot_to_elevated_ground(
    reader: RiseSimulationDataReader,
    frame: int,
    height: np.ndarray,
    x_size: float,
    y_size: float,
    target_xy_position: Tuple[float, float] = (0.0, 0.0),
):
    elevated_ground_size = np.array([x_size, y_size])
    elevated_ground_values_size = np.array([height.shape[0], height.shape[1]])
    xy_index = (
        (np.array(target_xy_position) - 1e-3 - elevated_ground_size / 2)
        / elevated_ground_size
        * elevated_ground_values_size
    ).astype(int)
    target_height = height[xy_index[1], xy_index[0]]
    voxel_positions = reader.read_voxel_data(frame)["position"]
    z_offset = float(np.min(voxel_positions[:, 2])) - target_height
    xy_offset = (
        np.min(voxel_positions[:, :2], axis=0, keepdims=True)
        + np.max(voxel_positions[:, :2], axis=0, keepdims=True)
    ) / 2 - np.array([target_xy_position])

    def transform_frame(positions, orientations):
        new_positions = np.copy(positions)
        new_positions[:, 2] -= z_offset
        new_positions[:, :2] -= xy_offset
        return new_positions, orientations

    return transform_frame


def compute_transform_for_put_robot_to_floor(
    reader: RiseSimulationDataReader,
    frame: int,
    target_xy_position: Tuple[float, float] = (0.0, 0.0),
):
    floor_config = reader.read_config().get("floor_elevation_config", {})
    if len(floor_config) == 0:
        raise ValueError(
            "Floor is not configured in config, "
            "use compute_transform_for_put_robot_to_ground instead"
        )

    floor_size = np.array([floor_config["x_size"], floor_config["y_size"]])
    floor_values_size = np.array([floor_config["x_values"], floor_config["y_values"]])
    xy_index = (
        (np.array(target_xy_position) - 1e-3 - floor_size / 2)
        / floor_size
        * floor_values_size
    ).astype(int)
    target_height_normalized = floor_config["height"][
        xy_index[0] + xy_index[1] * floor_values_size[0]
    ]
    target_height = (
        target_height_normalized
        / 65536
        * (floor_config["h_max"] - floor_config["h_min"])
        + floor_config["h_min"]
    )
    voxel_positions = reader.read_voxel_data(frame)["position"]
    z_offset = float(np.min(voxel_positions[:, 2])) - target_height
    xy_offset = (
        np.min(voxel_positions[:, :2], axis=0, keepdims=True)
        + np.max(voxel_positions[:, :2], axis=0, keepdims=True)
    ) / 2 - np.array([target_xy_position])

    def transform_frame(positions, orientations):
        new_positions = np.copy(positions)
        new_positions[:, 2] -= z_offset
        new_positions[:, :2] -= xy_offset
        return new_positions, orientations

    return transform_frame


def compute_transform_for_align_trajectory_to_y(
    reader: RiseSimulationDataReader, frame_start: int = 0, frame_end: int = -1
):
    """
    Compute the necessary rotation and translation to align the trajectory along the y-axis.

    Returns:
        transform_frame: a function that transforms positions and orientations for a frame
    """
    # Compute Center of Mass (com) for the start and end frames
    com_start = np.mean(reader.read_voxel_data(frame_start)["position"], axis=0)
    com_end = np.mean(
        reader.read_voxel_data(len(reader) - 1 if frame_end == -1 else frame_end)[
            "position"
        ],
        axis=0,
    )

    # Compute delta and distance
    delta = com_end - com_start
    dx, dy = delta[0], delta[1]
    distance = np.hypot(dx, dy)

    # Calculate rotation angle theta
    alpha = np.arctan2(dy, dx)
    theta = np.pi / 2 - alpha  # Rotation angle in radians

    # Create rotation matrix for rotation about the z-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Create rotation quaternion representing rotation about z-axis by angle theta
    theta_half = theta / 2.0
    sin_theta_half = np.sin(theta_half)
    cos_theta_half = np.cos(theta_half)
    # Quaternion in [x, y, z, w] format
    q_r = np.array([0.0, 0.0, sin_theta_half, cos_theta_half])

    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions q1 and q2.
        Quaternions are in [x, y, z, w] format.
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])

    def transform_frame(frame_positions, frame_orientations):
        """
        Transform positions and orientations for a single frame.

        Parameters:
        - frame_positions: numpy array with shape [N, 3]
        - frame_orientations: numpy array with shape [N, 4] in [x, y, z, w] format

        Returns:
        - transformed_positions: numpy array with shape [N, 3]
        - transformed_orientations: numpy array with shape [N, 4] in [x, y, z, w] format
        """
        # Copy to avoid modifying the original data
        transformed_positions = frame_positions.copy()
        transformed_orientations = frame_orientations.copy()

        # Subtract x, y of com_start (translation to origin)
        transformed_positions[:, :2] -= com_start[None, :2]

        # Apply rotation
        transformed_positions = transformed_positions.dot(rotation_matrix.T)

        # Translate along y-axis to center the trajectory
        transformed_positions[:, 1] += -distance / 2.0

        # Adjust orientations by rotating them with q_r
        for i in range(transformed_orientations.shape[0]):
            q_i = transformed_orientations[i]
            q_t = quaternion_multiply(q_r, q_i)
            transformed_orientations[i] = q_t

        return transformed_positions, transformed_orientations

    return transform_frame
