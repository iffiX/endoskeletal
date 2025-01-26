import bpy

from .reader import RiseSimulationDataReader

def register():
    pass

def unregister():
    pass

bl_info = {
    "name": "Rise Simulation",
    "blender": (4, 2, 1),
    "category": "Object",
}

classes = {
    RiseSimulationDataReader
}

