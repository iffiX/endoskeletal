# RISE Simulator Configuration & Usage Guide

This document describes how to structure a configuration file (RSC XML format) for the **RISE** simulator and how to use the provided **Python interface** (`Rise` class and related functionality) to run simulations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Configuration File Structure](#configuration-file-structure)
   1. [Root Elements](#root-elements)
   2. [Simulator Block](#simulator-block)
   3. [Voxel Block](#voxel-block)
   4. [Structure Block](#structure-block)
   5. [Bodies Block](#bodies-block)
   6. [Constraints Block](#constraints-block)
   7. [Save Block](#save-block)
3. [Python Interface](#python-interface)
   1. [Installing and Importing RISE](#installing-and-importing-rise)
   2. [Core Classes and Types](#core-classes-and-types)
      - [Rise](#rise)
      - [RiseFrame](#riseframe)
      - [RiseKernelCallback](#risekernelcallback)
      - [Data Types](#data-types)
   3. [Running Simulations](#running-simulations)
      - [Configuring Simulations](#configuring-simulations)
      - [Example Usage](#example-usage)
4. [Appendix: Full Example RSC Files](#appendix-full-example-rsc-files)

---

## Introduction

**RISE** is a physics-based simulator that supports soft voxels, rigid bodies, joints, and various mechanical properties. You provide a configuration file (in XML-based RSC format), which describes:
- How the simulator should run (time step, damping, solver settings, etc.).
- The palette of voxel materials (with elastic modulus, density, etc.).
- The geometry of bodies (how voxels are laid out in 3D grids).
- Constraints or joints between bodies.
- Output paths and recording settings.

**RISE** offers a **Python interface** to run these simulations on the GPU. You can also attach a callback function to control signals (like expansion or rotation signals) for closed-loop or more advanced control.

---

## Configuration File Structure

You can have one or more `<RSC>` blocks in your configuration. The simulator typically expects a combined XML text, or multiple `<RSC>` blocks appended. The general form is:

```xml
<RSC Version="0.1">
    <Simulator>
        ...
    </Simulator>
    <Voxel>
        ...
    </Voxel>
</RSC>

<RSC>
    <Structure>
        ...
    </Structure>
    <Simulator>
        ...
    </Simulator>
    <Save>
        ...
    </Save>
</RSC>
```

Below we describe each part in detail.

### Root Elements

- **`<RSC>`**: The outer container for each piece of the configuration.
  - You may define multiple `<RSC>` blocks in separate files, the simulator will combine the input RSC files. The usual
  practice is to have a general environment RSC file and a specific robot RSC file.

### Simulator Block

```xml
<Simulator>
    <Integration>
        <DtFrac>1</DtFrac>
    </Integration>
    <Damping>
        <InternalDampingZ>0.5</InternalDampingZ>
        <CollisionDampingZ>0.2</CollisionDampingZ>
        <GlobalDampingZ>0.03</GlobalDampingZ>
    </Damping>
    <RigidSolver>
        <RigidIterations>10</RigidIterations>
        <BaumgarteRatio>0.01</BaumgarteRatio>
    </RigidSolver>
    <Condition>
        <ResultStartCondition>
            t >= 0
        </ResultStartCondition>
        <ResultStopCondition>
            <![CDATA[
                t >= 10
            ]]>
        </ResultStopCondition>
        <StopCondition>
            <![CDATA[
                t >= 10
            ]]>
        </StopCondition>
    </Condition>
    <Gravity>
        <GravAcc>-9.81</GravAcc>
        <FloorEnabled>1</FloorEnabled>
    </Gravity>
    <Signal>
        <ControlFrequency>10</ControlFrequency>
    </Signal>
</Simulator>
```

- **`<Integration>`**:
  - `<DtFrac>`: The fraction that determines the simulation time step. The default value is 1, which uses the integration 
  timestep determined automatically from voxel properties. Smaller values typically increase stability but take longer to simulate.

- **`<Damping>`**:
  - `<InternalDampingZ>`: Damping factor for internal interactions between voxels through links connecting them.
  - `<CollisionDampingZ>`: Damping factor for collisions.
  - `<GlobalDampingZ>`: Overall global damping for velocities.

- **`<RigidSolver>`**:
  - `<RigidIterations>`: Number of constraint-solver iterations for rigid bodies each step.
  - `<BaumgarteRatio>`: Baumgarte stabilization factor for constraints.

- **`<Condition>`**:
  - `<ResultStartCondition>`: When to start recording or collecting results.
  - `<ResultStopCondition>`: When to stop recording or collecting results.
  - `<StopCondition>`: When to fully stop the simulation.
  - The condition is an expression that computes a boolean value. The result is evaluated as a float and compared with 0.0.
    A positive result indicates a true condition.
    - **Constants**:
      - `E`: Represents the mathematical constant e.
      - `PI`: Represents the mathematical constant π.

    - **Variables**:
      - Supports variables `x`, `y`, `z`, `x0`, `y0`, `z0`, and `t`, with `x`, `y`, and `z` representing the current 
      center of mass position of the structure, `x0`, `y0`, and `z0` representing the initial position of the structure, 
      and `t` representing the current simulation time.

    - **Functions**:
      - Supported mathematical functions include `sqrt` (square root), `sin` (sine), `cos` (cosine), `tan` (tangent), `atan` (arc tangent), `log` (natural logarithm), `int` (integer part), and `abs` (absolute value).

    - **Operators**:
      - Arithmetic operators: `+` (addition), `-` (subtraction), `*` (multiplication), `/` (division), `%` (modulus).
      - Relational and logical operators: `!=` (not equal), `<=` (less than or equal), `>=` (greater than or equal), `==` (equal), `<` (less than), `>` (greater than), `||` (logical OR), `&&` (logical AND).
      - Power operator: `^` (exponentiation).

- **`<Gravity>`**:
  - `<GravAcc>`: The acceleration due to gravity (negative value means downward).
  - `<FloorEnabled>`: Whether to enable a floor collision plane.

  - `<FloorElevation>`: *(Optional)* Defines the elevation profile of the floor.

      ```xml
      <FloorElevation>
          <X_Size>0.3</X_Size>
          <Y_Size>0.3</Y_Size>
          <X_Values>2</X_Values>
          <Y_Values>2</Y_Values>
          <Height>
              0, 0.05, 0, 0.05
          </Height>
      </FloorElevation>
      ```

      - **`<X_Size>`**: The size of the floor in the X direction (meters).
      - **`<Y_Size>`**: The size of the floor in the Y direction (meters).
      - **`<X_Values>`**: The number of subdivisions or grid points along the X axis.
      - **`<Y_Values>`**: The number of subdivisions or grid points along the Y axis.
      - **`<Height>`**: A comma-separated list of height values corresponding to each grid point. The number of height values should match `X_Values * Y_Values`. This defines the elevation at each grid point, allowing for a non-flat floor surface.

      **Example Explanation**:
      ```xml
      <FloorElevation>
          <X_Size>0.3</X_Size>
          <Y_Size>0.3</Y_Size>
          <X_Values>2</X_Values>
          <Y_Values>2</Y_Values>
          <Height>
              0, 0.05, 0, 0.05
          </Height>
      </FloorElevation>
      ```
      - **Floor Dimensions**: The floor spans 0.3 meters in both X and Y directions.
      - **Grid Subdivisions**: The floor is divided into a 2x2 grid.
      - **Height Values**: The heights at the four grid points are 0, 0.05, 0, and 0.05 meters respectively, creating a stepped elevation pattern.

      **Notes**:
      - The `<FloorElevation>` block is optional. If not provided, the floor will be flat at height `0`.
      - Ensure that the number of height values matches the product of `<X_Values>` and `<Y_Values>`.

- **`<Signal>`**:
  - `<ControlFrequency>`: The frequency at which you can update signals (expansion, rotation, etc.) if using the callback interface.

### Voxel Block

```xml
<Voxel>
    <Size>0.01</Size>
    <Palette>
        <Material ID="1">
            <Name>Body</Name>
            <Display>
                <Red>1</Red>
                <Green>0</Green>
                <Blue>0</Blue>
                <Alpha>0.3</Alpha>
            </Display>
            <Mechanical>
                <ElasticMod>3e4</ElasticMod>
                <Density>800</Density>
                <PoissonsRatio>0.35</PoissonsRatio>
                <FrictionStatic>1</FrictionStatic>
                <FrictionDynamic>0.8</FrictionDynamic>
                <MaxExpansion>0.5</MaxExpansion>
                <MinExpansion>-0.5</MinExpansion>
            </Mechanical>
        </Material>
        <Material ID="2">
            <Name>Body</Name>
            <Display>
                <Red>0</Red>
                <Green>0</Green>
                <Blue>1</Blue>
                <Alpha>0.75</Alpha>
            </Display>
            <Mechanical>
                <ElasticMod>3e4</ElasticMod>
                <Density>1500</Density>
                <PoissonsRatio>0.35</PoissonsRatio>
                <FrictionStatic>1</FrictionStatic>
                <FrictionDynamic>0.8</FrictionDynamic>
                <MaxExpansion>0.5</MaxExpansion>
                <MinExpansion>-0.5</MinExpansion>
            </Mechanical>
        </Material>
    </Palette>
</Voxel>
```

- **`<Size>`**: The edge length of a single voxel (in meters).
- **`<Palette>`**: List of materials, each with:
  - **`<Material ID="...">`**: Unique ID to reference in the structure.
    - `<Name>`: A string name (just for reference).
    - `<Display>`: Visualization properties (RGBA).
    - `<Mechanical>`: Mechanical properties:
      - `ElasticMod` (Young’s modulus),
      - `Density`,
      - `PoissonsRatio`,
      - `FrictionStatic`, `FrictionDynamic` (for contact),
      - `MaxExpansion`, `MinExpansion` (allowable expansion range if using expansion signals).

### Structure Block

```xml
<Structure>
    <Bodies>
        ...
    </Bodies>
    <Constraints>
        ...
    </Constraints>
</Structure>
```

The **`<Structure>`** element contains multiple bodies and the constraints (joints) between them.

### Bodies Block

```xml
<Bodies>
    <Body ID="1">
        <Orientation>0,0,0,1</Orientation>
        <OriginPosition>0,0,0</OriginPosition>
        <X_Voxels>3</X_Voxels>
        <Y_Voxels>3</Y_Voxels>
        <Z_Voxels>9</Z_Voxels>
        <MaterialID>
            <Layer>2,2,2,2,2,2,2,2,2</Layer>
            ...
        </MaterialID>
        <SegmentID>
            <Layer>1,1,1,1,1,1,1,1,1</Layer>
            ...
        </SegmentID>
        <SegmentType>
            <Layer>0,0,0,0,0,0,0,0,0</Layer>
            ...
        </SegmentType>
    </Body>
    <Body ID="2">
        ...
    </Body>
    <Body ID="3">
        ...
    </Body>
</Bodies>
```

For **each `<Body>`**:
- **`ID`**: Unique ID among all bodies.
- **`<Orientation>`**: Quaternion (x, y, z, w) specifying the body’s initial rotation.
- **`<OriginPosition>`**: (x, y, z) specifying the body’s initial location.
- **`<X_Voxels>`, `<Y_Voxels>`, `<Z_Voxels>`**: Dimensions of the grid of voxels composing this body.
- **`<MaterialID>`** / **`<SegmentID>`** / **`<SegmentType>`**: Each dimension (X, Y, Z) is laid out in “layers” for each Z or Y. 
  - A `<Layer>` tag typically lists values in X across the row. Multiple `<Layer>` tags stack along Y or Z.
  - **`<MaterialID>`** references the IDs defined in the `<Palette>`.
  - **`<SegmentID>`** can define sub-segments within the same body (useful for constraints).
  - **`<SegmentType>`** can specify voxel type (0 = default, 1 = some special type, etc.).

### Constraints Block

```xml
<Constraints>
    <Constraint>
        <Type>HINGE_JOINT</Type>
        <RigidBodyA>
            <BodyID>1</BodyID>
            <SegmentID>2</SegmentID>
            <Anchor>0.015,0.03,0.085</Anchor>
        </RigidBodyA>
        <RigidBodyB>
            <BodyID>2</BodyID>
            <SegmentID>1</SegmentID>
            <Anchor>0.005,0,0.005</Anchor>
        </RigidBodyB>
        <HingeAAxis>0, 1, 0</HingeAAxis>
        <HingeBAxis>0, -1, 0</HingeBAxis>
        <HingeRotationSignalID>0</HingeRotationSignalID>
    </Constraint>
    <Constraint>
        <Type>BALL_AND_SOCKET_JOINT</Type>
        <RigidBodyA>
            <BodyID>2</BodyID>
            <SegmentID>1</SegmentID>
            <Anchor>0.005,0.005,0.03</Anchor>
        </RigidBodyA>
        <RigidBodyB>
            <BodyID>3</BodyID>
            <SegmentID>1</SegmentID>
            <Anchor>0.005,0.005,0</Anchor>
        </RigidBodyB>
    </Constraint>
</Constraints>
```

Each `<Constraint>` can be:
- **`HINGE_JOINT`**: A rotational joint around a given axis. 
  - `HingeRotationSignalID` can link this hinge’s rotation to a signal array index. 
- **`BALL_AND_SOCKET_JOINT`**: A free-rotating joint with no single axis constraint.
- Notes:
  - **RigidBodyA**, **RigidBodyB**: Reference to the two bodies to join. To select a rigid
    body segment, you will reference the body ID and segment ID in the body.
  - **Anchor**: (x, y, z) position of the constraint in the local body frame.
  - **HingeAAxis**, **HingeBAxis**: (x, y, z) axis of the hinge in the local body frame.

### Save Block

```xml
<Save>
    <ResultPath>robot.result</ResultPath>
    <Record>
        <Text>
            <Rescale>0.001</Rescale>
            <Path>robot.history</Path>
        </Text>
        <HDF5>
            <Path>robot.h5_history</Path>
        </HDF5>
    </Record>
</Save>
```

- **`<ResultPath>`**: Final result file path.
- **`<Record>`**: Time-series recording:
  - `<Text>` (Optional): Write a `.history` text file with frames (optionally rescaled positions).
  - `<HDF5>` (Optional): Write an HDF5 file with recorded data.

---

## Python Interface

The **RISE** Python interface allows you to:
1. Create or load multiple RSC configurations.
2. Run them in parallel or sequentially on CPU/GPU.
3. Optionally attach a callback to adjust signals in real time.
4. Collect data (either through files or via callback frames).

### Installing and Importing RISE

Link the correct pre-compiled rise library version from rise-lib to this repository root.

Then in your Python code:

```python
from rise import Rise, RiseKernelCallback, RiseFrame
```

### Core Classes and Types

#### `Rise`

This is the main interface to the RISE engine:

```python
class Rise:
    def __init__(self, devices: List[int] = ..., batch_size_per_device: int = ...) -> None:
        ...
    
    def run_sims(
        self,
        configs: List[List[str]],
        ids: List[int],
        callback: Optional[RiseKernelCallback] = None,
        dt_update_interval: int = 10,
        collision_update_interval: int = 10,
        constraint_update_interval: int = 2,
        divergence_check_interval: int = 100,
        record_buffer_size: int = 500,
        max_steps: int = 1_000_000,
        save_result: bool = True,
        save_record: bool = True,
        policy: str = "batched",
        log_level: str = "info",
    ) -> List[bool]:
        ...
```

1. **Constructor**:
   - `devices`: List of CUDA device indices (e.g., `[0, 1]` for two GPUs). An empty list means use all GPUs.
   - `batch_size_per_device`: How many simulations to run in parallel per device.

2. **`run_sims` method**:
   - `configs`: A list of configurations, each config is a list of **XML strings**. Typically, you read in your `.rsc` or `.xml` text and split it into lines or single string, then pass as `configs[i]`.
   - `ids`: Unique IDs for each simulation. Must have same length as `configs`.
   - `callback`: A user-defined Python callable that conforms to [`RiseKernelCallback`](#risekernelcallback), used for real-time control signals.
   - `dt_update_interval`, `collision_update_interval`, etc.: Fine-tuning for how frequently to update certain aspects in the solver.
   - `record_buffer_size`: Memory buffer for storing simulation frames if you are recording a large number of steps. Use a small value
   such as 10 if your simulation is large, if you have specified callback, the minimum value is 1.
   - `max_steps`: Maximum simulation steps allowed.
   - `save_result` / `save_record`: Whether to produce output files as specified in your `<Save>` block. You may use these bools
   to turn off result and record saving completely for increased speed.
   - `policy`: `"batched"` or `"sequential"`. Batched means run simulations in parallel, sequential means run them one 
   - at a time. Sequential mode is ideal for large simulations with many voxels.
   - `log_level`: `"err"`, `"warn"`, `"info"`, `"debug"`, or `"trace"`.

   **Returns**: A list of booleans indicating success (`True`) or failure (`False`) for each simulation.

#### `RiseFrame`

This class encapsulates a snapshot of the simulation at a given step/time. It provides methods to:
- Get the simulation step (`.step()`) and time (`.time_point()`).
- Access arrays of data about voxels, links, rigid bodies, and joints, for instance:
  - `.voxels()` -> array of `RS_SimulationVoxelRecord`.
  - `.rigid_bodies()` -> array of `RS_SimulationRigidBodyRecord`.
  - `.joints()` -> array of `RS_SimulationJointRecord`.
  - `.links()` -> array of `RS_SimulationLinkRecord`.
  - Also convenience arrays for positions, velocities, etc.

#### `RiseKernelCallback`

This is a protocol (interface) for a callback that receives:
- A list of **kernel IDs** (one per parallel simulation).
- A list of **time points**.
- A list of **`RiseFrame`** objects.
- A list of optional **expansion signals** arrays and **rotation signals** arrays (if declared in RSC).

You can modify the signal arrays in-place for real-time closed-loop control.

```python
class RiseKernelCallback(Protocol):
    def __call__(
        self,
        kernel_ids: List[int],
        time_points: List[float],
        frames: List[RiseFrame],
        expansion_signals: List[Optional[np.ndarray]],
        rotation_signals: List[Optional[np.ndarray]],
    ) -> None:
        ...
```

### Data Types

In the provided `.pyi` stub, you will see custom `numpy.dtype` definitions for each record (voxel, link, rigid body, joint). Examples include:

- `RS_SimulationVoxelRecord`
- `RS_SimulationRigidBodyRecord`
- `RS_SimulationJointRecord`
- etc.

---

## Running Simulations

### Configuring Simulations

1. **Build your XML-based RSC** configuration(s).
2. You may store these in separate files or as a single file with multiple `<RSC>` blocks. 
3. In Python, read the file(s) into a string or a list of lines. Each simulation can have its own config text.

**Example** (reading from a file and putting into `configs`):

```python
with open("env.rsc", "r") as f:
    env_text = f.read()
with open("my_robot.rsc", "r") as f:
    robot_text = f.read()

# Usually you'd have one or more config texts in a list of lists:
configs = [
    [env_text, robot_text],  # Simulation 0
    [env_text, robot_text],  # Simulation 1
]
ids = [123, 456]  # Must match length of configs
```

### Example Usage

```python
import numpy as np
from typing import List, Optional
from rise import Rise, RiseKernelCallback, RiseFrame

# 1. Define an optional callback for closed-loop control
def my_callback(
    kernel_ids: List[int],
    time_points: List[float],
    frames: List[RiseFrame],
    expansion_signals: List[Optional[np.ndarray]],
    rotation_signals: List[Optional[np.ndarray]],
) -> None:
    """
    This callback is invoked periodically (based on <Signal><ControlFrequency> in the RSC)
    for each parallel simulation kernel.
    """
    for i, k_id in enumerate(kernel_ids):
        time = time_points[i]
        frame = frames[i]

        # Example: linearly increase the hinge rotation signal over time
        if rotation_signals[i] is not None:
            # rotation_signals[i] might be an array, e.g. shape (num_rotation_signals,)
            rotation_signals[i][0] = np.sin(time)  # Some function of time

        # You can also inspect frame.voxels(), frame.rigid_bodies(), etc.

# 2. Read or define your RSC XML text
with open("my_robot.rsc", "r") as f:
    config_text = f.read()

configs = [config_text.split("\n")]  # single simulation
ids = [1]

# 3. Create a Rise instance
sim = Rise(devices=[0], batch_size_per_device=128)  # First GPU

# 4. Run the simulation(s)
results = sim.run_sims(
    configs=configs,
    ids=ids,
    callback=my_callback,
    dt_update_interval=10,
    collision_update_interval=10,
    constraint_update_interval=2,
    divergence_check_interval=100,
    record_buffer_size=500,
    max_steps=20000,
    save_result=True,
    save_record=True,
    policy="batched",
    log_level="info",
)

# Check success/failure
success = results[0]
if success:
    print("Simulation completed successfully!")
else:
    print("Simulation failed or was interrupted.")
```

---

## Appendix: Full Example RSC Files

Please refer to `env.rsc` and `example_robot.rsc` in the `data` directory for full example RSC files.

## Appendix: Wrapped simulator interface class

Please refer to `sim/env.py` for a wrapped interface class for the simulator. Which already comes with
easy to use debug functionalities such as debug log saving, and replay of simulation. (So you can 
perform simulation without saving records for faster speed, and then replay certain simulation records)

## Appendix: Rise python interface

Please refer to `rise.pyi` for the full definition of Rise class and RiseFrame class.
