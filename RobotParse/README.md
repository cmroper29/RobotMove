# robotparse

Turn industrial robot programs into a **TCP trajectory sampled at a fixed time step**: position
`x, y, z` plus the tool **direction vector**, ready to feed a simulation.

| Language | Files |
|---|---|
| KUKA KRL | `.src` (+ matching `.dat`, optional `$config.dat`) |
| FANUC TP (ASCII) | `.ls` |
| ABB RAPID | `.mod`, `.modx`, `.prg`, `.sys` |

Two planning modes:

* **dumb**: a Cartesian estimate. Motions become lines, arcs and splines, with corners rounded by the
  programmed zone/CNT/approximation. The path is timed with the programmed TCP and orientation
  speeds and a Cartesian acceleration limit. Joint moves are approximated as straight lines.
  No robot model is needed.
* **smart**: uses a URDF. Joint moves interpolate in joint space (IK picks the programmed arm
  configuration). Linear, circular and spline moves are tracked with IK. The path is timed with the
  URDF joint velocity limits, joint acceleration limits and the programmed speeds, and the TCP comes
  from forward kinematics of the resulting joint trajectory. Joint values can be written too.

Both modes compute a time-optimal parameterisation with [TOPP-RA](https://github.com/hungpham2511/toppra),
so the robot accelerates, slows through tight blends and stops at exact-stop points instead of
jumping between constant speeds.

## Install

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[plot]"
```

Python ≥ 3.10. Dependencies: numpy, scipy, lark, pyyaml, toppra, pin (pinocchio); matplotlib for `--plot`.

## Usage

```bash
# dumb mode, 10 ms step
robotparse examples/kuka/weld_demo.src -c examples/kuka/config.yaml -o weld.csv

# smart mode with a URDF, 4 ms step, joints and quaternion columns, metres
robotparse examples/abb/demo.mod -c examples/abb/config.yaml -m smart --dt 0.004 --joints --quat --units m -o demo.csv

# check what was parsed, without planning
robotparse examples/fanuc/demo.ls -c examples/fanuc/config.yaml --list

# quick-look plot of path, speed and joints
robotparse examples/fanuc/demo.ls -c examples/fanuc/config.yaml -m smart --plot demo.png
```

Useful flags: `--urdf`, `--tip-link`, `--axis z|x|-z…`, `--speed`, `--source` (program line being
executed), `--entry <routine>`, `--vendor`, `-v/--verbose` (list every warning), `-q` (result line
only), `--strict` (exit 3 on warnings), `--no-template`, `--debug` (show tracebacks).
Pass several files to resolve calls across files: FANUC `CALL SUB` needs `SUB.ls`; KUKA and RAPID
subroutines in other modules work the same way. The first file is the main program.

### First run with your own program

1. Run it with just the program and a time step:
   `robotparse my_program.src --dt 0.05`
2. Read the **readiness report** printed at the end. It shows:
   * how many motions were used, and which were skipped and why;
   * information the program uses that isn't in the files (tool and user frames, position
     registers, undefined data, missing `.dat` files or modules, a robot model for joint positions),
     with where to find each value on the controller;
   * problems, such as unreachable targets in smart mode;
   * the assumptions to check: direction axis, frame, acceleration, and branches that depend on inputs.
3. If something configurable is missing, a template is written next to the output
   (`my_program_dumb_missing.yaml`). Replace each `TODO` with the real value, or delete the
   entries you don't have. A config that still contains `TODO` is rejected, so a forgotten
   value can't silently become zero.
4. Run again with the template added: `robotparse my_program.src --dt 0.05 -c my_program_dumb_missing.yaml`.
   `-c` can be repeated; later files override earlier ones. Existing templates are never overwritten.

Exit codes: `0` complete; `4` output written but incomplete (skipped motions, missing data, or
unreachable/joint-limit problems); `1` failed and nothing was written; `2` bad command-line
arguments; `3` `--strict` with warnings.

### Input files

* Files can be UTF-8 (with or without BOM), UTF-16 or Windows-1252, with any line endings or
  upper-case extensions.
* ABB `.pgf` program files expand to the modules they list.
* KUKA: pass the `.src`; the `.dat` with the same name is loaded automatically.
* FANUC binary `.tp` files, archives (`.zip`) and folders are rejected with a message saying what to
  export or pass instead.

### Output

CSV (or `.npz` if the output name ends in `.npz`), one row per time step `t = k·dt`, from 0 through the end of the program:

```
t,x_mm,y_mm,z_mm,dir_x,dir_y,dir_z[,qw,qx,qy,qz][,speed_mm_s][,joint_…][,source]
```

* Positions are in the robot base frame, or the world frame if `world_from_robot` is set.
* `dir_*` is a unit vector along the tool axis. Defaults follow each vendor's tool convention:
  **KUKA +X**, **ABB +Z**, **FANUC +Z**. Override with `--axis` or `direction_axis`.
* Joints are in controller units (deg) after applying the joint mapping from the config.

### Python API

```python
from robotparse import load_config, parse_program
from robotparse.kinematics import RobotModel
from robotparse.planner import plan

cfg = load_config("examples/fanuc/config.yaml")
program = parse_program(["examples/fanuc/demo.ls"], cfg)
robot = RobotModel(cfg["robot"])  # only needed for smart mode / joint targets
traj = plan(program, cfg, mode="smart", dt=0.005, robot=robot)
traj.t, traj.position, traj.rotation, traj.joints, traj.warnings
```

## Configuration

All keys are optional. Relative paths are resolved against the config file. Unknown keys are
rejected with a suggestion (`motion.cart_acel` → `motion.cart_accel`), and frames are checked when
the config is loaded.

```yaml
direction_axis: z            # tool axis reported as the direction (vendor default if omitted)
units: mm                    # or m
world_from_robot: {xyz: [0, 0, 0], rpy: [0, 0, 0]}
output_tool: active          # active | flange | <tool number> | frame spec (see below)
entry: main                  # routine to run

tools:                       # KUKA TOOL_DATA[n] / FANUC UTOOL n (ABB tools come from the program)
  1: {xyz: [150, 0, 0], abc: [0, 0, 0]}
bases:                       # KUKA BASE_DATA[n] / FANUC UFRAME n
  1: {xyz: [400, 0, 0], wpr: [0, 0, 0]}

motion:
  cart_accel: 2500           # mm/s² path acceleration for Cartesian moves
  default_tcp_speed: 250     # mm/s if the program never sets one
  max_ori_speed: 500         # deg/s orientation speed cap
  ptp_tcp_speed: 2000        # dumb mode: TCP speed of a 100 % joint move
  ptp_ori_speed: 360         # dumb mode: orientation speed of a 100 % joint move
  fanuc_cnt_time: 0.25       # s: CNTn rounds a corner by about n% · speed · this
  kuka_spline_join: 0.25     # blend fraction between segments inside SPLINE blocks
  grid_step: 2.0             # mm: timing grid (dumb)

robot:                       # smart mode (and FK of joint targets in dumb mode)
  urdf: robot.urdf
  tip_link: tool0            # URDF link matching the controller flange frame
  tip_offset: {xyz: [0, 0, 0], rpy: [0, 0, 0]}   # controller flange relative to tip_link
  joint_offsets: [0, 0, 0, 0, 0, 0]              # controller = sign·urdf + offset (deg)
  joint_signs: [1, 1, 1, 1, 1, 1]
  fanuc_j23_coupling: false  # FANUC J3 is measured from horizontal: urdf_j3 = J3 + J2
  joint_vel_limits: null     # deg/s, overrides URDF <limit velocity>
  joint_acc_limits: null     # deg/s² (URDF has none)
  accel_time: 0.4            # s to reach full joint speed when joint_acc_limits is null
  initial_joints: null       # deg; where the robot starts (default: first target)
  grid_step: 4.0             # mm between IK/timing grid points
  ik_random_seeds: 24

fanuc:
  registers: {5: 250}                                  # R[n]
  position_registers:                                  # PR[n]
    1: {xyz: [0, 0, 100], wpr: [0, 0, 0]}
    2: {joints: [0, -90, 90, 0, 90, 0]}

kuka:
  include_dat: ["$config.dat"]   # TOOL_DATA, BASE_DATA, XHOME, …
```

Frame specs are `{xyz, abc}` (KUKA ZYX), `{xyz, wpr}` / `{xyz, rpy}` (fixed XYZ, degrees),
`{xyz, quat: [w, x, y, z]}` (ABB order), or a 4×4 matrix.

### Getting the robot model right (smart mode)

* **URDF source**: ROS-Industrial support packages (`kuka_experimental`, `fanuc`, `abb`) cover many
  industrial arms, and [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) packages some.
  Convert `.xacro` first, e.g. `pip install xacro && xacro robot.urdf.xacro > robot.urdf`.
  Only kinematics and `<limit>` are used; meshes are ignored.
* **tip_link**: must match the controller flange frame, because program tool data is relative to
  it. In ROS-I models `tool0` has Z pointing out of the flange, which matches ABB and FANUC. KUKA's
  flange has X pointing out: use the `flange` link where one exists, or `tip_offset`.
* **Joint zeros and directions** in the URDF must match the controller. If they don't, set
  `joint_offsets`/`joint_signs`. FANUC arms need `fanuc_j23_coupling: true` unless the URDF
  already models the J2/J3 interaction.
* **Acceleration limits** aren't in URDF. Set `joint_acc_limits` from the robot datasheet for
  realistic cycle times.

## How programs are interpreted

**Common**
* Subroutines are inlined, and FOR loops with constant bounds are unrolled (loop variables work
  in expressions such as `path{i}`).
* Program logic is only known offline as far as the program's own data goes. In KRL, IF, WHILE,
  REPEAT and SWITCH conditions are evaluated when every value in them is known. A condition that
  depends on run-time state (inputs, variables without a value), and every ABB/FANUC condition,
  falls back: IF/SWITCH take the first branch and WHILE/REPEAT run once. LOOP bodies run once.
  Each fallback is reported as a warning.
* A variable declared without a value is *unknown*, never zero. A statement that needs it is
  skipped with a warning.
* The robot starts at the first reachable target, or at `robot.initial_joints`. Dwell/wait
  statements hold the pose.
* A blend (zone) is capped at half of the shorter adjacent segment. A robot stops at `fine`, at
  corners without a blend, and at tool changes.

**KUKA KRL**
* `PTP/LIN/CIRC` (incl. `CA`), `SPTP/SLIN/SCIRC`, `SPLINE…ENDSPLINE` with `SPL`, and `_REL` moves (`#BASE`/`#TOOL`).
* Inline forms: `PDAT_ACT/LDAT_ACT/FDAT_ACT` + `BAS(#PTP_PARAMS|#CP_PARAMS|…)`, KSS 8 `WITH` clauses (`SVEL_CP`, `SAPO`, `STOOL2`, …).
* `$VEL.CP` (m/s), `$VEL.ORI1/2`, `$VEL_AXIS`, `$ACC.CP`, `$ACC_AXIS`, `$APO.CDIS/CPTP/CVEL/CORI`, `$TOOL`, `$BASE`, `$OV_PRO`, geometric operator `:`, partial positions.
* `C_DIS` blends at `$APO.CDIS` mm, and `C_PTP` at `$APO.CPTP` % of half the segment (mm when `APO_MODE #CDIS`). `C_VEL` and `C_ORI` are approximated.
* User routines anywhere in the loaded files: `DEF` subroutines and `DEFFCT` functions (also
  `GLOBAL` ones in other `.src` files), called from statements or inside any expression
  (`LIN MyOffset(P1, 50)`, `$VEL.CP = WeldSpeed(3)`). `RETURN` is supported, `:IN` parameters are
  copies, and `:OUT` parameters (the KRL default) write back to the caller.
* Conditions: `== <> < > <= >=`, `AND OR EXOR NOT` (KRL precedence, so comparisons need
  parentheses when combined), and built-ins `ABS SQRT SIN COS TAN ACOS ATAN2 INV_POS`.
* The E6POS Turn value (`T`) guides the IK solution choice.

**ABB RAPID**
* `MoveJ/MoveL/MoveC/MoveAbsJ` and the `DO`, `Sync`, `AO`, `GO`, `Trigg*`, `Search*`, `Arc*` variants (arguments are recognised by data type).
* `speeddata` (incl. `\V`, `\T`), `zonedata` (incl. `\Z`), `tooldata`, `wobjdata` (`uframe·oframe`), `AccSet`, `VelSet`, `WaitTime`.
* Expressions: `Offs`, `RelTool` (`\Rx/\Ry/\Rz`), `CRobT`, `OrientZYX`, `PoseMult`, `PoseInv`, arithmetic, arrays, record fields and assignments.
* Zones use `pzone_tcp`. `MoveJ` speed data is treated as an approximate TCP speed, as the controller does. `cf1/cf4/cf6` guide the IK solution choice.

**FANUC TP**
* `J/L/C/A` motions. Speeds in `%`, `mm/sec`, `cm/min`, `inch/min`, `deg/sec`, `sec`, `msec`, `max_speed` and `R[n]`.
* `FINE`, `CNTn` (speed-based rounding, see `fanuc_cnt_time`), `CD`/`CR` (mm), `ACCn`, `Offset,PR[n]`, `Tool_Offset,PR[n]`, `INC`, `OFFSET CONDITION`.
* `UFRAME_NUM`/`UTOOL_NUM` and the per-point `UF`/`UT`, `OVERRIDE`, `WAIT (sec)`, `CALL`, `FOR/ENDFOR`, `R[n]=…`, and `PR[n]=P[m]` / `PR[n,i]=…` arithmetic.
* Three or more consecutive `A` moves form circular arcs through neighbouring points.
* The CONFIG turn counts for J1/J4/J6 guide the IK solution choice.

## Limitations

* Conditions that depend on run-time state are not evaluated (see above), ABB and FANUC conditions
  are never evaluated, and jumps (`JMP LBL`, `GOTO`) are not followed.
* Controller-only data (FANUC PR/R values, UFRAME/UTOOL, KUKA `$config.dat`) must be supplied through the config. Missing data produces a warning and defaults to identity or zero.
* External axes, conveyor tracking, stationary tools/robot-held workpieces, and interrupts/triggers are not supported.
* Joint jerk is not limited, and vendor blend shapes are approximations (parabolic corner blends; FANUC CNT is a speed heuristic).
* Only revolute/prismatic serial chains (no continuous joints) are supported in smart mode. IK config matching uses turn/quadrant data, not the full shoulder/elbow/wrist flags.

## Development

```bash
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

Layout: `robotparse/parsers/{kuka,abb,fanuc}.py` (Lark expression grammars plus statement
interpreters), `textio.py` (encodings), `geometry.py` (lines, arcs, splines, blends), `kinematics.py`
(pinocchio FK/IK), `timing.py` (TOPP-RA), `planner.py` (dumb/smart), `report.py` (readiness report
and template), `output.py`, `cli.py`.
`examples/urdf/kr6_r900.urdf` is a KR6 R900-class kinematic model based on ROS-Industrial
`kuka_experimental` (Apache-2.0). Use it for testing, not for accurate cycle times.
