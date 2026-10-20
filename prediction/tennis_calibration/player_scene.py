"""Presentation-only court for the native tennis predictor viewer.

The court is a visual reference arranged around each measured trajectory. Its
placement is not a calibration of the acquisition frame to a real tennis court.
Measured positions and all simulator dynamics remain in the original frame.
User-scene decoration never enters MjModel collision geometry or force callbacks.
"""
from __future__ import annotations

import numpy as np


def configure_scene(sim, trace, height):
    """Set only renderer properties and return a reusable decoration layout.

    Call when changing trajectory, before the next draw. ``height`` is the
    selected model's effective ball-center contact height; the rendered surface
    follows height minus its assumed radius. No qpos, qvel or option is changed.
    """
    positions = np.asarray(trace["raw_xyz"], float)
    finite = positions[np.isfinite(positions).all(axis=1)]
    if finite.size == 0:
        raise ValueError("Scene needs at least one finite measured position")
    lower, upper = np.percentile(finite, [1, 99], axis=0)
    center = (lower + upper) / 2
    length = max(23.77, float(upper[0] - lower[0]) + 6.0)
    width = max(10.97, float(upper[1] - lower[1]) + 5.0)
    floor = float(height) - float(sim.parameters["radius_m"])
    model = sim.model
    ground = model.geom("effective_ground_visual").id
    ball = model.geom("ball_visual").id
    model.geom_rgba[ground] = [.42, .53, .54, 1.]
    # This particular plane is already contype=conaffinity=0. Its dimensions
    # and XY position therefore only control the visible backdrop footprint.
    model.geom_size[ground, :2] = [500., 500.]
    model.geom_pos[ground, :2] = center[:2]
    model.geom_rgba[ball] = [.92, .99, .08, 1.]
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.25, .27, .29]
    model.vis.headlight.diffuse[:] = [.50, .50, .48]
    model.vis.headlight.specular[:] = [.12, .12, .10]
    model.vis.rgba.fog[:] = [.74, .85, .91, 1.]
    model.vis.rgba.haze[:] = [.74, .85, .91, 1.]
    model.vis.map.fogstart = 8.
    model.vis.map.fogend = 22.
    model.vis.map.zfar = 150.
    model.vis.quality.shadowsize = 2048
    model.vis.quality.numslices = 20
    model.vis.quality.numstacks = 12
    model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(720, model.vis.global_.offheight)
    if model.nlight:
        model.light_pos[0] = [center[0]-5., center[1]-8., floor+15.]
        model.light_dir[0] = [.20, .30, -1.]
        model.light_type[0] = int(sim.mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        model.light_ambient[0] = [.28, .29, .27]
        model.light_diffuse[0] = [.80, .79, .72]
        model.light_specular[0] = [.20, .20, .18]
    return {
        "schema": "TENNIS_PLAYER_SCENE/1",
        "center_xy": center[:2].tolist(), "court_length_m": length,
        "court_width_m": width, "floor_z_m": floor,
        "court_visible": True, "net_visible": False,
        "camera": {"lookat": [float(center[0]), float(center[1]), max(1., float(center[2]))],
            "distance": max(16., float(np.linalg.norm(upper-lower))*1.2),
            "azimuth": 135., "elevation": -27.},
        "coordinate_note": "Presentation court; acquisition-frame court extrinsics are unconfirmed",
    }


def _geom(mujoco, scene, kind, size, position, color, *, emission=0.):
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(position, float),
                      np.eye(3).ravel(), np.asarray(color, float))
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    geom.specular = .03
    geom.shininess = .08
    geom.reflectance = 0.
    geom.emission = float(emission)
    scene.ngeom += 1
    return True


def _box(mujoco, scene, half_size, position, color, *, emission=0.):
    return _geom(mujoco, scene, mujoco.mjtGeom.mjGEOM_BOX, half_size,
                 position, color, emission=emission)


def _line(mujoco, scene, start, end, color, width=.025):
    """Thin flat court marking: actual width is twice ``width`` in metres."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    middle = (a+b)/2
    span = np.abs(b-a)/2
    _box(mujoco, scene, [max(width, span[0]), max(width, span[1]), .0007], middle, color,
         emission=.15)


def draw_environment(mujoco, scene, style):
    """Append bounded, purely visual geoms after clearing ``viewer.user_scn``.

    Draw the environment before trajectories. It uses fewer than 100 geoms,
    leaving room for the currently received samples and future prediction paths.
    A caller can toggle ``style['court_visible']`` and ``style['net_visible']``.
    """
    cx, cy = style["center_xy"]
    # Thin measurement and forecast capsules cast distracting aliased shadows;
    # shaded surfaces retain depth while this presentation avoids those marks.
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    z = style["floor_z_m"]
    length, width = style["court_length_m"], style["court_width_m"]
    lx, wy = length/2, width/2
    outer_x, outer_y = lx+4.6, wy+3.8
    if not style.get("court_visible", True):
        return
    # Sub-millimetre presentation offsets prevent coplanar raster flicker. They
    # do not modify the effective event surface used by dynamics/prediction.
    _box(mujoco, scene, [outer_x, outer_y, .003], [cx, cy, z-.0028],
         [.17, .38, .32, 1.])
    _box(mujoco, scene, [lx+.35, wy+.35, .002], [cx, cy, z-.0016],
         [.09, .27, .42, 1.])
    _box(mujoco, scene, [lx, wy, .002], [cx, cy, z-.0014],
         [.13, .34, .52, 1.])
    line_z = z+.002
    white = [.93, .96, .95, 1.]
    for x in (-lx, lx):
        _line(mujoco, scene, [cx+x, cy-wy, line_z], [cx+x, cy+wy, line_z], white)
    for y in (-wy, wy):
        _line(mujoco, scene, [cx-lx, cy+y, line_z], [cx+lx, cy+y, line_z], white)
    singles = wy*(8.23/10.97)
    service = lx*(6.40/(23.77/2))
    for y in (-singles, singles):
        _line(mujoco, scene, [cx-lx, cy+y, line_z], [cx+lx, cy+y, line_z], white, .020)
    for x in (-service, service):
        _line(mujoco, scene, [cx+x, cy-singles, line_z], [cx+x, cy+singles, line_z], white, .020)
    _line(mujoco, scene, [cx-service, cy, line_z], [cx+service, cy, line_z], white, .020)
    for sign in (-1, 1):
        _line(mujoco, scene, [cx+sign*lx, cy, line_z],
              [cx+sign*(lx+.15), cy, line_z], white, .018)
    if style.get("net_visible", False):
        # Optional faint top tape only: its position is presentation geometry.
        _box(mujoco, scene, [.016, wy+.4, .012], [cx, cy, z+.914], [.75, .81, .84, .30])
    fence = [.21, .34, .33, 1.]
    rail = [.35, .49, .47, 1.]
    # An open low railing gives depth without obscuring trajectories with mesh.
    for sign in (-1, 1):
        _box(mujoco, scene, [outer_x, .028, .025],
             [cx, cy+sign*outer_y, z+.65], rail)
        _box(mujoco, scene, [.028, outer_y, .025],
             [cx+sign*outer_x, cy, z+.65], rail)
        for x in np.linspace(-outer_x, outer_x, 8):
            _box(mujoco, scene, [.035, .035, .34],
                 [cx+x, cy+sign*outer_y, z+.34], fence)
        for y in np.linspace(-outer_y, outer_y, 5)[1:-1]:
            _box(mujoco, scene, [.035, .035, .34],
                 [cx+sign*outer_x, cy+y, z+.34], fence)
    # Small planting strips make the surroundings less empty and remain clear
    # of the court and measured path. No imported mesh or texture is required.
    for sign in (-1, 1):
        _box(mujoco, scene, [outer_x-1., .40, .10],
             [cx, cy+sign*(outer_y+1.1), z+.10], [.38, .51, .43, 1.])
        for x in np.linspace(-outer_x+2., outer_x-2., 7):
            _geom(mujoco, scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  [.65, .48, .35], [cx+x, cy+sign*(outer_y+1.1), z+.36],
                  [.33, .52, .38, 1.])
