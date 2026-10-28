"""BWF court in metres: near baseline x=0, net x=6.70, centreline y=0.

Court dimensions include the 40 mm lines. The surrounding mat is decorative;
the net is a rigid two-panel approximation, not a deformable fabric model.
"""
import xml.etree.ElementTree as ET

LENGTH = 13.40
DOUBLES_WIDTH = 6.10
SINGLES_WIDTH = 5.18
LINE_WIDTH = 0.04
NET_X = LENGTH / 2
SHORT_SERVICE = 1.98
LONG_SERVICE_INSET = 0.76
NET_CENTRE_HEIGHT = 1.524
NET_POST_HEIGHT = 1.55
NET_DEPTH = 0.76


def add_badminton_court(root: ET.Element, *, collision_enabled: bool = False) -> None:
    """Replace the court only; preserve robot, floor and generalized coordinates."""
    world = root.find("worldbody")
    if world is None:
        raise ValueError("scene has no worldbody")
    old = world.find("body[@name='court']")
    if old is not None:
        world.remove(old)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").startswith("ct_"):
            asset.remove(mesh)
    court = ET.SubElement(world, "body", name="court", pos="0 0 0")

    def geom(name, kind, rgba, collides=False, **attributes):
        return ET.SubElement(court, "geom", name=name, type=kind, rgba=rgba,
                             mass="0", contype=str(int(collides)),
                             conaffinity=str(int(collides)), **attributes)

    def box(name, xmin, xmax, ymin, ymax, z, color):
        geom(name, "box", color,
             pos=f"{(xmin + xmax) / 2:g} {(ymin + ymax) / 2:g} {z:g}",
             size=f"{(xmax - xmin) / 2:g} {(ymax - ymin) / 2:g} 0.0005")

    half_w = DOUBLES_WIDTH / 2
    half_s = SINGLES_WIDTH / 2
    lw = LINE_WIDTH
    white = "0.98 0.98 0.95 1"
    box("ct_surround", -2, LENGTH + 2, -half_w - 2, half_w + 2,
        0.001, "0.10 0.22 0.26 1")
    box("ct_court", 0, LENGTH, -half_w, half_w, 0.002, "0.08 0.40 0.28 1")

    def line(name, xmin, xmax, ymin, ymax):
        box(name, xmin, xmax, ymin, ymax, 0.004, white)

    line("ct_base_near", 0, lw, -half_w, half_w)
    line("ct_base_far", LENGTH - lw, LENGTH, -half_w, half_w)
    for suffix, width in (("d", half_w), ("s", half_s)):
        line(f"ct_side_{suffix}_l", 0, LENGTH, -width, -width + lw)
        line(f"ct_side_{suffix}_r", 0, LENGTH, width - lw, width)

    # Service boundaries include the line on the receiving service court side.
    near_short = NET_X - SHORT_SERVICE
    far_short = NET_X + SHORT_SERVICE
    line("ct_svc_near", near_short - lw, near_short, -half_w, half_w)
    line("ct_svc_far", far_short, far_short + lw, -half_w, half_w)
    line("ct_long_near", LONG_SERVICE_INSET, LONG_SERVICE_INSET + lw, -half_w, half_w)
    line("ct_long_far", LENGTH - LONG_SERVICE_INSET - lw,
         LENGTH - LONG_SERVICE_INSET, -half_w, half_w)
    line("ct_mid_near", 0, near_short, -lw / 2, lw / 2)
    line("ct_mid_far", far_short, LENGTH, -lw / 2, lw / 2)

    # Thin convex prisms: exact endpoint heights, open space below the net.
    # Two straight slopes approximate the 26 mm sag without cloth dynamics.
    for side, y0, y1, h0, h1 in (
        ("l", -half_w, 0, NET_POST_HEIGHT, NET_CENTRE_HEIGHT),
        ("r", 0, half_w, NET_CENTRE_HEIGHT, NET_POST_HEIGHT),
    ):
        for part, depth, thickness, color, collides in (
            ("net", NET_DEPTH, 0.005, "0.10 0.12 0.12 0.55", collision_enabled),
            ("tape", 0.075, 0.006, white, False),
        ):
            name = f"ct_{part}_{side}"
            vertices = [v for x in (-thickness / 2, thickness / 2)
                        for y, h in ((y0, h0), (y1, h1))
                        for z in (h - depth, h) for v in (x, y, z)]
            ET.SubElement(asset, "mesh", name=name,
                          vertex=" ".join(f"{v:g}" for v in vertices))
            geom(name, "mesh", color, collides, mesh=name, pos=f"{NET_X:g} 0 0")
        geom(f"ct_post_{side}", "cylinder", "0.85 0.85 0.87 1", collision_enabled,
             pos=f"{NET_X:g} {y0 if side == 'l' else y1:g} {NET_POST_HEIGHT / 2:g}",
             size=f"0.025 {NET_POST_HEIGHT / 2:g}")
