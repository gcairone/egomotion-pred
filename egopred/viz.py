"""Overlays of ego-frame trajectories on camera images and a simple bird's-eye view."""

import numpy as np
from PIL import Image, ImageDraw

from .geometry import inv_T, project, transform_points

HALF_WIDTH = 0.9  # m, half the vehicle width drawn as a ribbon around the path


def densify(xyyaw: np.ndarray, z: np.ndarray | None = None, start=(0.0, 0.0, 0.0), n: int = 8):
    """Insert the start pose and interpolate n points per segment. Returns (P, 3) xyyaw, (P,) z."""
    pts = np.vstack([np.asarray(start)[None], xyyaw])
    zz = np.concatenate([[0.0], z]) if z is not None else np.zeros(len(pts))
    yaw = np.unwrap(pts[:, 2])
    s = np.linspace(0, len(pts) - 1, (len(pts) - 1) * n + 1)
    i = np.arange(len(pts))
    out = np.stack([np.interp(s, i, pts[:, 0]), np.interp(s, i, pts[:, 1]), np.interp(s, i, yaw)], 1)
    return out, np.interp(s, i, zz)


def ribbon_points(xyyaw: np.ndarray, z: np.ndarray, half_width: float = HALF_WIDTH):
    """Centre, left and right edges of the vehicle footprint along a path, as (P, 3) ego points."""
    c = np.stack([xyyaw[:, 0], xyyaw[:, 1], z], 1)
    n = np.stack([-np.sin(xyyaw[:, 2]), np.cos(xyyaw[:, 2]), np.zeros(len(xyyaw))], 1)
    return c, c + half_width * n, c - half_width * n


def draw_path(draw: ImageDraw.ImageDraw, xyyaw, z, T_ego_cam, K, dist, color, size, scale=1.0,
              width=3, ribbon=True, waypoints=True):
    """Project a path onto the image. Returns the fraction of densified points visible."""
    dense, dz = densify(xyyaw, z)
    T_cam_ego = inv_T(T_ego_cam)
    lines = ribbon_points(dense, dz) if ribbon else (ribbon_points(dense, dz)[0],)
    visible = []
    for k, pts in enumerate(lines):
        uv, ok = project(transform_points(T_cam_ego, pts), K, dist, min_depth=1.0)
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
        if k == 0:
            visible.append(inside.mean())
        uv = uv * scale
        for a, b, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                draw.line([tuple(a), tuple(b)], fill=color, width=width if k == 0 else max(1, width - 1))
    if waypoints:
        uv, ok = project(transform_points(T_cam_ego, np.c_[xyyaw[:, :2], np.zeros(len(xyyaw)) if z is None else z]),
                         K, dist, min_depth=1.0)
        for p, o in zip(uv * scale, ok):
            if o:
                draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=color, width=2)
    return float(visible[0])


def overlay(image: Image.Image, paths: list[tuple[np.ndarray, np.ndarray | None, tuple, str]],
            T_ego_cam, K, dist, out_width: int = 960, text: str = "") -> tuple[Image.Image, dict]:
    """paths: (xyyaw (W, 3), z (W,) or None, rgb colour, label). Projection uses the original
    image resolution; drawing happens on a copy resized to out_width."""
    size = image.size
    scale = out_width / size[0]
    im = image.convert("RGB").resize((out_width, round(size[1] * scale)))
    d = ImageDraw.Draw(im)
    vis = {}
    for xyyaw, z, color, label in paths:
        vis[label] = draw_path(d, xyyaw, z, T_ego_cam, K, dist, color, size, scale,
                               ribbon=label == "gt")
    y = 6
    for line in [text] + [label for *_, label in paths]:
        if not line:
            continue
        color = next((c for *_, c, l in paths if l == line), (255, 255, 255))
        d.rectangle([4, y - 2, 12 + 7 * len(line), y + 12], fill=(0, 0, 0))
        d.text((8, y), line, fill=color)
        y += 16
    return im, vis


def bev(paths: list[tuple[np.ndarray, tuple, str]], size: int = 320, meters: float = 60.0) -> Image.Image:
    """Bird's-eye view, ego at bottom centre, x forward = up."""
    im = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(im)
    s = size / meters

    def px(x, y):
        return size / 2 - y * s, size - 10 - x * s

    for m in range(10, int(meters), 10):
        d.line([px(m, -meters / 2), px(m, meters / 2)], fill=(50, 50, 50))
    for xyyaw, color, _ in paths:
        pts = [px(0, 0)] + [px(x, y) for x, y in xyyaw[:, :2]]
        d.line(pts, fill=color, width=2)
        for p in pts[1:]:
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)
    d.rectangle([px(0, 0)[0] - 3, px(0, 0)[1] - 6, px(0, 0)[0] + 3, px(0, 0)[1]], fill=(255, 255, 255))
    return im


def side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width, h))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def virtual_camera(back: float = 9.0, height: float = 6.0, pitch_deg: float = 24.0, size=(800, 540),
                   focal: float = 520.0):
    """A camera behind and above the ego vehicle looking forward and down.
    Returns (T_ego_cam, K) in the project's conventions."""
    from .geometry import make_T

    p = np.radians(pitch_deg)
    fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    T = make_T(np.stack([right, down, fwd], axis=1), [-back, 0.0, height])
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] * 0.62], [0, 0, 1.0]])
    return T, K


def perspective_view(paths: list[tuple[np.ndarray, tuple, str]], size=(800, 540), reach: float = 40.0,
                     half_width: float = 8.0, ss: int = 2, title: str = "") -> Image.Image:
    """Ground-plane grid seen from a virtual camera behind the ego vehicle, with the paths
    drawn on it as ribbons with waypoint markers. paths: (xyyaw (W, 3), rgb colour, label)."""
    W, H = size[0] * ss, size[1] * ss
    T_ego_cam, K = virtual_camera(size=(W, H), focal=520.0 * ss)
    T_cam_ego = inv_T(T_ego_cam)

    def proj(pts):
        return project(transform_points(T_cam_ego, np.asarray(pts, dtype=float)), K, None, min_depth=0.3)

    def depth_of(pts):
        return transform_points(T_cam_ego, np.asarray(pts, dtype=float))[:, 2]

    bg = Image.new("RGBA", (W, H), (18, 20, 24, 255))
    d = ImageDraw.Draw(bg)

    def polyline(pts, fill, width, draw=d):
        uv, ok = proj(pts)
        for a, b, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                draw.line([tuple(a), tuple(b)], fill=fill, width=width)

    # grid: lines across (constant x) and along (constant y), fading with distance
    xs = np.arange(-5, reach + 0.1, 5.0)
    ys = np.arange(-half_width, half_width + 0.1, 2.0)
    for x in xs:
        pts = np.stack([np.full(41, x), np.linspace(-half_width, half_width, 41), np.zeros(41)], 1)
        shade = int(95 - 55 * max(x, 0) / reach)
        polyline(pts, (shade, shade + 6, shade + 14, 255), ss)
    for y in ys:
        pts = np.stack([np.linspace(-5, reach, 91), np.full(91, y), np.zeros(91)], 1)
        shade = 100 if y == 0 else 62
        polyline(pts, (shade, shade + 6, shade + 14, 255), ss)
    f = _font(15 * ss)
    for x in (10, 20, 30, 40):
        if x <= reach:
            uv, ok = proj([[x, half_width + 0.4, 0.0]])
            if ok[0]:
                d.text(tuple(uv[0]), f"{x} m", fill=(150, 160, 175, 255), font=f, anchor="lm")

    # ego vehicle: wireframe box, rear axle at the origin
    x0, x1, hw, hh = -1.0, 3.6, 0.95, 1.5
    c = np.array([[x, y, z] for x in (x0, x1) for y in (-hw, hw) for z in (0.0, hh)])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        polyline(c[[a, b]], (230, 230, 235, 255), ss)

    # paths: translucent footprint ribbon + centre line + waypoint markers
    first_pred = next((p[2] for p in paths if p[2] != "gt"), None)
    # ground truth first and slightly larger, so close predictions stay visible on top of it
    for xyyaw, color, label in sorted(paths, key=lambda p: p[2] != "gt"):
        dense, dz = densify(xyyaw, None, n=10)
        ctr, left, right = ribbon_points(dense, dz, half_width=0.9)
        uv_l, ok_l = proj(left)
        uv_r, ok_r = proj(right)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for i in range(len(dense) - 1):
            if ok_l[i] and ok_l[i + 1] and ok_r[i] and ok_r[i + 1]:
                ld.polygon([tuple(uv_l[i]), tuple(uv_l[i + 1]), tuple(uv_r[i + 1]), tuple(uv_r[i])],
                           fill=color + (70,))
        bg = Image.alpha_composite(bg, layer)
        d = ImageDraw.Draw(bg)
        polyline(ctr, color + (255,), 3 * ss)
        wp = np.c_[xyyaw[:, :2], np.zeros(len(xyyaw))]
        uv, ok = proj(wp)
        dep = depth_of(wp)
        for p, o, z in zip(uv, ok, dep):
            if o:
                # gt: hollow ring; A (first prediction) larger than C, so overlapping paths stay readable
                r = (0.42 if label == "gt" else 0.30 if label == first_pred else 0.20) * K[0, 0] / z
                r = max(3 * ss, r)
                if label == "gt":
                    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=color + (255,), width=2 * ss)
                else:
                    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color + (255,),
                              outline=(0, 0, 0, 255), width=ss)

    out = bg.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    if title:
        ImageDraw.Draw(out).text((10, 8), title, fill=(220, 225, 232), font=_font(16))
    return out
