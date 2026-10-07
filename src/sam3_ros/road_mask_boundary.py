"""Road-mask -> left/right boundary pixels, and road-mask -> 3D boundary points.

Kept free of rclpy, torch and cv2 (numpy + scipy.ndimage only) so it is unit-testable on any
host and shared by segmentation.py (2D contours) and sam3_mask_transform.py (3D lifting).

Why the legacy extraction fails near obstacles (segmentation._extract_and_publish_boundaries):
it keeps the LARGEST external contour and splits each row band at the band's MEAN x. A car on the
road makes a notch, so the contour runs around the car and the mean-x split hands the car's outline
to the boundary; a car that cuts off the far road leaves the far part as a separate, smaller
contour, which is discarded. The contour's top edge (the mask's horizon line) and its image-border
segments are split into left/right as well, though neither is a road edge.

`rowwise_boundary` replaces that with: fill holes, keep the component the ego vehicle is on, take
each row's OUTERMOST road pixels, and drop the horizon rows, image-border pixels and (optionally)
pixels adjacent to an obstacle mask.

`lift_mask_boundary` replaces contour lifting in sam3_mask_transform: instead of matching contour
pixels to LiDAR returns within a pixel radius (which near an edge also catches returns BEHIND it),
every projected ground return whose pixel is road becomes a road point, and the boundary is the
per-bin lateral extreme of those points in the vehicle frame.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

__all__ = ["ego_component", "rowwise_boundary", "lift_mask_boundary", "lidar_to_vehicle",
           "vehicle_to_lidar"]

_EIGHT = np.ones((3, 3), dtype=bool)


def ego_component(road_mask: np.ndarray, bridge_mask: np.ndarray | None = None,
                  seed_rows_frac: float = 0.15, seed_cols_frac: float = 0.3) -> np.ndarray:
    """The hole-filled road component the vehicle is on.

    Holes (a car, a person, a pothole) are filled: road continues under them. ``bridge_mask``
    (obstacles) joins components for the labelling only, so road beyond a vehicle that cuts the
    near road off from the far road stays connected; obstacle pixels are not returned as road
    unless enclosed by it. The component kept is the one with the most pixels in the bottom-centre
    seed window (the road just ahead of the vehicle); if none reaches the window, the one nearest
    its centre. Returns an empty mask when there is no road.
    """
    road = np.asarray(road_mask, dtype=bool)
    joined = road | np.asarray(bridge_mask, dtype=bool) if bridge_mask is not None else road
    m = ndimage.binary_fill_holes(joined)
    labels, n = ndimage.label(m, structure=_EIGHT)
    if n == 0:
        return np.zeros_like(m)
    h, w = m.shape
    r0 = int(h * (1.0 - seed_rows_frac))
    c0, c1 = int(w * (0.5 - seed_cols_frac / 2)), int(math.ceil(w * (0.5 + seed_cols_frac / 2)))
    window = labels[r0:, c0:c1]
    ids, counts = np.unique(window[window > 0], return_counts=True)
    if ids.size:
        keep = ids[np.argmax(counts)]
    else:
        ys, xs = np.nonzero(labels)
        d = (ys - (h - 1)) ** 2 + (xs - w / 2.0) ** 2
        keep = labels[ys[np.argmin(d)], xs[np.argmin(d)]]
    comp = labels == keep
    if bridge_mask is None:
        return comp
    # The kept component's road -- near and far parts alike, even when only an obstacle joins
    # them -- plus whatever that road encloses.
    return ndimage.binary_fill_holes(comp & road)


def rowwise_boundary(road_mask: np.ndarray, obstacle_mask: np.ndarray | None = None, *,
                     row_step: int = 2, horizon_frac: float = 0.05, border_px: int = 2,
                     obstacle_dilate_px: int = 6, min_row_width_px: int = 3):
    """Left / right boundary pixels ``(N,2)`` as (x, y) from a road mask.

    ``road_mask`` should already be the ego component (see :func:`ego_component`). Per sampled
    row: the leftmost and rightmost road pixel. Dropped:
      * rows in the top ``horizon_frac`` of the mask's vertical extent -- the far end of the road
        is a horizon line, not an edge;
      * points within ``border_px`` of the image border -- the field of view ends there, the road
        does not;
      * points with an obstacle within ``obstacle_dilate_px`` (when an obstacle mask is given) --
        the road there ends at a vehicle, not at its edge.
    """
    m = np.asarray(road_mask, dtype=bool)
    h, w = m.shape
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    top, bottom = rows[0], rows[-1]
    first = top + int(math.ceil(horizon_frac * (bottom - top + 1)))
    sample = np.arange(first, bottom + 1, max(1, int(row_step)))
    sample = sample[m[sample].any(axis=1)]
    if sample.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    sub = m[sample]
    left_x = np.argmax(sub, axis=1)
    right_x = w - 1 - np.argmax(sub[:, ::-1], axis=1)
    ok = (right_x - left_x + 1) >= min_row_width_px
    ok_l = ok & (left_x >= border_px)
    ok_r = ok & (right_x <= w - 1 - border_px)
    if obstacle_mask is not None and np.asarray(obstacle_mask).any():
        ob = ndimage.binary_dilation(np.asarray(obstacle_mask, dtype=bool),
                                     structure=_EIGHT, iterations=max(1, obstacle_dilate_px))
        ok_l &= ~ob[sample, left_x]
        ok_r &= ~ob[sample, right_x]
    left = np.column_stack((left_x[ok_l], sample[ok_l])).astype(np.float32)
    right = np.column_stack((right_x[ok_r], sample[ok_r])).astype(np.float32)
    return left, right


def _rot(yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]])


def lidar_to_vehicle(xy, ego_yaw_deg: float) -> np.ndarray:
    """Rotate (N,2) lidar_tc points into the vehicle frame (lidar_tc is yawed ``ego_yaw_deg``)."""
    return np.atleast_2d(np.asarray(xy, float)) @ _rot(-math.radians(ego_yaw_deg)).T


def vehicle_to_lidar(xy, ego_yaw_deg: float) -> np.ndarray:
    return np.atleast_2d(np.asarray(xy, float)) @ _rot(math.radians(ego_yaw_deg)).T


def lift_mask_boundary(xyz, u, v, road_px, image_wh, *, ground=None, obstacle_px=None,
                       ego_yaw_deg: float = -5.35, x_min: float = 5.0, x_max: float = 60.0,
                       bin_len: float = 2.0, min_bin_points: int = 3,
                       max_above_road: float = 0.3, obstacle_dilate_px: int = 6,
                       require_evidence: bool = True, edge_evidence_m: float = 1.0,
                       border_frac: float = 0.02):
    """Boundary points (lidar_tc xyz) from a road mask lifted onto a LiDAR projection.

    ``xyz, u, v`` are the projection's points and pixel coordinates (full image resolution,
    ``image_wh``); ``road_px`` is the road mask at any resolution. A return is road when its pixel
    is road and it is on the ground: the ``ground`` flag when the projection carries one, else no
    more than ``max_above_road`` above the bin's median road height (a road pixel over a car's
    lower body otherwise lifts onto the car). Per fixed vehicle-frame x-bin the lateral extremes
    are the boundary; an extreme whose pixel touches the (dilated) obstacle mask is dropped.

    With ``require_evidence`` an extreme is kept only when a projected return whose pixel is NOT
    road lies within ``edge_evidence_m`` beyond it in the same bin. Without it a bin holding one
    ground-ring arc reports where the arc ends (measured on the Sep 22 replays: a "right edge"
    at +1.4..+2.6 m at 20-30 m, on the wrong side of the road), and returns at the edge of the
    camera's field of view -- where nothing is projected beyond -- read as road edges.
    An extreme whose pixel lies within ``border_frac`` of the image's left or right border is
    where the field of view ends, not the road, and is dropped as well.
    Returns ``(left, right)`` as (N,3) float32 in lidar_tc.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    empty = (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32))
    if xyz.shape[0] == 0:
        return empty
    road_px = np.asarray(road_px, dtype=bool)
    mh, mw = road_px.shape
    W, H = image_wh
    ui = np.clip((np.asarray(u) * (mw / W)).astype(np.int64), 0, mw - 1)
    vi = np.clip((np.asarray(v) * (mh / H)).astype(np.int64), 0, mh - 1)
    road = road_px[vi, ui]
    if ground is not None:
        road &= np.asarray(ground, dtype=bool)
    if not road.any():
        return empty
    other = None
    if require_evidence:
        # Evidence must be non-road GROUND: a pedestrian or a car standing on the road is not
        # road in the mask either, and would otherwise "confirm" an edge at their feet.
        not_road = ~road_px[vi, ui]
        if ground is not None:
            not_road &= np.asarray(ground, dtype=bool)
        other = lidar_to_vehicle(xyz[not_road, :2], ego_yaw_deg)
        other_z = xyz[not_road, 2]
    pts, ui, vi = xyz[road], ui[road], vi[road]
    veh = lidar_to_vehicle(pts[:, :2], ego_yaw_deg)
    nb = int(math.floor((x_max - x_min) / bin_len))
    b = np.floor((veh[:, 0] - x_min) / bin_len).astype(np.int64)
    ob_bin = (np.floor((other[:, 0] - x_min) / bin_len).astype(np.int64)
              if other is not None else None)
    ob = None
    if obstacle_px is not None and np.asarray(obstacle_px).any():
        ob = ndimage.binary_dilation(np.asarray(obstacle_px, dtype=bool), structure=_EIGHT,
                                     iterations=max(1, obstacle_dilate_px))
    left, right = [], []
    for i in range(nb):
        k = np.flatnonzero(b == i)
        if k.size < min_bin_points:
            continue
        if ground is None:
            zmed = np.median(pts[k, 2])
            k = k[pts[k, 2] <= zmed + max_above_road]
            if k.size < min_bin_points:
                continue
        oth_y = None
        if other is not None:
            sel = ob_bin == i
            if ground is None:      # no ground flag: "ground" = not above this bin's road
                sel &= other_z <= np.median(pts[k, 2]) + max_above_road
            oth_y = other[sel, 1]
        for out, j, sgn in ((left, k[np.argmax(veh[k, 1])], 1.0),
                            (right, k[np.argmin(veh[k, 1])], -1.0)):
            if ob is not None and ob[vi[j], ui[j]]:
                continue
            if not (border_frac * mw <= ui[j] <= (1.0 - border_frac) * mw):
                continue
            if oth_y is not None:
                d = sgn * (oth_y - veh[j, 1])
                if not ((d > 0.0) & (d <= edge_evidence_m)).any():
                    continue
            out.append(pts[j])
    as_arr = lambda a: np.asarray(a, np.float32).reshape(-1, 3)  # noqa: E731
    return as_arr(left), as_arr(right)
