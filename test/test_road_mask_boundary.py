"""road_mask_boundary on synthetic masks and a synthetic ground scene."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sam3_ros"))

from road_mask_boundary import (  # noqa: E402
    ego_component, lift_mask_boundary, lidar_to_vehicle, rowwise_boundary, vehicle_to_lidar,
)

H, W = 240, 320


def road_trapezoid():
    """Perspective road: 300 px wide at the bottom row, 40 px at the top (row 80)."""
    m = np.zeros((H, W), bool)
    for r in range(80, H):
        f = (r - 80) / (H - 1 - 80)
        half = 20 + f * 130
        m[r, int(W / 2 - half):int(W / 2 + half)] = True
    return m


def edges(m):
    rows = np.flatnonzero(m.any(1))
    return {r: (np.argmax(m[r]), W - 1 - np.argmax(m[r, ::-1])) for r in rows}


def test_car_inside_the_road_is_filled_not_traced():
    m = road_trapezoid()
    car = np.zeros_like(m)
    car[150:180, 140:180] = True
    m_obs = m & ~car                                   # the road mask has a hole where the car is
    left, right = rowwise_boundary(ego_component(m_obs))
    truth = edges(m)
    for x, y in left:
        assert x == truth[int(y)][0]
    for x, y in right:
        assert x == truth[int(y)][1]


def test_car_on_the_edge_is_dropped_with_an_obstacle_mask():
    m = road_trapezoid()
    car = np.zeros_like(m)
    car[150:180, 200:300] = True                       # covers the right edge in these rows
    m_obs = m & ~car
    _, right_no = rowwise_boundary(ego_component(m_obs))
    assert ((right_no[:, 1] >= 150) & (right_no[:, 1] < 180)).any()      # pulled in, kept
    _, right = rowwise_boundary(ego_component(m_obs, bridge_mask=car), obstacle_mask=car)
    assert not ((right[:, 1] >= 150) & (right[:, 1] < 180)).any()        # dropped
    assert ((right[:, 1] >= 185)).sum() > 10                             # rest of the edge kept


def test_far_road_beyond_a_crossing_vehicle_is_kept_when_bridged():
    m = road_trapezoid()
    truck = np.zeros_like(m)
    truck[120:135, :] = True                           # spans the full width
    m_obs = m & ~truck
    near_only = ego_component(m_obs)
    assert not near_only[100].any()                    # far road lost without the bridge
    bridged = ego_component(m_obs, bridge_mask=truck)
    assert bridged[100].any() and not bridged[125].any()


def test_horizon_and_image_border_are_not_edges():
    m = np.zeros((H, W), bool)
    m[100:, :] = True                                  # road filling the full image width
    left, right = rowwise_boundary(m)
    assert left.size == 0 and right.size == 0         # every row touches the border
    m2 = road_trapezoid()
    left, _ = rowwise_boundary(m2, horizon_frac=0.05)
    top = 80 + math.ceil(0.05 * (H - 80))
    assert left[:, 1].min() >= top


def test_detached_road_patch_is_ignored():
    m = road_trapezoid()
    m[90:110, 0:30] = True                             # road-labelled patch off to the side
    comp = ego_component(m)
    assert not comp[95, 10]


def _project(xyz, f=300.0, cam_h=1.5):
    """Simple forward-looking pinhole at the lidar origin (x forward, y left, z up)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = W / 2 - f * y / x
    v = H / 2 - f * (z + 0.0) / x
    return u, v


def test_lift_mask_boundary_finds_the_road_edge_in_the_vehicle_frame():
    yaw = -5.35
    gx, gy = np.meshgrid(np.arange(6.0, 50.0, 0.25), np.arange(-8.0, 8.0, 0.1))
    veh = np.column_stack((gx.ravel(), gy.ravel()))
    lid = vehicle_to_lidar(veh, yaw)
    xyz = np.column_stack((lid, np.full(len(lid), -2.4)))
    # A parked car standing on the road edge: its body is above the ground.
    car_v = np.column_stack((np.full(40, 20.0), np.linspace(2.5, 3.5, 40)))
    car_l = vehicle_to_lidar(car_v, yaw)
    xyz = np.vstack((xyz, np.column_stack((car_l, np.full(40, -1.2)))))
    u, v = _project(xyz)
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (xyz[:, 0] > 0)
    xyz, u, v = xyz[keep], u[keep], v[keep]
    # Ground behind the car is hidden from a sensor at (nearly) the camera's viewpoint.
    is_car = xyz[:, 2] > -2.0
    car_px = set(zip(u[is_car].astype(int), v[is_car].astype(int)))
    hidden = np.array([(not c) and (int(a), int(b)) in car_px and x > 20.0
                       for a, b, x, c in zip(u, v, xyz[:, 0], is_car)])
    xyz, u, v = xyz[~hidden], u[~hidden], v[~hidden]
    # Road mask: the pixels of every point within |y_vehicle| < 3.5, plus the car's pixels
    # (a road pixel bleeding onto the car's lower body).
    veh_y = lidar_to_vehicle(xyz[:, :2], yaw)[:, 1]
    road_px = np.zeros((H, W), bool)
    on = (np.abs(veh_y) < 3.5)
    road_px[v[on].astype(int), u[on].astype(int)] = True
    left, right = lift_mask_boundary(xyz, u, v, road_px, (W, H), ego_yaw_deg=yaw,
                                     x_min=6.0, x_max=40.0, bin_len=2.0)
    lv = lidar_to_vehicle(left[:, :2], yaw)
    rv = lidar_to_vehicle(right[:, :2], yaw)
    assert len(lv) >= 10 and len(rv) >= 10
    # Within a pixel of the edge at range (1 px = x / 300 m here); the car body is not lifted.
    assert np.all(np.abs(lv[:, 1] - 3.5) < 0.3)
    assert np.all(np.abs(rv[:, 1] + 3.5) < 0.3)


def test_lift_drops_ring_arcs_and_field_of_view_ends():
    yaw = -5.35
    gx, gy = np.meshgrid(np.arange(6.0, 50.0, 0.25), np.arange(-8.0, 8.0, 0.1))
    veh = np.column_stack((gx.ravel(), gy.ravel()))
    # Beyond 30 m only an arc of each ring reaches the right half of the road.
    keep = (veh[:, 0] < 30.0) | (veh[:, 1] > -0.5)
    veh = veh[keep]
    lid = vehicle_to_lidar(veh, yaw)
    xyz = np.column_stack((lid, np.full(len(lid), -2.4)))
    u, v = _project(xyz)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    xyz, u, v = xyz[ok], u[ok], v[ok]
    vy = lidar_to_vehicle(xyz[:, :2], yaw)[:, 1]
    road_px = np.zeros((H, W), bool)
    on = np.abs(vy) < 3.5
    road_px[v[on].astype(int), u[on].astype(int)] = True
    _, right_no = lift_mask_boundary(xyz, u, v, road_px, (W, H), ego_yaw_deg=yaw,
                                     x_min=6.0, x_max=46.0, require_evidence=False)
    assert lidar_to_vehicle(right_no[:, :2], yaw)[:, 1].max() > -1.0     # arc end read as edge
    _, right = lift_mask_boundary(xyz, u, v, road_px, (W, H), ego_yaw_deg=yaw,
                                  x_min=6.0, x_max=46.0)
    ry = lidar_to_vehicle(right[:, :2], yaw)[:, 1]
    assert len(ry) >= 5 and np.all(np.abs(ry + 3.5) < 0.3)              # only real edges kept
