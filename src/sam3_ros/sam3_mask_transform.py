#!/usr/bin/env python3
# type: ignore
"""
SAM3 LiDAR boundary transform node.

Subscribes to:
  - /lidar_2d_projection  (PointCloud2 with x,y,z,u,v fields from transform node)
  - /sam3_left_contour    (sam2_msgs/DetectedRoadArea — left boundary 2D pixel coords)
  - /sam3_right_contour   (sam2_msgs/DetectedRoadArea — right boundary 2D pixel coords)

Publishes:
  - /sam3_left_boundary   (PointCloud2 — left boundary 3D LiDAR points)
  - /sam3_right_boundary  (PointCloud2 — right boundary 3D LiDAR points)

A contour describes an image captured several hundred milliseconds before the message
arrives (camera transport plus SAM3 inference), while the projection arrives ~50ms
behind its own capture. Pairing each contour with whatever projection is current at
arrival time therefore lifts stale pixels onto fresh geometry, and the boundary lands
wherever the vehicle was when the camera saw it — measured at ~1.1s of skew, roughly
5m of travel at 5m/s.

So the projections are buffered instead, and each contour is fused against the cloud
captured closest to its own header stamp. Skew drops to at most half a LiDAR period,
independent of how far behind the camera pipeline runs. The cost is that a boundary is
published ~one pipeline latency after the scene it describes; the output carries the
matched cloud's stamp so consumers can account for that. Stamps are only ever compared
to each other and never to the node clock, which keeps this valid under bag replay
without any use_sim_time configuration.
"""

import os
import time as _time
from collections import deque
from threading import Lock

from ament_index_python.packages import get_package_share_directory
import numpy as np
import ros2_numpy as ros_numpy
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import yaml
from scipy.spatial import KDTree

from perception_common.utils import is_zero_stamp, stamp_to_seconds
from sam2_msgs.msg import DetectedRoadArea

TOPICS_PATH = os.path.join(
    get_package_share_directory("perception_common"), "topics.yaml"
)

with open(TOPICS_PATH, "r") as f:
    _config = yaml.safe_load(f)

_topics = _config.get("topics", {})
_sam_topics = _topics.get("sam3") or {}
_transform_topics = _topics.get("transform") or {}

LIDAR_2D_PROJ_TOPIC = _transform_topics.get("lidar_2d_projection", "/lidar_2d_projection")
SAM3_LEFT_CONTOUR_TOPIC = _sam_topics.get("left_contour", "/sam3_left_contour")
SAM3_RIGHT_CONTOUR_TOPIC = _sam_topics.get("right_contour", "/sam3_right_contour")
SAM3_LEFT_BOUNDARY = _sam_topics.get("left_boundary", "/sam3_left_boundary")
SAM3_RIGHT_BOUNDARY = _sam_topics.get("right_boundary", "/sam3_right_boundary")

# KDTree pixel search radius (pixels)
PIXEL_LIM = 5


class BufferedProjection:
    """A buffered projection with its pixel KDTree built on first use.

    Left and right contours for one frame match the same cloud, so the parse and the
    tree build are shared between them, and clouds that no contour ever matches cost
    nothing beyond the memory to hold them.
    """

    __slots__ = ("msg", "stamp", "_tree", "_points", "_parsed")

    def __init__(self, msg: PointCloud2, stamp: float):
        self.msg = msg
        self.stamp = stamp
        self._tree = None
        self._points = None
        self._parsed = False

    def pixel_tree(self):
        """Return ``(kdtree_over_uv, xyz_points)``, or ``(None, None)`` if empty."""
        if not self._parsed:
            self._parsed = True
            cloud = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg)
            cloud = cloud[~np.isnan(cloud["x"])]
            if cloud.size:
                self._points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
                self._tree = KDTree(
                    np.column_stack(
                        (cloud["u"].astype(np.float32), cloud["v"].astype(np.float32))
                    )
                )
        return self._tree, self._points


class Sam3MaskTransform(Node):
    def __init__(self):
        super().__init__("sam3_mask_transform")

        lidar_topic = (
            self.declare_parameter("lidar_projection_topic", LIDAR_2D_PROJ_TOPIC)
            .get_parameter_value()
            .string_value
        )
        left_contour_topic = (
            self.declare_parameter("left_contour_topic", SAM3_LEFT_CONTOUR_TOPIC)
            .get_parameter_value()
            .string_value
        )
        right_contour_topic = (
            self.declare_parameter("right_contour_topic", SAM3_RIGHT_CONTOUR_TOPIC)
            .get_parameter_value()
            .string_value
        )
        left_boundary_topic = (
            self.declare_parameter("left_boundary_topic", SAM3_LEFT_BOUNDARY)
            .get_parameter_value()
            .string_value
        )
        right_boundary_topic = (
            self.declare_parameter("right_boundary_topic", SAM3_RIGHT_BOUNDARY)
            .get_parameter_value()
            .string_value
        )
        qos_depth = int(
            self.declare_parameter("qos_depth", 20)
            .get_parameter_value()
            .integer_value
        )
        # Half a 10Hz LiDAR period is the worst case for a dense buffer; the default
        # leaves headroom for jitter and the occasional dropped scan without ever
        # admitting a cloud from a neighbouring frame.
        max_pairing_skew = float(
            self.declare_parameter("max_pairing_skew", 0.08)
            .get_parameter_value()
            .double_value
        )
        # Must exceed the camera-to-contour latency, or the matching cloud will have
        # been evicted before its contour arrives.
        buffer_duration = float(
            self.declare_parameter("projection_buffer_duration", 2.0)
            .get_parameter_value()
            .double_value
        )
        # How long a boundary may go unrefreshed before the node publishes an empty
        # cloud. Without this a stalled segmenter leaves the last boundary standing in
        # RViz and in any consumer, which reads as valid, current geometry.
        boundary_timeout = float(
            self.declare_parameter("boundary_timeout", 0.5)
            .get_parameter_value()
            .double_value
        )

        self._max_pairing_skew = max(0.0, max_pairing_skew)
        self._buffer_duration = max(0.0, buffer_duration)
        self._boundary_timeout = max(0.0, boundary_timeout)

        self.publishers_by_side = {
            "left": self.create_publisher(PointCloud2, left_boundary_topic, 1),
            "right": self.create_publisher(PointCloud2, right_boundary_topic, 1),
        }

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self._buffer_lock = Lock()
        self._projections: deque[BufferedProjection] = deque()
        self._last_publish = {"left": 0.0, "right": 0.0}
        self._last_unmatched_log = 0.0
        self._matched_count = 0
        self._unmatched_count = 0

        # Sides are handled independently: SAM3 skips publishing a side when it finds
        # no points there, and one silent side must not hold back the other.
        self.create_subscription(
            DetectedRoadArea,
            left_contour_topic,
            lambda msg: self._on_contour(msg, "left"),
            qos_depth,
        )
        self.create_subscription(
            DetectedRoadArea,
            right_contour_topic,
            lambda msg: self._on_contour(msg, "right"),
            qos_depth,
        )
        self.create_subscription(PointCloud2, lidar_topic, self._on_projection, qos_depth)

        # Fixed period, so boundary_timeout is enforced to within 0.5s of granularity.
        self._watchdog_timer = self.create_timer(0.5, self._watchdog)

        # The pairing bound and the buffer depth both depend on measured pipeline
        # latency, so keep them settable at runtime for calibration against a
        # replaying bag without restarting the node.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"Sam3MaskTransform ready: {lidar_topic} + {left_contour_topic} + {right_contour_topic}, "
            f"max_pairing_skew={self._max_pairing_skew:.3f}s "
            f"projection_buffer_duration={self._buffer_duration:.3f}s "
            f"boundary_timeout={self._boundary_timeout:.3f}s qos_depth={qos_depth} "
            f"(stamp-matched pairing)"
        )

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Validate every parameter before applying any, so a rejected request cannot
        # leave the node half-updated.
        bounded = {
            "max_pairing_skew": "_max_pairing_skew",
            "projection_buffer_duration": "_buffer_duration",
            "boundary_timeout": "_boundary_timeout",
        }
        for p in params:
            if p.name in bounded and float(p.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be >= 0"
                )
        for p in params:
            if p.name in bounded:
                setattr(self, bounded[p.name], float(p.value))
                self.get_logger().info(f"{p.name} set to {float(p.value):.3f}s")
        return SetParametersResult(successful=True)

    def _on_projection(self, msg: PointCloud2) -> None:
        """Buffer the projection so a later contour can be paired against it."""
        if is_zero_stamp(msg.header):
            return
        stamp = stamp_to_seconds(msg.header.stamp)
        with self._buffer_lock:
            self._projections.append(BufferedProjection(msg, stamp))
            # Bag replay can jump backwards; drop the buffer rather than carrying
            # clouds from the previous run forward as match candidates.
            if self._projections[0].stamp > stamp:
                self._projections.clear()
                self._projections.append(BufferedProjection(msg, stamp))
            cutoff = stamp - self._buffer_duration
            while self._projections and self._projections[0].stamp < cutoff:
                self._projections.popleft()

    def _on_contour(self, msg: DetectedRoadArea, side: str) -> None:
        entry, skew = self._match_projection(msg.header)
        if entry is None:
            self._unmatched_count += 1
            self._log_unmatched(side, skew)
            return
        self._matched_count += 1
        self._publish_boundary(msg, entry, side)

    def _match_projection(self, header):
        """Return ``(projection, signed_skew)`` for the cloud nearest this contour.

        ``skew`` is positive when the matched cloud was captured after the contour's
        image, and ``float('inf')`` when there is nothing to match against.
        """
        if is_zero_stamp(header):
            return None, float("inf")
        stamp = stamp_to_seconds(header.stamp)

        with self._buffer_lock:
            candidates = list(self._projections)
        if not candidates:
            return None, float("inf")

        best = min(candidates, key=lambda entry: abs(entry.stamp - stamp))
        skew = best.stamp - stamp
        if abs(skew) <= self._max_pairing_skew:
            return best, skew
        return None, skew

    def _log_unmatched(self, side: str, skew: float) -> None:
        """Rate-limited warning so a starved or misaligned pipeline stays visible."""
        now = _time.monotonic()
        if now - self._last_unmatched_log < 1.0:
            return
        self._last_unmatched_log = now

        if skew == float("inf"):
            detail = "no projection buffered (or zero-stamped header)"
        elif skew > 0:
            # Everything still buffered was captured after this contour's image: the
            # cloud it should have paired with has already been evicted.
            detail = (
                f"nearest projection is {skew:.3f}s newer than the contour — "
                f"contour latency exceeds projection_buffer_duration="
                f"{self._buffer_duration:.3f}s"
            )
        else:
            # No cloud as recent as the contour's image has arrived, so the LiDAR
            # stream is stalled or running behind the camera.
            detail = (
                f"nearest projection is {-skew:.3f}s older than the contour — "
                f"no projection was captured near this image"
            )
        self.get_logger().warning(
            f"Unmatched {side} contour: {detail}; "
            f"max_pairing_skew={self._max_pairing_skew:.3f}s "
            f"matched={self._matched_count} unmatched={self._unmatched_count}"
        )

    def _publish_boundary(
        self,
        msg: DetectedRoadArea,
        entry: BufferedProjection,
        side: str,
    ) -> None:
        data = np.asarray(msg.road_area.data, dtype=np.float32)
        points_3d = np.empty((0, 3), dtype=np.float32)

        if data.size >= 2:
            tree, cloud_points = entry.pixel_tree()
            if tree is None:
                self.get_logger().warning("No points in projected PointCloud2")
            else:
                points_3d = self.find_matching_points_kdtree(
                    data.reshape(-1, 2), tree, cloud_points
                )

        # Published even when empty: an absent boundary must read as absent
        # downstream rather than as the last one that happened to work.
        self.create_cloud(points_3d, self.publishers_by_side[side], entry.msg.header)
        self._last_publish[side] = _time.monotonic()

    def find_matching_points_kdtree(
        self,
        boundary_points: np.ndarray,
        tree: KDTree,
        pc_arr: np.ndarray,
    ) -> np.ndarray:
        # Neighbouring contour pixels resolve to the same LiDAR returns, so the raw
        # match lists overlap heavily; de-duplicating keeps the published cloud the
        # size of the boundary rather than the size of the contour.
        hits = [m for m in tree.query_ball_point(boundary_points, PIXEL_LIM) if m]
        if not hits:
            return np.empty((0, 3), dtype=np.float32)
        idx = np.unique(np.concatenate([np.asarray(m, dtype=np.intp) for m in hits]))
        return pc_arr[idx]

    def _watchdog(self) -> None:
        """Publish empty boundaries while a side is not being refreshed."""
        if self._boundary_timeout <= 0.0:
            return
        with self._buffer_lock:
            newest = self._projections[-1] if self._projections else None
        if newest is None:
            return

        now = _time.monotonic()
        empty = np.empty((0, 3), dtype=np.float32)
        for side, last in self._last_publish.items():
            if now - last > self._boundary_timeout:
                self.create_cloud(empty, self.publishers_by_side[side], newest.msg.header)

    def create_cloud(
        self,
        points_3d: np.ndarray,
        publisher,
        header,
    ) -> None:
        intensity = np.ones((points_3d.shape[0], 1), dtype=np.float32)
        cloud_data = np.hstack((points_3d[:, :3], intensity))
        cloud_msg = pc2.create_cloud(header, self.fields, cloud_data)
        publisher.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Sam3MaskTransform()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
