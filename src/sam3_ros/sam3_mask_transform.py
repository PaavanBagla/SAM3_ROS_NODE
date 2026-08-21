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
matched cloud's stamp so consumers can account for that.

Matching is two-sided: a contour with no cloud yet captured at or after it is deferred for
up to wait_for_newer rather than forced backwards onto an older cloud. Deferred contours are
resolved by _pump, which runs after every buffered cloud and on a short timer.

Message stamps are only ever compared to each other, so the pairing itself is valid under bag
replay regardless of use_sim_time. Deferral expiry and the watchdog do read the node clock, so
run with use_sim_time:=true against a bag if you want them to track playback.

The pairing itself lives in perception_common.stamp_sync, shared with the YOLO fusion node and
the CLRerNet lane transform. This node used to carry its own copy; the two drifted, which is
what the shared module exists to prevent.
"""

import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import yaml

from perception_common.stamp_sync import (
    DEFAULT_PIXEL_LIM,
    DEFERRED,
    StampMatchedBuffer,
    apply_bounded_parameters,
)
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
PIXEL_LIM = DEFAULT_PIXEL_LIM

# The two contour streams are paired independently, so a side that publishes nothing cannot
# hold back the other. They still share one projection buffer, which is what keeps the parse
# and the KDTree of a frame shared between its left and right contour.
SIDES = ("left", "right")


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
        #
        # This MUST stay above the SAM3 frame period, or the watchdog stops reporting "the
        # segmenter died" and starts reporting "the segmenter is slow". Measured 2026-08-17:
        # SAM3 inference runs 0.5-0.8s per frame (~1.5Hz) with the five-term OR prompt, and
        # at the old 0.5s default the watchdog fired between consecutive real contours --
        # 28% of everything published on /sam3_*_boundary was a spurious empty, so the road
        # boundary blinked out on more than a quarter of frames. 2.0s clears a ~0.65s period
        # with room for a missed frame. Raise it further if the prompt list grows.
        boundary_timeout = float(
            self.declare_parameter("boundary_timeout", 2.0)
            .get_parameter_value()
            .double_value
        )
        # How long a contour may wait for a cloud captured at or after it. Without this,
        # matching can only reach backwards and skew doubles. Set to 0.0 to restore the old
        # one-sided behaviour at runtime, which is the rollback if this ever starves.
        wait_for_newer = float(
            self.declare_parameter("wait_for_newer", 0.06)
            .get_parameter_value()
            .double_value
        )
        # Deferrals are almost always resolved by the _pump that follows each buffered cloud;
        # this timer only bounds the wait when the projection stream stalls. Read-only because
        # a timer period cannot be changed after construction, and silently ignoring a
        # ros2 param set would be worse than rejecting it.
        pump_period = float(
            self.declare_parameter(
                "deferral_pump_period", 0.02, ParameterDescriptor(read_only=True)
            )
            .get_parameter_value()
            .double_value
        )
        stats_log_period = float(
            self.declare_parameter(
                "stats_log_period", 5.0, ParameterDescriptor(read_only=True)
            )
            .get_parameter_value()
            .double_value
        )

        self._boundary_timeout = max(0.0, boundary_timeout)
        self._projections = StampMatchedBuffer(
            "projection",
            buffer_duration=max(0.0, buffer_duration),
            max_skew=max(0.0, max_pairing_skew),
            wait_for_newer=max(0.0, wait_for_newer),
        )

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

        # None until the first publish on that side: under sim time the node clock reads 0
        # until /clock arrives, and 0.0 here would make the first watchdog tick see a
        # ~1.7e9s gap and fire immediately.
        self._last_publish = {side: None for side in SIDES}
        self._last_unmatched_log = None

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
        if pump_period > 0.0:
            self._pump_timer = self.create_timer(pump_period, self._pump)
        if stats_log_period > 0.0:
            self._stats_timer = self.create_timer(stats_log_period, self._log_stats)

        # The pairing bound and the buffer depth both depend on measured pipeline
        # latency, so keep them settable at runtime for calibration against a
        # replaying bag without restarting the node.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"Sam3MaskTransform ready: {lidar_topic} + {left_contour_topic} + {right_contour_topic}, "
            f"max_pairing_skew={self._projections.max_skew:.3f}s "
            f"projection_buffer_duration={self._projections.buffer_duration:.3f}s "
            f"wait_for_newer={self._projections.wait_for_newer:.3f}s "
            f"boundary_timeout={self._boundary_timeout:.3f}s qos_depth={qos_depth} "
            f"use_sim_time={self.get_parameter('use_sim_time').value} "
            f"(stamp-matched pairing)"
        )

    def _now(self) -> float:
        """Seconds on the node clock -- sim time when use_sim_time is set, else wall time."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_set_parameters(self, params) -> SetParametersResult:
        targets = {
            "max_pairing_skew": (self._projections, "max_skew"),
            "projection_buffer_duration": (self._projections, "buffer_duration"),
            "boundary_timeout": (self, "_boundary_timeout"),
            "wait_for_newer": (self._projections, "wait_for_newer"),
        }
        ok, reason, applied = apply_bounded_parameters(params, targets)
        if not ok:
            return SetParametersResult(successful=False, reason=reason)
        for name, value in applied:
            self.get_logger().info(f"{name} set to {value:.3f}s")
        return SetParametersResult(successful=True)

    def _on_projection(self, msg: PointCloud2) -> None:
        """Buffer the projection, then release any contour that was waiting for it."""
        self._projections.add(msg)
        # This is what resolves nearly every deferral, in the same tick the awaited cloud
        # lands; the pump timer only matters when this stream stalls.
        self._pump()

    def _on_contour(self, msg: DetectedRoadArea, side: str) -> None:
        # key=side keeps the two sides' pending queues independent, so a silent side never
        # blocks the other, while both still match against one shared cloud buffer.
        pairing = self._projections.match(
            msg.header, now=self._now(), payload=msg, key=side
        )
        if pairing.outcome is not DEFERRED:
            self._complete(pairing)

    def _pump(self) -> None:
        for pairing in self._projections.drain(self._now()):
            self._complete(pairing)

    def _complete(self, pairing) -> None:
        if pairing.value is None:
            self._log_unmatched(pairing.key, pairing.skew, pairing.reason)
            return
        self._publish_boundary(pairing.payload, pairing.value, pairing.key)

    def _log_unmatched(self, side: str, skew: float, reason=None) -> None:
        """Rate-limited warning so a starved or misaligned pipeline stays visible."""
        now = self._now()
        if self._last_unmatched_log is not None and now - self._last_unmatched_log < 1.0:
            return
        self._last_unmatched_log = now
        self.get_logger().warning(
            f"Unmatched {side} contour: "
            f"{self._projections.describe_unmatched(skew, 'contour', reason=reason)}; "
            f"{self._projections.status()}"
        )

    def _log_stats(self) -> None:
        """Periodic pairing health, so a skew regression is visible without instrumentation."""
        self.get_logger().info(f"pairing: {self._projections.status()}")

    def _publish_boundary(
        self,
        msg: DetectedRoadArea,
        entry,
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
        self.create_cloud(points_3d, self.publishers_by_side[side], entry.header)
        self._last_publish[side] = self._now()

    def find_matching_points_kdtree(
        self,
        boundary_points: np.ndarray,
        tree,
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
        newest = self._projections.newest()
        if newest is None:
            return

        now = self._now()
        empty = np.empty((0, 3), dtype=np.float32)
        for side, last in self._last_publish.items():
            if last is None:
                continue
            if now - last > self._boundary_timeout:
                self.create_cloud(empty, self.publishers_by_side[side], newest.header)

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
