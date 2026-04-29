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

Logic is a direct port of samv2_mask_transform.py adapted for SAM3 topics.
"""

import os
import threading
import time as _time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import ros2_numpy as ros_numpy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import yaml
from scipy.spatial import KDTree

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
        sync_queue_size = int(
            self.declare_parameter("sync_queue_size", 20)
            .get_parameter_value()
            .integer_value
        )
        sync_slop = float(
            self.declare_parameter("sync_slop", 0.6)
            .get_parameter_value()
            .double_value
        )

        self.left_boundary_pub = self.create_publisher(PointCloud2, left_boundary_topic, 1)
        self.right_boundary_pub = self.create_publisher(PointCloud2, right_boundary_topic, 1)

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self._sync_slop = max(0.0, sync_slop)
        self._cache_lock = threading.Lock()
        self._last_left: DetectedRoadArea | None = None
        self._last_right: DetectedRoadArea | None = None
        self._last_left_recv: float = 0.0
        self._last_right_recv: float = 0.0

        self.create_subscription(DetectedRoadArea, left_contour_topic, self._left_cb, sync_queue_size)
        self.create_subscription(DetectedRoadArea, right_contour_topic, self._right_cb, sync_queue_size)
        self.create_subscription(PointCloud2, lidar_topic, self._lidar_cb, sync_queue_size)

        self.get_logger().info(
            f"Sam3MaskTransform ready: {lidar_topic} + {left_contour_topic} + {right_contour_topic}, "
            f"slop={sync_slop:.3f}s queue={sync_queue_size} (arrival-time sync)"
        )

    def _left_cb(self, msg: DetectedRoadArea) -> None:
        with self._cache_lock:
            self._last_left = msg
            self._last_left_recv = _time.monotonic()

    def _right_cb(self, msg: DetectedRoadArea) -> None:
        with self._cache_lock:
            self._last_right = msg
            self._last_right_recv = _time.monotonic()

    def _lidar_cb(self, msg_proj: PointCloud2) -> None:
        now = _time.monotonic()
        with self._cache_lock:
            left = self._last_left
            right = self._last_right
            left_age = now - self._last_left_recv
            right_age = now - self._last_right_recv
        if left is None or right is None:
            return
        if left_age > self._sync_slop or right_age > self._sync_slop:
            return
        self.callback(msg_proj, left, right)

    def callback(
        self,
        msg_proj: PointCloud2,
        msg_left: DetectedRoadArea,
        msg_right: DetectedRoadArea,
    ) -> None:
        cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg_proj)
        valid = ~np.isnan(cloud["x"])
        cloud = cloud[valid]
        if cloud.size == 0:
            self.get_logger().warning("No points in projected PointCloud2")
            return

        pc_arr = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
        u = cloud["u"].astype(np.float32)
        v = cloud["v"].astype(np.float32)

        left_data = np.array(msg_left.road_area.data, dtype=np.float32)
        right_data = np.array(msg_right.road_area.data, dtype=np.float32)

        if left_data.size >= 2:
            left_pts = left_data.reshape(-1, 2)
            left_3d = self.find_matching_points_kdtree(left_pts, u, v, pc_arr)
            if left_3d.size > 0:
                self.create_cloud(left_3d, self.left_boundary_pub, msg_proj.header)

        if right_data.size >= 2:
            right_pts = right_data.reshape(-1, 2)
            right_3d = self.find_matching_points_kdtree(right_pts, u, v, pc_arr)
            if right_3d.size > 0:
                self.create_cloud(right_3d, self.right_boundary_pub, msg_proj.header)

    def find_matching_points_kdtree(
        self,
        boundary_points: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        pc_arr: np.ndarray,
    ) -> np.ndarray:
        tree = KDTree(np.column_stack((u, v)))
        idx: list[int] = []
        for contour_point in boundary_points:
            matches = tree.query_ball_point(contour_point, PIXEL_LIM)
            idx.extend(matches)
        return pc_arr[np.array(idx)] if idx else np.empty((0, 3))

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
