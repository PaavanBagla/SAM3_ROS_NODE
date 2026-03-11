import copy
import threading
import time

import cv2
import cv_bridge
import geometry_msgs.msg
import numpy
import rclpy.callback_groups
import rclpy.node
import rclpy.qos
import sam3
import sam3.model.sam3_image_processor
import sensor_msgs.msg
import std_msgs.msg
import torch
import vision_msgs.msg


class Segmentation(rclpy.node.Node):
    class Result:
        def __init__(
            self,
            header: std_msgs.msg.Header,
            image: numpy.ndarray,
            masks: torch.Tensor,
            boxes: torch.Tensor,
            scores: torch.Tensor,
        ) -> None:
            self.header = header
            self.image = image
            self.masks = masks
            self.boxes = boxes
            self.scores = scores

    def __init__(self) -> None:
        super().__init__("segmentation")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_compressed_image = (
            self.declare_parameter("use_compressed_image", True).get_parameter_value().bool_value
        )
        self.text_prompt = (
            self.declare_parameter("text_prompt", "").get_parameter_value().string_value
        )
        confidence_threshold = (
            self.declare_parameter("confidence_threshold", 0.5).get_parameter_value().double_value
        )
        result_visualization_rate = (
            self.declare_parameter("result_visualization_rate", 10.0)
            .get_parameter_value()
            .double_value
        )

        self.model = sam3.build_sam3_image_model()
        self.model.to(self.device)

        self.processor = sam3.model.sam3_image_processor.Sam3Processor(
            self.model, confidence_threshold=confidence_threshold
        )

        self.result_lock = threading.Lock()
        self.result: Segmentation.Result | None = None

        self.colors = torch.tensor(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
            ],
            dtype=torch.uint8,
        ).to(self.device)

        self.cv_bridge = cv_bridge.CvBridge()

        self.mask_image_pub = self.create_publisher(
            sensor_msgs.msg.CompressedImage, "~/mask_image/compressed", 1
        )
        self.detections_pub = self.create_publisher(
            vision_msgs.msg.Detection2DArray, "~/detections", 1
        )

        if use_compressed_image:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                "image/compressed",
                self._image_compressed_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
        else:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                "image",
                self._image_callback,
                rclpy.qos.qos_profile_sensor_data,
            )

        if result_visualization_rate > 0.0:
            self.result_image_pub = self.create_publisher(
                sensor_msgs.msg.CompressedImage, "~/result_image/compressed", 1
            )
            self.visualize_result_timer = self.create_timer(
                1.0 / result_visualization_rate,
                self._visualize_result,
                rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
            )

    def _image_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage) -> None:
        image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        self._run_inference(image, msg.header)

    def _image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        image = self.cv_bridge.imgmsg_to_cv2(msg)
        self._run_inference(image, msg.header)

    def _run_inference(self, image: numpy.ndarray, header: std_msgs.msg.Header) -> None:
        start = time.time_ns()
        inference_state = self.processor.set_image(
            torch.from_numpy(image.transpose(2, 0, 1)).to(self.device)
        )
        output = self.processor.set_text_prompt(self.text_prompt, inference_state)
        self.get_logger().debug(f"Inference time: {(time.time_ns() - start) / 1e6:.3f} ms")

        with self.result_lock:
            self.result = Segmentation.Result(
                header,
                image,
                output["masks"],
                output["boxes"],
                output["scores"],
            )

        mask_image = (
            (
                self.result.masks
                * torch.arange(1, self.result.masks.shape[0] + 1).to(self.device).view(-1, 1, 1, 1)
            )
            .to(torch.uint64)
            .sum(dim=0)
        )

        detections = vision_msgs.msg.Detection2DArray()

        for box, score in zip(self.result.boxes, self.result.scores):
            x1, y1, x2, y2 = box.float().tolist()
            detection = vision_msgs.msg.Detection2D()
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            detection.bbox.center.position.x = (x1 + x2) / 2
            detection.bbox.center.position.y = (y1 + y2) / 2
            detection.results.append(  # pyright: ignore[reportAttributeAccessIssue]
                vision_msgs.msg.ObjectHypothesisWithPose(
                    hypothesis=vision_msgs.msg.ObjectHypothesis(
                        class_id=self.text_prompt,
                        score=score.item(),
                    ),
                    pose=geometry_msgs.msg.PoseWithCovariance(
                        pose=geometry_msgs.msg.Pose(
                            position=geometry_msgs.msg.Point(
                                x=detection.bbox.center.position.x,
                                y=detection.bbox.center.position.y,
                                z=0.0,
                            )
                        ),
                    ),
                )
            )
            detections.detections.append(detection)  # pyright: ignore[reportAttributeAccessIssue]

        mask_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(
            mask_image.cpu().numpy().transpose(1, 2, 0)
        )
        mask_image_msg.header = header
        self.mask_image_pub.publish(mask_image_msg)

        detections.header = header
        self.detections_pub.publish(detections)

    def _visualize_result(self) -> None:
        with self.result_lock:
            if self.result is None:
                return
            result = copy.deepcopy(self.result)

        if result.masks.nelement() == 0:
            result_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(result.image)
            result_image_msg.header = result.header
            self.result_image_pub.publish(result_image_msg)
            return

        colored_masks = (
            (
                result.masks
                * self.colors[torch.arange(result.masks.shape[0]) % self.colors.shape[0]].reshape(
                    -1, 3, 1, 1
                )
            )
            .max(dim=0)
            .values.to(torch.uint8)
        )

        result.image = cv2.addWeighted(
            src1=result.image,
            src2=colored_masks.permute(1, 2, 0).cpu().numpy(),
            alpha=1.0,
            beta=0.5,
            gamma=0,
        )

        box_thickness = 3
        font_scale = 2.0
        text_color = (255, 255, 255)
        text_thickness = 2

        for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
            x1, y1, x2, y2 = box.int().tolist()
            text = f"{score.item():.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=text_thickness,
            )
            text_ys = (
                (y1 - text_height - 5, y1) if y1 - text_height - 5 > 0 else (y1, y1 + text_height)
            )
            box_color = self.colors[i % self.colors.nelement()].tolist()

            cv2.rectangle(
                result.image,
                (x1, y1),
                (x2, y2),
                color=box_color,
                thickness=box_thickness,
            )
            cv2.rectangle(
                result.image,
                (x1, text_ys[0]),
                (x1 + text_width, text_ys[1]),
                color=box_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                result.image,
                text,
                (x1, text_ys[0] + text_height),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=text_color,
                thickness=text_thickness,
            )

        result_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(result.image)
        result_image_msg.header = result.header
        self.result_image_pub.publish(result_image_msg)


def main(args=None):
    rclpy.init(args=args)
    segmentation = Segmentation()
    rclpy.spin(segmentation)
    segmentation.destroy_node()
    rclpy.shutdown()
