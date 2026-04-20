import threading
import time
import traceback

import cv2
import cv_bridge
import geometry_msgs.msg
import numpy
import rclpy
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
        self.model_dtype = torch.float32

        use_compressed_image = (
            self.declare_parameter("use_compressed_image", True)
            .get_parameter_value()
            .bool_value
        )
        self.text_prompt = (
            self.declare_parameter("text_prompt", "")
            .get_parameter_value()
            .string_value
        )
        confidence_threshold = (
            self.declare_parameter("confidence_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        result_visualization_rate = (
            self.declare_parameter("result_visualization_rate", 10.0)
            .get_parameter_value()
            .double_value
        )

        self.get_logger().info(f"Segmentation node starting on device: {self.device}")
        self.get_logger().info(
            f"use_compressed_image={use_compressed_image}, "
            f"text_prompt='{self.text_prompt}', "
            f"confidence_threshold={confidence_threshold}, "
            f"result_visualization_rate={result_visualization_rate}"
        )

        if self.text_prompt == "":
            self.get_logger().warn("text_prompt is empty; model may return empty detections.")

        self.get_logger().info("Building SAM3 model...")
        model_build_start = time.time()
        self.model = sam3.build_sam3_image_model()
        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()

        first_param = next(self.model.parameters())
        self.get_logger().info(
            f"Model loaded in {time.time() - model_build_start:.2f}s, "
            f"device={first_param.device}, dtype={first_param.dtype}"
        )

        self.get_logger().info("Creating processor...")
        self.processor = sam3.model.sam3_image_processor.Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold,
        )
        self.get_logger().info("Processor ready.")

        self.result_lock = threading.Lock()
        self.result: Segmentation.Result | None = None
        self.received_frame_count = 0
        self.inference_count = 0

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
            device=self.device,
        )

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
            self.get_logger().info("Subscribed to topic: image/compressed")
        else:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                "image",
                self._image_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
            self.get_logger().info("Subscribed to topic: image")

        if result_visualization_rate > 0.0:
            self.result_image_pub = self.create_publisher(
                sensor_msgs.msg.CompressedImage, "~/result_image/compressed", 1
            )
            self.visualize_result_timer = self.create_timer(
                1.0 / result_visualization_rate,
                self._visualize_result,
                rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
            )
            self.get_logger().info(
                f"Visualization timer started at {result_visualization_rate:.2f} Hz"
            )

        self.get_logger().info("Segmentation node ready. Waiting for images...")

    def _normalize_input_image(self, image: numpy.ndarray) -> numpy.ndarray:
        if image is None:
            raise ValueError("Input image is None")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.dtype != numpy.uint8:
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)

        return numpy.ascontiguousarray(image)

    def _image_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage) -> None:
        self.received_frame_count += 1
        self.get_logger().info(
            f"Received compressed image #{self.received_frame_count}, format={msg.format}"
        )
        try:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image = self._normalize_input_image(image)
            self.get_logger().info(
                f"Decoded compressed image #{self.received_frame_count}, "
                f"shape={image.shape}, dtype={image.dtype}"
            )
            self._run_inference(image, msg.header)
        except Exception as e:
            self.get_logger().error(
                f"Exception in _image_compressed_callback: {type(e).__name__}: {e}"
            )
            self.get_logger().error(traceback.format_exc())

    def _image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        self.received_frame_count += 1
        self.get_logger().info(
            f"Received raw image #{self.received_frame_count}, encoding={msg.encoding}"
        )
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image = self._normalize_input_image(image)
            self.get_logger().info(
                f"Decoded raw image #{self.received_frame_count}, "
                f"shape={image.shape}, dtype={image.dtype}"
            )
            self._run_inference(image, msg.header)
        except Exception as e:
            self.get_logger().error(f"Exception in _image_callback: {type(e).__name__}: {e}")
            self.get_logger().error(traceback.format_exc())

    def _prepare_image_tensor(self, image: numpy.ndarray) -> torch.Tensor:
        image_chw = numpy.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image_chw).to(
            device=self.device,
            dtype=self.model_dtype,
            non_blocking=True,
        )
        image_tensor = image_tensor / 255.0
        return image_tensor.contiguous()

    def _normalize_masks(self, masks: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
        h, w = image_shape

        if masks is None:
            return torch.zeros((0, h, w), dtype=torch.bool, device=self.device)

        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks, device=self.device)

        masks = masks.to(self.device)

        if masks.numel() == 0:
            return torch.zeros((0, h, w), dtype=torch.bool, device=self.device)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0)

        if masks.ndim != 3:
            raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")

        return masks > 0.5

    def _run_inference(self, image: numpy.ndarray, header: std_msgs.msg.Header) -> None:
        self.inference_count += 1
        inference_id = self.inference_count
        total_start = time.time()

        self.get_logger().info(f"[Inference {inference_id}] Starting inference")

        try:
            t0 = time.time()
            image_tensor = self._prepare_image_tensor(image)
            self.get_logger().info(
                f"[Inference {inference_id}] Image tensor ready in {time.time() - t0:.3f}s, "
                f"shape={tuple(image_tensor.shape)}, device={image_tensor.device}, dtype={image_tensor.dtype}"
            )

            t1 = time.time()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast("cuda", enabled=False):
                        inference_state = self.processor.set_image(image_tensor)
                else:
                    inference_state = self.processor.set_image(image_tensor)
            self.get_logger().info(
                f"[Inference {inference_id}] set_image() done in {time.time() - t1:.3f}s"
            )

            t2 = time.time()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast("cuda", enabled=False):
                        output = self.processor.set_text_prompt(self.text_prompt, inference_state)
                else:
                    output = self.processor.set_text_prompt(self.text_prompt, inference_state)
            self.get_logger().info(
                f"[Inference {inference_id}] set_text_prompt() done in {time.time() - t2:.3f}s"
            )

            masks = self._normalize_masks(output["masks"], image.shape[:2])
            boxes = torch.as_tensor(output["boxes"], device=self.device, dtype=torch.float32)
            scores = torch.as_tensor(output["scores"], device=self.device, dtype=torch.float32)

            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            if scores.numel() == 0:
                scores = scores.reshape(0)

            self.get_logger().info(
                f"[Inference {inference_id}] Output: "
                f"masks_shape={tuple(masks.shape)}, "
                f"boxes_shape={tuple(boxes.shape)}, "
                f"scores_shape={tuple(scores.shape)}"
            )

            with self.result_lock:
                self.result = Segmentation.Result(
                    header=header,
                    image=image.copy(),
                    masks=masks.detach().clone(),
                    boxes=boxes.detach().clone(),
                    scores=scores.detach().clone(),
                )

            if masks.shape[0] == 0:
                self.get_logger().warn(f"[Inference {inference_id}] No masks detected.")

            if masks.shape[0] > 0:
                mask_ids = (
                    masks.to(torch.uint16)
                    * torch.arange(
                        1,
                        masks.shape[0] + 1,
                        device=self.device,
                        dtype=torch.uint16,
                    ).view(-1, 1, 1)
                ).amax(dim=0)
            else:
                mask_ids = torch.zeros(image.shape[:2], dtype=torch.uint8, device=self.device)

            mask_image_np = mask_ids.clamp(0, 255).to(torch.uint8).cpu().numpy()
            mask_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(mask_image_np)
            mask_image_msg.header = header
            self.mask_image_pub.publish(mask_image_msg)
            self.get_logger().info(f"[Inference {inference_id}] Published mask_image/compressed")

            detections = vision_msgs.msg.Detection2DArray()
            detections.header = header

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.tolist()

                detection = vision_msgs.msg.Detection2D()
                detection.bbox.size_x = float(max(0.0, x2 - x1))
                detection.bbox.size_y = float(max(0.0, y2 - y1))
                detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                detection.bbox.center.position.y = float((y1 + y2) / 2.0)

                detection.results.append(
                    vision_msgs.msg.ObjectHypothesisWithPose(
                        hypothesis=vision_msgs.msg.ObjectHypothesis(
                            class_id=self.text_prompt,
                            score=float(score.item()),
                        ),
                        pose=geometry_msgs.msg.PoseWithCovariance(
                            pose=geometry_msgs.msg.Pose(
                                position=geometry_msgs.msg.Point(
                                    x=float((x1 + x2) / 2.0),
                                    y=float((y1 + y2) / 2.0),
                                    z=0.0,
                                )
                            )
                        ),
                    )
                )
                detections.detections.append(detection)

            self.detections_pub.publish(detections)
            self.get_logger().info(
                f"[Inference {inference_id}] Published detections, count={len(detections.detections)}"
            )

            self.get_logger().info(
                f"[Inference {inference_id}] Finished in {time.time() - total_start:.3f}s"
            )

        except Exception as e:
            self.get_logger().error(
                f"[Inference {inference_id}] Exception: {type(e).__name__}: {e}"
            )
            self.get_logger().error(traceback.format_exc())

    def _visualize_result(self) -> None:
        try:
            with self.result_lock:
                if self.result is None:
                    return
                result = Segmentation.Result(
                    header=self.result.header,
                    image=self.result.image.copy(),
                    masks=self.result.masks.detach().clone(),
                    boxes=self.result.boxes.detach().clone(),
                    scores=self.result.scores.detach().clone(),
                )

            if result.masks.numel() == 0 or result.masks.shape[0] == 0:
                result_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(result.image)
                result_image_msg.header = result.header
                self.result_image_pub.publish(result_image_msg)
                return

            colored_masks = (
                result.masks.to(torch.uint8).unsqueeze(1)
                * self.colors[
                    torch.arange(result.masks.shape[0], device=self.device) % self.colors.shape[0]
                ].view(-1, 3, 1, 1)
            ).amax(dim=0).to(torch.uint8)

            overlay = colored_masks.permute(1, 2, 0).cpu().numpy()

            result.image = cv2.addWeighted(
                result.image,
                1.0,
                overlay,
                0.5,
                0.0,
            )

            box_thickness = 3
            font_scale = 1.2
            text_color = (255, 255, 255)
            text_thickness = 2

            for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
                x1, y1, x2, y2 = box.to(torch.int32).tolist()
                box_color = tuple(int(v) for v in self.colors[i % self.colors.shape[0]].cpu().tolist())
                text = f"{self.text_prompt}:{score.item():.2f}" if self.text_prompt else f"{score.item():.2f}"

                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    thickness=text_thickness,
                )

                text_y1 = y1 - text_height - 8 if y1 - text_height - 8 > 0 else y1
                text_y2 = y1 if y1 - text_height - 8 > 0 else y1 + text_height + 8

                cv2.rectangle(
                    result.image,
                    (x1, y1),
                    (x2, y2),
                    color=box_color,
                    thickness=box_thickness,
                )
                cv2.rectangle(
                    result.image,
                    (x1, text_y1),
                    (x1 + text_width + 8, text_y2),
                    color=box_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    result.image,
                    text,
                    (x1 + 4, text_y2 - 4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=text_thickness,
                )

            result_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(result.image)
            result_image_msg.header = result.header
            self.result_image_pub.publish(result_image_msg)

        except Exception as e:
            self.get_logger().error(f"Exception in _visualize_result: {type(e).__name__}: {e}")
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    segmentation = Segmentation()
    rclpy.spin(segmentation)
    segmentation.destroy_node()
    rclpy.shutdown()