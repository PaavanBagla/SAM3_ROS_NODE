import queue
import threading
import time
import traceback
import re

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
from sam3.model import box_ops
from sam3.model.data_misc import FindStage
import sensor_msgs.msg
import std_msgs.msg
import torch
import vision_msgs.msg
from sam2_msgs.msg import DetectedRoadArea

from sam3_ros.road_mask_boundary import ego_component, rowwise_boundary


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
        use_amp = (
            self.declare_parameter("use_amp", True)
            .get_parameter_value()
            .bool_value
        )
        self.amp_enabled = bool(use_amp and self.device.type == "cuda")
        # Keep model weights in fp32 for compatibility; autocast handles mixed precision.
        self.model_dtype = torch.float32

        use_compressed_image = (
            self.declare_parameter("use_compressed_image", False)
            .get_parameter_value()
            .bool_value
        )
        self.text_prompt = (
            self.declare_parameter(
                "text_prompt",
                "road or gravel road or paved road or asphalt road or tarmac road",
            )
            .get_parameter_value()
            .string_value
        )
        self.confidence_threshold = (
            self.declare_parameter("confidence_threshold", 0.3)
            .get_parameter_value()
            .double_value
        )
        result_visualization_rate = (
            self.declare_parameter("result_visualization_rate", 10.0)
            .get_parameter_value()
            .double_value
        )
        self.input_max_size = int(
            self.declare_parameter("input_max_size", 1024)
            .get_parameter_value()
            .integer_value
        )
        # A surface-colour change or a patch reaching the road edge splits the road mask
        # in two, and only the largest contour survives boundary extraction. Closing with
        # a kernel wider than the seam rejoins the halves first. Closing only bridges
        # neighbouring regions, so this cannot pull a distant false positive into the road.
        self.boundary_close_kernel_size = int(
            self.declare_parameter("boundary_close_kernel_size", 15)
            .get_parameter_value()
            .integer_value
        )
        self._boundary_close_kernel = self._build_close_kernel(self.boundary_close_kernel_size)
        # How the road mask becomes left/right contours. "rowwise": the outermost pixels of the
        # hole-filled road component the vehicle is on, without the horizon rows or image-border
        # pixels (road_mask_boundary.rowwise_boundary). "legacy": the largest external contour
        # split at each row band's mean x, which hands a car's outline to the boundary and drops
        # road beyond a car that cuts the mask in two. Settable at runtime.
        self.boundary_mode = str(self.declare_parameter("boundary_mode", "rowwise").value)
        # Obstacles prompted on the SAME image encoding: set_image is the expensive call, a text
        # prompt on its state is cheap. Their mask bridges road components (far road beyond a
        # vehicle stays connected) and drops boundary pixels next to a vehicle, where the road
        # ends at the vehicle rather than at its edge. Published on ~/obstacle_mask/compressed.
        # Empty string disables it. Settable at runtime.
        self.obstacle_prompt = str(
            self.declare_parameter("obstacle_prompt", "car or truck or person").value)
        self.obstacle_dilate_px = int(self.declare_parameter("obstacle_dilate_px", 6).value)
        # Log, every N inferences, how often each prompt term returns a mask and with what
        # score -- "No masks detected" says nothing about WHICH term failed. 0 disables.
        self.prompt_stats_period = int(self.declare_parameter("prompt_stats_period", 50).value)
        # Run every OR term of the road AND obstacle prompts as ONE batched grounding pass over the
        # shared image features, with the text encodings cached (they never change between frames).
        # Masks are identical to the per-term path (IoU 1.000 on Route2 frames); measured on GPU 1:
        # 177 -> 123 ms for four terms on an idle machine, 269 -> 143 ms with the full stack running,
        # because the per-term path is many small, launch-bound kernels. false restores per-term.
        self.batch_prompts = bool(self.declare_parameter("batch_prompts", True).value)
        # Per-stage timing of every inference, logged every prompt_stats_period frames. Off by
        # default: it synchronises the GPU at each stage boundary so GPU work is attributed to the
        # stage that queued it, which itself costs a little.
        self.profile_stages = bool(self.declare_parameter("profile_stages", False).value)
        # Row-wise boundary extraction runs on the mask downsampled by this factor. Hole filling
        # and labelling cost scales with pixel count: 70-90 ms per frame at 1024x766 under the
        # full stack. At 2 a boundary pixel is still within 2 px at 1024 width. 1 disables.
        self.boundary_downsample = max(1, int(self.declare_parameter("boundary_downsample", 2).value))
        self._stage_times: dict = {}
        self._stage_t = None
        self._text_cache: dict = {}
        self._prompt_stats: dict = {}
        self._frames_total = 0
        self._frames_with_road = 0
        # The debayered /camera_fl/image_color arrives ~0.5s behind its own capture
        # stamp because image_proc runs a permanent backlog, and every millisecond of
        # that lands in the contour-to-LiDAR skew downstream. The raw Bayer topic is
        # ~0.02s behind, a third of the bandwidth, and cv_bridge debayers it here for
        # a few milliseconds of CPU.
        image_topic = (
            self.declare_parameter("image_topic", "/camera_fl/image")
            .get_parameter_value()
            .string_value
        )
        compressed_image_topic = (
            self.declare_parameter("compressed_image_topic", "image/compressed")
            .get_parameter_value()
            .string_value
        )
        left_contour_topic = (
            self.declare_parameter("left_contour_topic", "/sam3_left_contour")
            .get_parameter_value()
            .string_value
        )
        right_contour_topic = (
            self.declare_parameter("right_contour_topic", "/sam3_right_contour")
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(f"Segmentation node starting on device: {self.device}")
        self.get_logger().info(
            f"use_compressed_image={use_compressed_image}, "
            f"text_prompt='{self.text_prompt}', "
            f"confidence_threshold={self.confidence_threshold}, "
            f"boundary_close_kernel_size={self.boundary_close_kernel_size}, "
            f"boundary_mode={self.boundary_mode}, obstacle_prompt='{self.obstacle_prompt}', "
            f"result_visualization_rate={result_visualization_rate}, "
            f"use_amp={self.amp_enabled}"
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
            confidence_threshold=self.confidence_threshold,
        )
        self.get_logger().info("Processor ready.")

        self.result_lock = threading.Lock()
        self.result: Segmentation.Result | None = None
        self.received_frame_count = 0
        self.inference_count = 0
        self._zero_stamp_warned = False

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
        self.obstacle_mask_pub = self.create_publisher(
            sensor_msgs.msg.CompressedImage, "~/obstacle_mask/compressed", 1
        )
        self.result_image_pub = self.create_publisher(sensor_msgs.msg.Image, "~/result_image", 1)
        self.detections_pub = self.create_publisher(
            vision_msgs.msg.Detection2DArray, "~/detections", 1
        )
        self.left_contour_pub = self.create_publisher(DetectedRoadArea, left_contour_topic, 1)
        self.right_contour_pub = self.create_publisher(DetectedRoadArea, right_contour_topic, 1)

        # Cached prompt parse: avoid re-parsing regex every frame
        self._cached_prompt_parse: dict = {}

        # Async inference: callbacks drop frames into a maxsize=1 queue; worker thread
        # pops and runs inference so the ROS executor is never blocked. Created before
        # the subscriptions so no callback can ever reach a queue that does not exist.
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self.superseded_frame_count = 0

        if use_compressed_image:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                compressed_image_topic,
                self._image_compressed_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribed to topic: {compressed_image_topic}")
        else:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                image_topic,
                self._image_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribed to topic: {image_topic}")

        if result_visualization_rate > 0.0:
            self.result_image_compressed_pub = self.create_publisher(
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

        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="sam3_infer"
        )
        self._infer_thread.start()

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

    def _enqueue_latest(self, image, header: std_msgs.msg.Header) -> None:
        """Queue a frame for inference, discarding any frame it supersedes.

        Dropping the *incoming* frame instead would leave inference to start on the
        oldest frame of the busy window — up to one inference period stale before the
        model even sees it. Single-producer, so the drain-then-put is not racy.
        """
        while True:
            try:
                self._infer_queue.put_nowait((image, header))
                return
            except queue.Full:
                try:
                    self._infer_queue.get_nowait()
                    self.superseded_frame_count += 1
                except queue.Empty:
                    pass

    # The callbacks only queue the MESSAGE; the worker converts the one frame it runs on.
    # Converting every frame here (10 Hz, 9.5 MB each) cost GIL time the inference thread needs to
    # launch kernels, for frames of which ~80% were then superseded unused.
    def _image_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage) -> None:
        self.received_frame_count += 1
        self._enqueue_latest(msg, msg.header)

    def _image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        self.received_frame_count += 1
        self._enqueue_latest(msg, msg.header)

    def _decode(self, msg) -> numpy.ndarray:
        if isinstance(msg, sensor_msgs.msg.CompressedImage):
            image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            # desired_encoding="bgr8" makes cv_bridge debayer bayer_rggb8 input,
            # matching what image_proc would have produced.
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return self._normalize_input_image(image)

    def _inference_worker(self) -> None:
        while True:
            item = self._infer_queue.get()
            if item is None:
                break
            msg, header = item
            self._stage_t = time.perf_counter() if self.profile_stages else None
            try:
                image = self._decode(msg)
                self._mark("decode")
            except Exception as e:
                self.get_logger().error(f"Image decode failed: {type(e).__name__}: {e}")
                continue
            self._run_inference(image, header)

    def _sanitize_header(self, header: std_msgs.msg.Header) -> std_msgs.msg.Header:
        sanitized = std_msgs.msg.Header()
        sanitized.frame_id = header.frame_id

        if header.stamp.sec == 0 and header.stamp.nanosec == 0:
            sanitized.stamp = self.get_clock().now().to_msg()
            if not self._zero_stamp_warned:
                self.get_logger().warn(
                    "Received image header with zero timestamp; using current ROS time for outputs."
                )
                self._zero_stamp_warned = True
        else:
            sanitized.stamp = header.stamp

        return sanitized

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

    def _empty_outputs(
        self, image_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = image_shape
        masks = torch.zeros((0, h, w), dtype=torch.bool, device=self.device)
        boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
        scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
        return masks, boxes, scores

    def _run_prompt_query(
        self,
        prompt: str,
        inference_state,
        image_shape: tuple[int, int],
        inference_id: int,
        label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_start = time.time()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", enabled=self.amp_enabled, dtype=torch.float16):
                    output = self.processor.set_text_prompt(prompt, inference_state)
            else:
                output = self.processor.set_text_prompt(prompt, inference_state)

        masks = self._normalize_masks(output["masks"], image_shape)
        boxes = torch.as_tensor(output["boxes"], device=self.device, dtype=torch.float32)
        scores = torch.as_tensor(output["scores"], device=self.device, dtype=torch.float32)

        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        if scores.numel() == 0:
            scores = scores.reshape(0)

        st = self._prompt_stats.setdefault(prompt, [0, 0, 0.0, 0.0])
        st[0] += 1
        st[1] += int(masks.shape[0] > 0)
        st[2] += float(scores.max().item()) if scores.numel() > 0 else 0.0
        st[3] += time.time() - t_start

        suffix = f" [{label}]" if label else ""
        self.get_logger().debug(
            f"[Inference {inference_id}] set_text_prompt('{prompt}'){suffix} done in {time.time() - t_start:.3f}s"
        )

        return masks, boxes, scores

    @staticmethod
    def _or_terms(expression: str):
        """The OR terms of a prompt expression, or None when it is an AND/mixed expression."""
        lower = expression.strip().lower()
        if re.search(r"\s(?:and|&)\s", lower):
            return None
        return [t.strip() for t in re.split(r"\s*(?:\bor\b|\|)\s*", expression.strip()) if t.strip()]

    def _run_terms_batched(self, terms, inference_state, image_shape, inference_id):
        """All ``terms`` as one batched grounding pass. Returns per-term (masks, boxes, scores)."""
        key = tuple(terms)
        k = len(terms)
        t_start = time.time()
        with torch.inference_mode(), torch.autocast(
                "cuda", enabled=self.amp_enabled and self.device.type == "cuda", dtype=torch.float16):
            if key not in self._text_cache:
                if len(self._text_cache) > 8:
                    self._text_cache.clear()
                self._text_cache[key] = self.model.backbone.forward_text(list(terms), device=self.device)
            inference_state["backbone_out"].update(self._text_cache[key])
            find = FindStage(img_ids=torch.zeros(k, device=self.device, dtype=torch.long),
                             text_ids=torch.arange(k, device=self.device, dtype=torch.long),
                             input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                             input_points=None, input_points_mask=None)
            out = self.model.forward_grounding(
                backbone_out=inference_state["backbone_out"], find_input=find,
                geometric_prompt=self.model._get_dummy_prompt(num_prompts=k), find_target=None)
            probs = (out["pred_logits"].sigmoid()
                     * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            h, w = inference_state["original_height"], inference_state["original_width"]
            scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
            results = []
            for i, term in enumerate(terms):
                keep = probs[i] > self.confidence_threshold
                masks = out["pred_masks"][i][keep]
                if masks.shape[0]:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(), (h, w), mode="bilinear",
                        align_corners=False).sigmoid() > 0.5
                masks = self._normalize_masks(masks, image_shape)
                boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][i][keep]).float() * scale
                scores = probs[i][keep].float()
                st = self._prompt_stats.setdefault(term, [0, 0, 0.0, 0.0])
                st[0] += 1
                st[1] += int(masks.shape[0] > 0)
                st[2] += float(scores.max().item()) if scores.numel() else 0.0
                results.append((masks, boxes.reshape(-1, 4), scores))
        dt = time.time() - t_start
        for term in terms:
            self._prompt_stats[term][3] += dt / k
        self.get_logger().debug(
            f"[Inference {inference_id}] batched {k} terms {list(terms)} done in {dt:.3f}s")
        return results

    def _concat(self, parts, image_shape):
        parts = [p for p in parts if p[0].shape[0] > 0]
        if not parts:
            return self._empty_outputs(image_shape)
        return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                torch.cat([p[2] for p in parts]))

    def _run_prompt_expression(
        self,
        inference_state,
        image_shape: tuple[int, int],
        inference_id: int,
        expression: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expression = (self.text_prompt if expression is None else expression).strip()
        lower = expression.lower()

        # Cache parse result so regex runs only when prompt changes
        if expression not in self._cached_prompt_parse:
            has_or = bool(re.search(r"\s(?:or|\|)\s", lower))
            has_and = bool(re.search(r"\s(?:and|&)\s", lower))
            self._cached_prompt_parse[expression] = (has_or, has_and)
        has_or, has_and = self._cached_prompt_parse[expression]

        if has_or and has_and:
            self.get_logger().warn(
                f"[Inference {inference_id}] Mixed AND/OR expressions are not supported yet; using literal prompt."
            )
            return self._run_prompt_query(expression, inference_state, image_shape, inference_id)

        if has_or:
            terms = [t.strip() for t in re.split(r"\s*(?:\bor\b|\|)\s*", expression) if t.strip()]
            if not terms:
                return self._empty_outputs(image_shape)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_scores: list[torch.Tensor] = []

            for term in terms:
                masks, boxes, scores = self._run_prompt_query(
                    term, inference_state, image_shape, inference_id, label="OR"
                )
                if masks.shape[0] > 0:
                    all_masks.append(masks)
                if boxes.shape[0] > 0:
                    all_boxes.append(boxes)
                if scores.shape[0] > 0:
                    all_scores.append(scores)

            if not all_masks:
                return self._empty_outputs(image_shape)

            return (
                torch.cat(all_masks, dim=0),
                torch.cat(all_boxes, dim=0),
                torch.cat(all_scores, dim=0),
            )

        if has_and:
            terms = [t.strip() for t in re.split(r"\s*(?:\band\b|&)\s*", expression) if t.strip()]
            if not terms:
                return self._empty_outputs(image_shape)

            term_unions: list[torch.Tensor] = []
            term_best_scores: list[float] = []

            for term in terms:
                masks, _, scores = self._run_prompt_query(
                    term, inference_state, image_shape, inference_id, label="AND"
                )
                if masks.shape[0] == 0:
                    return self._empty_outputs(image_shape)
                term_unions.append(masks.any(dim=0))
                term_best_scores.append(float(scores.max().item()) if scores.numel() > 0 else 0.0)

            combined_mask = term_unions[0]
            for term_mask in term_unions[1:]:
                combined_mask = torch.logical_and(combined_mask, term_mask)

            if not torch.any(combined_mask):
                return self._empty_outputs(image_shape)

            ys, xs = torch.where(combined_mask)
            x1 = float(xs.min().item())
            y1 = float(ys.min().item())
            x2 = float(xs.max().item() + 1)
            y2 = float(ys.max().item() + 1)

            masks = combined_mask.unsqueeze(0)
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)
            score = min(term_best_scores) if term_best_scores else 0.0
            scores = torch.tensor([score], dtype=torch.float32, device=self.device)
            return masks, boxes, scores

        return self._run_prompt_query(expression, inference_state, image_shape, inference_id)

    @staticmethod
    def _build_close_kernel(size: int) -> numpy.ndarray:
        # Elliptical rather than rectangular so bridging a seam does not square off the
        # road outline the boundary classifier then splits into left and right.
        size = max(1, int(size))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _refresh_tunable_parameters(self, inference_id: int) -> None:
        """Adopt live `ros2 param set` changes for the knobs worth sweeping on a bag."""
        current_prompt = self.get_parameter("text_prompt").get_parameter_value().string_value
        if current_prompt != self.text_prompt:
            self.text_prompt = current_prompt
            self._cached_prompt_parse.clear()
            self.get_logger().info(
                f"[Inference {inference_id}] text_prompt updated to '{self.text_prompt}'"
            )

        current_threshold = (
            self.get_parameter("confidence_threshold").get_parameter_value().double_value
        )
        if current_threshold != self.confidence_threshold:
            self.confidence_threshold = current_threshold
            # Called without a state, so it only re-arms the threshold for later queries.
            # This frame has not run set_text_prompt yet, so the new value takes effect now.
            self.processor.set_confidence_threshold(current_threshold)
            self.get_logger().info(
                f"[Inference {inference_id}] confidence_threshold updated to "
                f"{current_threshold:.3f}"
            )

        current_kernel_size = (
            self.get_parameter("boundary_close_kernel_size").get_parameter_value().integer_value
        )
        if current_kernel_size != self.boundary_close_kernel_size:
            self.boundary_close_kernel_size = current_kernel_size
            self._boundary_close_kernel = self._build_close_kernel(current_kernel_size)
            self.get_logger().info(
                f"[Inference {inference_id}] boundary_close_kernel_size updated to "
                f"{self.boundary_close_kernel_size}"
            )

    def _refresh_boundary_parameters(self, inference_id: int) -> None:
        for name, attr, cast in (("boundary_mode", "boundary_mode", str),
                                 ("obstacle_prompt", "obstacle_prompt", str),
                                 ("obstacle_dilate_px", "obstacle_dilate_px", int),
                                 ("prompt_stats_period", "prompt_stats_period", int),
                                 ("batch_prompts", "batch_prompts", bool),
                                 ("profile_stages", "profile_stages", bool)):
            value = cast(self.get_parameter(name).value)
            if value != getattr(self, attr):
                setattr(self, attr, value)
                self.get_logger().info(f"[Inference {inference_id}] {name} updated to '{value}'")

    def _log_prompt_stats(self) -> None:
        if self.prompt_stats_period <= 0 or self._frames_total < self.prompt_stats_period:
            return
        parts = [f"frames with a road mask {self._frames_with_road}/{self._frames_total} "
                 f"({100.0 * self._frames_with_road / self._frames_total:.0f}%)"]
        for term, (n, hits, score_sum, secs) in self._prompt_stats.items():
            parts.append(f"'{term}': masks in {100.0 * hits / max(n, 1):.0f}% of {n}, "
                         f"mean max score {score_sum / max(n, 1):.2f}, {1e3 * secs / max(n, 1):.0f} ms")
        self.get_logger().info("Prompt stats: " + " | ".join(parts))
        self._prompt_stats = {}
        self._frames_total = self._frames_with_road = 0

    def _mark(self, stage: str) -> None:
        """Close the current timing stage (profile_stages only)."""
        if not self.profile_stages:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._stage_t is not None and stage:
            self._stage_times.setdefault(stage, []).append(1e3 * (now - self._stage_t))
        self._stage_t = now

    def _log_stage_times(self) -> None:
        if not self.profile_stages or not self._stage_times:
            return
        n = len(next(iter(self._stage_times.values())))
        if n < max(1, self.prompt_stats_period):
            return
        parts = [f"{k} {numpy.median(v):.1f}" for k, v in self._stage_times.items()]
        total = sum(numpy.median(v) for v in self._stage_times.values())
        self.get_logger().info(f"Stage times (median ms over {n}): " + " | ".join(parts)
                               + f" | sum {total:.0f}")
        self._stage_times = {}

    def _run_inference(self, image: numpy.ndarray, header: std_msgs.msg.Header) -> None:
        self.inference_count += 1
        inference_id = self.inference_count
        total_start = time.time()
        header = self._sanitize_header(header)

        self._refresh_tunable_parameters(inference_id)
        self._refresh_boundary_parameters(inference_id)

        self.get_logger().debug(f"[Inference {inference_id}] Starting inference")

        # Optionally downscale large images before inference to improve speed.
        orig_h, orig_w = image.shape[:2]
        if self.input_max_size > 0 and max(orig_h, orig_w) > self.input_max_size:
            scale = self.input_max_size / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = cv2.resize(image, (new_w, new_h))
        infer_h, infer_w = image.shape[:2]
        self._mark("resize")

        try:
            t0 = time.time()
            image_tensor = self._prepare_image_tensor(image)
            self.get_logger().debug(
                f"[Inference {inference_id}] Image tensor ready in {time.time() - t0:.3f}s, "
                f"shape={tuple(image_tensor.shape)}, device={image_tensor.device}, dtype={image_tensor.dtype}"
            )

            t1 = time.time()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast("cuda", enabled=self.amp_enabled, dtype=torch.float16):
                        inference_state = self.processor.set_image(image_tensor)
                else:
                    inference_state = self.processor.set_image(image_tensor)
            self.get_logger().debug(
                f"[Inference {inference_id}] set_image() done in {time.time() - t1:.3f}s"
            )

            self._mark("set_image")
            road_terms = self._or_terms(self.text_prompt) if self.batch_prompts else None
            obs_terms = (self._or_terms(self.obstacle_prompt)
                         if self.obstacle_prompt.strip() else [])
            if road_terms and obs_terms is not None:
                parts = self._run_terms_batched(road_terms + obs_terms, inference_state,
                                                image.shape[:2], inference_id)
                masks, boxes, scores = self._concat(parts[:len(road_terms)], image.shape[:2])
                obs_masks = self._concat(parts[len(road_terms):], image.shape[:2])[0]
            else:
                masks, boxes, scores = self._run_prompt_expression(
                    inference_state=inference_state,
                    image_shape=image.shape[:2],
                    inference_id=inference_id,
                )
                obs_masks = None

            self._mark("prompts")
            self._frames_total += 1
            self._frames_with_road += int(masks.shape[0] > 0)
            obstacle_mask = None
            if self.obstacle_prompt.strip():
                if obs_masks is None:
                    obs_masks, _, _ = self._run_prompt_expression(
                        inference_state=inference_state,
                        image_shape=image.shape[:2],
                        inference_id=inference_id,
                        expression=self.obstacle_prompt,
                    )
                obstacle_mask = (obs_masks.any(dim=0).cpu().numpy() if obs_masks.shape[0] > 0
                                 else numpy.zeros(image.shape[:2], dtype=bool))
                obstacle_msg = self.cv_bridge.cv2_to_compressed_imgmsg(
                    obstacle_mask.astype(numpy.uint8) * 255, dst_format="png")
                obstacle_msg.header = header
                self.obstacle_mask_pub.publish(obstacle_msg)
            self._mark("obstacle_mask_pub")

            self.get_logger().debug(
                f"[Inference {inference_id}] Output: "
                f"masks_shape={tuple(masks.shape)}, "
                f"boxes_shape={tuple(boxes.shape)}, "
                f"scores_shape={tuple(scores.shape)}"
            )

            result_for_publish = Segmentation.Result(
                header=header,
                image=image.copy(),
                masks=masks.detach().clone(),
                boxes=boxes.detach().clone(),
                scores=scores.detach().clone(),
            )

            with self.result_lock:
                self.result = result_for_publish

            if masks.shape[0] == 0:
                self.get_logger().warn(f"[Inference {inference_id}] No masks detected.")

            if masks.shape[0] > 0:
                # Use int32 on CUDA to avoid unsupported uint16 arange kernels.
                mask_ids = (
                    masks.to(torch.int32)
                    * torch.arange(
                        1,
                        masks.shape[0] + 1,
                        device=self.device,
                        dtype=torch.int32,
                    ).view(-1, 1, 1)
                ).amax(dim=0)
            else:
                mask_ids = torch.zeros(image.shape[:2], dtype=torch.uint8, device=self.device)

            mask_image_np = mask_ids.clamp(0, 255).to(torch.uint8).cpu().numpy()
            # PNG, not cv_bridge's default JPEG: the values are mask ids, and JPEG rings them
            # around the outline every consumer then has to clean up.
            mask_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(mask_image_np, dst_format="png")
            mask_image_msg.header = header
            self.mask_image_pub.publish(mask_image_msg)
            self.get_logger().debug(f"[Inference {inference_id}] Published mask_image/compressed")
            self._mark("mask_image_pub")

            # The raw full-size result image is built and published only when someone listens;
            # the compressed visualisation timer below still runs at its own rate.
            publish_result = self.result_image_pub.get_subscription_count() > 0
            annotated = image.copy()
            if publish_result and masks.shape[0] > 0:
                colored_masks = (
                    masks.to(torch.uint8).unsqueeze(1)
                    * self.colors[
                        torch.arange(masks.shape[0], device=self.device) % self.colors.shape[0]
                    ].view(-1, 3, 1, 1)
                ).amax(dim=0).to(torch.uint8)
                overlay = colored_masks.permute(1, 2, 0).cpu().numpy()
                annotated = cv2.addWeighted(image, 1.0, overlay, 0.5, 0.0)
            if publish_result:
                result_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                result_image_msg.header = header
                self.result_image_pub.publish(result_image_msg)
            self.get_logger().debug(f"[Inference {inference_id}] Published result_image")
            self._mark("result_image_pub")

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
            self._mark("detections_pub")
            self.get_logger().debug(
                f"[Inference {inference_id}] Published detections, count={len(detections.detections)}"
            )

            self._extract_and_publish_boundaries(
                masks, header, orig_h, orig_w, infer_h, infer_w, obstacle_mask=obstacle_mask
            )
            self._mark("boundaries")
            self._log_prompt_stats()
            self._log_stage_times()

            self.get_logger().info(
                f"[Inference {inference_id}] Finished in {time.time() - total_start:.3f}s, "
                f"detections={len(detections.detections)}, "
                f"superseded={self.superseded_frame_count}"
            )

        except Exception as e:
            self.get_logger().error(
                f"[Inference {inference_id}] Exception: {type(e).__name__}: {e}"
            )
            self.get_logger().error(traceback.format_exc())

    @staticmethod
    def _classify_boundaries_using_horizontal_bins(
        contour_points: numpy.ndarray, frame_height: int, num_bins: int = 50
    ) -> tuple[list, list]:
        if len(contour_points) == 0:
            return [], []
        y_min = int(contour_points[:, 1].min())
        bin_height = max(1, (frame_height - y_min) // num_bins)
        left_pts: list = []
        right_pts: list = []
        for i in range(num_bins):
            y_lo = y_min + i * bin_height
            y_hi = y_min + (i + 1) * bin_height
            bin_pts = contour_points[
                (contour_points[:, 1] > y_lo) & (contour_points[:, 1] <= y_hi)
            ]
            if len(bin_pts) > 0:
                mean_x = bin_pts[:, 0].mean()
                for pt in bin_pts:
                    (left_pts if pt[0] < mean_x else right_pts).append(pt)
        return left_pts, right_pts

    def _extract_and_publish_boundaries(
        self,
        masks: torch.Tensor,
        header: std_msgs.msg.Header,
        orig_h: int,
        orig_w: int,
        infer_h: int,
        infer_w: int,
        obstacle_mask: numpy.ndarray | None = None,
    ) -> None:
        MIN_CONTOUR_AREA = 1000.0
        try:
            if masks.shape[0] == 0:
                return
            if self.boundary_mode != "legacy":
                self._publish_rowwise_boundaries(
                    masks, header, orig_h, orig_w, infer_h, infer_w, obstacle_mask,
                    MIN_CONTOUR_AREA,
                )
                return

            # Use the union of all masks as the road mask
            road_mask_bool = masks.any(dim=0)
            road_mask = road_mask_bool.cpu().numpy().astype(numpy.uint8)

            road_mask = cv2.morphologyEx(
                road_mask, cv2.MORPH_CLOSE, self._boundary_close_kernel
            )

            contours, _ = cv2.findContours(
                road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                self.get_logger().warn(
                    f"Boundary drop: {masks.shape[0]} mask(s) produced no contour "
                    f"({int(road_mask.sum())} px after close)."
                )
                return

            road_contour = max(contours, key=cv2.contourArea)
            kept_area = cv2.contourArea(road_contour)

            # A surface-colour change or a patch reaching the road edge splits the mask
            # into several external contours; only the largest survives below. Logging
            # the discarded share tells fragmentation apart from a weak detection.
            if len(contours) > 1:
                total_area = sum(cv2.contourArea(c) for c in contours)
                self.get_logger().debug(
                    f"Boundary fragmentation: {len(contours)} contours, keeping "
                    f"{kept_area:.0f}px of {total_area:.0f}px "
                    f"({100.0 * kept_area / max(total_area, 1.0):.0f}%)."
                )

            if kept_area < MIN_CONTOUR_AREA:
                self.get_logger().warn(
                    f"Boundary drop: largest of {len(contours)} contour(s) is "
                    f"{kept_area:.0f}px, below MIN_CONTOUR_AREA={MIN_CONTOUR_AREA:.0f}."
                )
                return

            contour_points = numpy.array(road_contour).reshape(-1, 2).astype(numpy.float32)

            left_pts, right_pts = self._classify_boundaries_using_horizontal_bins(
                contour_points, infer_h
            )

            # Scale boundary points back to original image coordinates
            scale_x = orig_w / infer_w
            scale_y = orig_h / infer_h

            def _to_msg(pts: list) -> DetectedRoadArea:
                msg = DetectedRoadArea()
                msg.header = header
                arr = numpy.array(pts, dtype=numpy.float32)
                arr[:, 0] *= scale_x
                arr[:, 1] *= scale_y
                msg.road_area.data = arr.flatten().tolist()
                return msg

            if left_pts:
                self.left_contour_pub.publish(_to_msg(left_pts))
            if right_pts:
                self.right_contour_pub.publish(_to_msg(right_pts))

        except Exception as e:
            self.get_logger().error(
                f"Exception in _extract_and_publish_boundaries: {type(e).__name__}: {e}"
            )

    def _publish_rowwise_boundaries(self, masks, header, orig_h, orig_w, infer_h, infer_w,
                                    obstacle_mask, min_area) -> None:
        f = self.boundary_downsample
        # Downsample on the GPU by striding before the copy, so less crosses to the CPU too.
        road = masks.any(dim=0)[::f, ::f].cpu().numpy().astype(numpy.uint8)
        obstacle = obstacle_mask[::f, ::f] if obstacle_mask is not None else None
        # The seam closing still applies: a surface-colour change can split the road mask, and
        # the ego component would then keep only the near half.
        kernel = (self._boundary_close_kernel if f == 1
                  else self._build_close_kernel(max(1, self.boundary_close_kernel_size // f)))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel).astype(bool)
        comp = ego_component(road, bridge_mask=obstacle)
        if comp.sum() * f * f < min_area:
            self.get_logger().warn(
                f"Boundary drop: ego road component is {int(comp.sum()) * f * f}px, below "
                f"{min_area:.0f}."
            )
            return
        left, right = rowwise_boundary(comp, obstacle_mask=obstacle, row_step=max(1, 2 // f),
                                       obstacle_dilate_px=max(1, self.obstacle_dilate_px // f))
        # Back to inference pixels (centre of each downsampled cell), then to the original image.
        left = left * f + (f - 1) / 2.0
        right = right * f + (f - 1) / 2.0
        scale = numpy.array([orig_w / infer_w, orig_h / infer_h], dtype=numpy.float32)
        for pts, pub in ((left, self.left_contour_pub), (right, self.right_contour_pub)):
            if pts.shape[0] == 0:
                continue
            msg = DetectedRoadArea()
            msg.header = header
            msg.road_area.data = (pts * scale).flatten().tolist()
            pub.publish(msg)

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
                self.result_image_compressed_pub.publish(result_image_msg)
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
            self.result_image_compressed_pub.publish(result_image_msg)

        except Exception as e:
            self.get_logger().error(f"Exception in _visualize_result: {type(e).__name__}: {e}")
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    segmentation = Segmentation()
    rclpy.spin(segmentation)
    segmentation.destroy_node()
    rclpy.shutdown()
