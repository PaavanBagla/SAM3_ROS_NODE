## Quick Test (Clone and Run)

The following steps reproduce a clean setup from scratch.

### 1. Clone the repository

```bash
git clone https://github.com/PaavanBagla/SAM3_ROS_NODE.git
```

### 2. Place it in a ROS2 workspace

```bash
mkdir -p test_sam3_ws/src
mv SAM3_ROS_NODE test_sam3_ws/src/
cd test_sam3_ws
```

### 3. Build the workspace

```bash
colcon build
```

### 4. Source the workspace and run the node

```bash
source install/setup.bash
ros2 run sam3_ros segmentation_node
```

If the setup is correct, the node should start and publish segmentation topics.

### 5. Verify topics

In another terminal run:

```bash
ros2 topic list
```

You should see topics similar to:

```bash
/segmentation/detections
/segmentation/mask_image/compressed
/segmentation/result_image/compressed
```

If these topics appear, the node is running correctly and the segmentation pipeline is active.

---

# sam3_ros

🤖 ROS 2 wrapper for [SAM 3](https://github.com/facebookresearch/sam3)

## 📋 Prerequisites

- Python 3.10+
- PyTorch 2.7+
- CUDA-compatible GPU (CUDA 12.6+)
- ROS 2 Humble+

## 🚀 Installation

### 1. Clone

```bash
cd ~/ros2_ws/src
git clone https://github.com/mitukou1109/sam3_ros.git
```

### 2. Install dependencies

```bash
cd ~/ros2_ws/src
rosdep install -iyr --from-paths .
```

### 3. Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

## 🔑 Authentication

> [!IMPORTANT]
> SAM 3 checkpoints require access approval:
>
> 1. Request access on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3)
> 2. Create a **Read** token in your [Hugging Face settings](https://huggingface.co/settings/tokens)
> 3. Login with your token:
>
> ```bash
> cd ~/ros2_ws/src/sam3_ros
> uv run hf auth login
> # Enter your token (input will not be visible): [paste your token]
> ```

## 💻 Usage

```bash
source ~/ros2_ws/install/local_setup.bash
ros2 run sam3_ros segmentation_node
```

## 📡 Topics

### Subscribed

| Topic                     | Type                              | Description                                                      |
| ------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| `/camera_fl/image`        | `sensor_msgs/msg/Image`           | Raw input image; Bayer input is debayered by `cv_bridge`         |
| `/image/compressed`       | `sensor_msgs/msg/CompressedImage` | Compressed input image (when `use_compressed_image` is `true`)   |
| `/lidar_2d_projection`    | `sensor_msgs/msg/PointCloud2`     | LiDAR points carrying `x,y,z,u,v` (`sam3_mask_transform` only)   |
| `/sam3_{left,right}_contour` | `sam2_msgs/msg/DetectedRoadArea` | Boundary pixels to lift into 3D (`sam3_mask_transform` only)     |

Prefer the raw camera topic over a debayered one: on this stack `image_proc` runs a
persistent backlog that puts `/camera_fl/image_color` ~0.5s behind its own capture
stamp, and that delay propagates into every contour.

### Published

| Topic                          | Type                               | Description                                     |
| ------------------------------ | ---------------------------------- | ----------------------------------------------- |
| `~/mask_image/compressed`      | `sensor_msgs/msg/CompressedImage`  | Road mask, **PNG** (lossless): pixel value = mask id, 0 = not road |
| `~/obstacle_mask/compressed`   | `sensor_msgs/msg/CompressedImage`  | Union of the `obstacle_prompt` masks, PNG, 255 = obstacle (not published while `obstacle_prompt` is empty) |
| `~/detections`                 | `vision_msgs/msg/Detection2DArray` | Bounding boxes of detected masks                |
| `~/result_image/compressed`    | `sensor_msgs/msg/CompressedImage`  | Image with colored masks for visualization (built only while subscribed) |
| `/sam3_{left,right}_contour`   | `sam2_msgs/msg/DetectedRoadArea`   | Road boundary pixels, in original image scale   |
| `/sam3_{left,right}_boundary`  | `sensor_msgs/msg/PointCloud2`      | Road boundary in 3D (`sam3_mask_transform`)     |

## ⚙️ Parameters

Parameters marked *runtime* are re-read every frame, so
`ros2 param set /segmentation <name> <value>` takes effect on the next frame. The others are read
once at start-up.

### `segmentation_node`

| Parameter                   | Type     | Description                                                                             | Default |
| --------------------------- | -------- | --------------------------------------------------------------------------------------- | ------- |
| `image_topic`               | `string` | Raw image topic to segment                                                              | `/camera_fl/image` |
| `compressed_image_topic`    | `string` | Compressed image topic, used when `use_compressed_image` is `true`                      | `image/compressed` |
| `use_compressed_image`      | `bool`   | Subscribe to `compressed_image_topic` instead of `image_topic`                          | `false` |
| `input_max_size`            | `int`    | Longest image side fed to the model; larger images are downscaled first                 | `1024` |
| `use_amp`                   | `bool`   | fp16 autocast on CUDA (the weights stay fp32)                                           | `true` |
| `text_prompt`               | `string` | *runtime.* Road prompt. Terms separated by `or`/`\|` run as independent queries whose masks are unioned; `and`/`&` intersects them. Commas are **not** separators — a comma list is encoded as one literal phrase. | `road or gravel road or paved road or asphalt road or tarmac road` |
| `confidence_threshold`      | `float`  | *runtime.* Minimum confidence for a mask to be kept                                     | `0.3`   |
| `obstacle_prompt`           | `string` | *runtime.* Obstacles, prompted on the same image encoding. Their mask joins road beyond a vehicle to the ego road and drops boundary pixels next to a vehicle. `""` disables it. Every term costs time; see [Speed and warm-up](#-speed-and-warm-up). | `car or truck or person` |
| `obstacle_dilate_px`        | `int`    | *runtime.* Dilation (px) of the obstacle mask before it is used                         | `6` |
| `boundary_mode`             | `string` | *runtime.* `rowwise`: the outermost pixels per row of the hole-filled road component the vehicle is on, without horizon rows or image-border pixels. `legacy`: the previous largest-contour split. | `rowwise` |
| `boundary_close_kernel_size`| `int`    | *runtime.* Diameter of the elliptical kernel that closes the road mask before contour extraction. Must exceed the width of any patch or seam that splits the road. | `15` |
| `boundary_downsample`       | `int`    | Row-wise extraction runs on the mask downsampled by this factor (70–90 → 21–24 ms under load); a boundary pixel stays within 2 px at 1024 width. `1` disables. | `2` |
| `batch_prompts`             | `bool`   | *runtime.* Run all OR terms of both prompts as ONE batched pass with cached text encodings. Same masks as the per-term path, about half the time. `false` restores the per-term path. | `true` |
| `cuda_graphs`               | `bool`   | *runtime.* Replay the batched pass as recorded CUDA graphs, ~45 ms per frame faster under load. **Needs a warm-up of 16 s to 3 min**; read [Speed and warm-up](#-speed-and-warm-up) first. Has no effect when `batch_prompts` is false or a prompt uses AND (the node warns). | `false` |
| `compile_cache_dir`         | `string` | Where `cuda_graphs` keeps compiled kernels across restarts (inductor and Triton caches). Must be on a mounted volume to survive a recreated container; the default is inside the Hugging Face mount. `""` leaves torch's default under `/tmp` (lost on restart). | `/root/.cache/huggingface/torchinductor_sam3` |
| `compile_threads`           | `int`    | Compiler worker threads for the `cuda_graphs` warm-up, capped so it does not take every core while the rest of the vehicle software runs | `4` |
| `prompt_stats_period`       | `int`    | *runtime.* Every N inferences, log per prompt term how often it returned a mask and its mean best score. `0` disables. | `50` |
| `profile_stages`            | `bool`   | *runtime.* Log per-stage timings every `prompt_stats_period` frames. Synchronises the GPU at each stage boundary, which itself costs a little. | `false` |
| `result_visualization_rate` | `float`  | Publish rate (Hz) for visualization results; `0` disables the timer                     | `10.0`  |
| `left_contour_topic`, `right_contour_topic` | `string` | Contour output topics                                                   | `/sam3_left_contour`, `/sam3_right_contour` |

### `sam3_mask_transform_node`

Each contour is fused against the buffered LiDAR projection captured nearest to the
contour's own header stamp, so camera and inference latency delay *when* a boundary
appears without displacing *where* its points land.

| Parameter                     | Type    | Description                                                                                          | Default |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `lift_mode`                   | `string`| *runtime.* `mask`: lift the WHOLE road mask (every ground return whose pixel is road) and take per-bin lateral extremes in the vehicle frame. An extreme counts only with non-road ground seen just beyond it, and extremes at the image border are dropped. `contour`: the previous contour-pixel matching. | `mask` |
| `mask_topic`, `obstacle_mask_topic` | `string` | Mask inputs for `lift_mode=mask`                                                               | `/segmentation/mask_image/compressed`, `/segmentation/obstacle_mask/compressed` |
| `camera_info_topic`           | `string`| Image size for the mask lifting                                                                      | `/camera_fl/camera_info` |
| `ego_yaw_deg`                 | `float` | Yaw of the vehicle axis in `lidar_tc` (the LiDAR is mounted ~5.35° off it)                           | `-5.35` |
| `lift_x_min`, `lift_x_max`, `lift_bin_len` | `float` | Vehicle-frame range and bin length (m) of the lifted boundary                           | `5.0`, `60.0`, `2.0` |
| `max_pairing_skew`            | `float` | Largest capture-time gap (s) allowed between a contour and the projection it is fused with           | `0.08`  |
| `projection_buffer_duration`  | `float` | How much projection history (s) to retain; must exceed camera-to-contour latency                     | `2.0`   |
| `wait_for_newer`              | `float` | How long (s) a contour may wait for a cloud captured at or after it; `0` restores backward-only matching | `0.06` |
| `boundary_timeout`            | `float` | Publish an empty boundary once a side has gone this long (s) without a refresh; `0` disables          | `2.0`   |
| `qos_depth`                   | `int`   | Subscription queue depth                                                                              | `20`    |

## ⏱ Speed and warm-up

Measured on the vehicle PC (RTX 5080, SAM3 on GPU 1) on Route2 frames, four prompt terms
(`paved road or gravel road` + `car or person`), with the rest of the perception stack replaying
the same drive:

| configuration | per frame | rate |
| --- | --- | --- |
| per-term prompts (`batch_prompts:=false`), before the post-processing fixes | ~566 ms | ~1.7 Hz |
| batched, before the post-processing fixes | ~270 ms | ~3.7 Hz |
| batched (default) | ~210 ms | ~4.5 Hz |
| batched + `cuda_graphs` | ~170 ms | ~5.6 Hz |
| batched + `cuda_graphs` + the CPU pinning overlay (`docker-compose.cpuset.yml`) | ~159 ms | ~6.0 Hz |

The image encoder (~80 ms) costs the same for any number of terms; every extra term adds
grounding work, so keep the prompts short.

### `cuda_graphs` needs a warm-up: read this before enabling it

With `cuda_graphs` on, the first **two** passes for a given **number of prompt terms** compile
and record instead of running. Measured:

| compile cache (`compile_cache_dir`) | idle machine | full stack running | full stack + CPU pinning |
| --- | --- | --- | --- |
| empty (first start ever, or the cache was lost) | ~70 s | ~166 s | not measured |
| already holds this term count | ~16 s | ~47 s | ~27 s |

What the node does about it:

- **Enabled at start-up** (`-p cuda_graphs:=true`, or `SAM3_CUDA_GRAPHS=true` in the environment
  of the compose stack): the warm-up runs on a blank image **before** the node processes any
  image. The log shows `cuda_graphs WARM-UP: ...` with the time to expect, then
  `cuda_graphs: ready for N prompt terms after X s`, then
  `Segmentation node ready. Waiting for images...`. **Do not drive until that has appeared**:
  until then SAM3 publishes nothing.
- **Enabled at runtime** (`ros2 param set /segmentation cuda_graphs true`), **or a prompt change
  that alters the number of terms** (for example `obstacle_prompt` from `car or person` to
  `car`): the next one or two LIVE frames pay the warm-up (37 s measured for a new term count
  on an idle machine, partly cached). SAM3 publishes nothing for that time, and
  `sam3_mask_transform` publishes empty boundaries once `boundary_timeout` has passed. The node
  logs a `cuda_graphs WARM-UP` warning when it starts and `cuda_graphs: ready` when it ends.
  **Do this only with the vehicle stopped**, or put the prompts on the command line instead.
  Changing the words of a prompt while keeping the number of terms needs no warm-up, and
  neither does going back to a term count that was already warmed up in this run.
- **Persistence:** compiled kernels are kept in `compile_cache_dir` (~200 MB). The compose file
  mounts `~/.cache/huggingface` from the host, so the cache survives restarts and recreated
  containers. The warm-up warning says whether the cache is empty. At the first compile the node
  also warns if the directory is not writable or (inside a container) not on a mounted volume,
  because then every start pays the cold warm-up. Each term count has its own cache entries;
  deleting the directory costs only one cold warm-up.
- **If compiling fails**, the node logs an error, sets `cuda_graphs` to `false` and carries on
  with the eager pass. It never stops publishing for that reason.
- **Limits:** torch recompiles a function at most 8 times (torch 2.10's `recompile_limit`) and
  runs it eagerly after that, so cycling through many term counts in one run ends up on the
  eager path again (still correct, just not faster). Each recorded term count also keeps its own
  GPU memory pool.
