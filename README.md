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

| Topic               | Type                              | Description            |
| ------------------- | --------------------------------- | ---------------------- |
| `/image`            | `sensor_msgs/msg/Image`           | Raw input image        |
| `/image/compressed` | `sensor_msgs/msg/CompressedImage` | Compressed input image |

### Published

| Topic                       | Type                               | Description                                     |
| --------------------------- | ---------------------------------- | ----------------------------------------------- |
| `~/mask_image/compressed`   | `sensor_msgs/msg/CompressedImage`  | Mask image (pixel value corresponds to mask ID) |
| `~/detections`              | `vision_msgs/msg/Detection2DArray` | Bounding boxes of detected masks                |
| `~/result_image/compressed` | `sensor_msgs/msg/CompressedImage`  | Image with colored masks for visualization      |

## ⚙️ Parameters

| Parameter                   | Type     | Description                                                                             | Default |
| --------------------------- | -------- | --------------------------------------------------------------------------------------- | ------- |
| `use_compressed_image`      | `bool`   | Use compressed image input. If `true`, subscribes to `/image/compressed`, else `/image` | `true`  |
| `text_prompt`               | `string` | Text prompt for guided segmentation                                                     | `""`    |
| `confidence_threshold`      | `float`  | Minimum confidence threshold for mask filtering                                         | `0.5`   |
| `result_visualization_rate` | `float`  | Publish rate (Hz) for visualization results                                             | `10.0`  |
