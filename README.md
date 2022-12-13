# YOLO v5 ROS

<!--
# ==================================================================================================
#
#   Overview
#
# ==================================================================================================
--->

# Overview <a id="Overview"></a>

The `yolov5_ros` package provides real-time object detection.

This package contains submodule of https://github.com/ultralytics/yolov5.

**Content:**

* [Overview](#Overview)
* [Dependencies](#Dependencies)
* [Weights](#Weights)
* [Example](#Example)
* [Nodes](#Nodes)

<!--
# ==================================================================================================
#
#   Dependencies
#
# ==================================================================================================
--->

## Dependencies <a id="Dependencies"></a>

Code wrapped from [ultralytics/yolov5](https://github.com/ultralytics/yolov5) at:

* URL: `https://github.com/ultralytics/yolov5`
* Branch: `master`
* Commit: [`8d3c3ef45ce1d530aa3751f6187f18cfd9c40791`](https://github.com/ultralytics/yolov5/tree/8d3c3ef45ce1d530aa3751f6187f18cfd9c40791)

Original `README.md`: https://github.com/ultralytics/yolov5/blob/8d3c3ef45ce1d530aa3751f6187f18cfd9c40791/README.md

Original `LICENSE`: https://github.com/ultralytics/yolov5/blob/8d3c3ef45ce1d530aa3751f6187f18cfd9c40791/LICENSE

<!--
# ==================================================================================================
#
#   Weights
#
# ==================================================================================================
--->

## Weights

The weight from YOLO v5 can be found here

    roscd yolov5_ros/weights
    wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt

<!--
# ==================================================================================================
#
#   Example
#
# ==================================================================================================
--->

## Example <a id="Example"></a>

Start the server.

    roslaunch yolov5_ros yolov5_ros.launch

<!--
# ==================================================================================================
#
#   Nodes
#
# ==================================================================================================
--->

## Node <a id="Node"></a>

This is the main YOLO ROS: Real-Time Object Detection for ROS node. It uses the camera measurements to detect pre-learned objects in the frames.

<!--
# ==================================================================================================
#   Subscribed Topics
# ==================================================================================================
--->

### Subscribed Topics

* **`/input/activation`** ([std_msgs/Bool])  
  Enable or disable image processing in subscriber callback.


* **`/input/image/compressed`** ([sensor_msgs/CompressedImage])  
  Subscribe to an image for object detection.

<!--
# ==================================================================================================
#   Published Topics
# ==================================================================================================
--->

### Published Topics

* **`output/image/compressed`** ([sensor_msgs/CompressedImage])  
  Publish detection image including the bounding boxes.

* **`output/bounding_boxes`** ([yolo_ros_msgs/BoundingBoxes])  
  Publish an array of bounding boxes that contains the position, size, class name, class probability.

<!--
# ==================================================================================================
#   Actions
# ==================================================================================================
--->

### Actions

* **`check_for_objects`** ([yolo_ros_msgs/CheckForObjectsAction])  
  Sends an action with an image, the result is an array of bounding boxes.

<!--
# ==================================================================================================
#   Parameters
# ==================================================================================================
--->

### Parameters

* **`~is_autostart`** (bool, default: true)  
  Enable or disable automatic start of image processing in subscriber callback.


* **`~frame_rate`** (int, default: 60)
  The maximum FrameRate of subscriber callback.


* **`~is_publish_bounding_box_image`** (bool, default: true)  
  Enable or disable publishing detection image including the bounding boxes in subscriber callback.


* **`~is_publish_bounding_box`** (bool, default: true)  
  Enable or disable publishing an array of bounding boxes in subscriber callback.


* **`~score_threshold`** (float, default: 0.5)  
  The maximum threshold of class probability to publish.


* **`~iou_threshold`** (float, default: 0.45)  
  The maximum threshold of Intersection over Union (IoU) in Non-Maximum Suppression (NMS).


* **`~config`** (str, default: "$(find yolov5_ros)/yolov5/models/yolov5m.yaml")  
  The config file path of YOLO V5.


* **`~weights_path`** (str, default: "$(find yolov5_ros)/weights/yolov5m.pt")  
  The weights file path of YOLO V5.


* **`~device`** (str, default: "cuda")  
  The processing device on PyTorch ("cpu" or "cuda").
  