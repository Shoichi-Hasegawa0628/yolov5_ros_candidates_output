#! /usr/bin/env python

import sys
import time
from pathlib import Path

import actionlib
import cv2
import cv_bridge
import numpy as np
import rospkg
import rospy
import sensor_msgs.msg as sensor_msgs
import torch
import yolo_ros_msgs.msg as yolo_ros_msgs
from torch import nn

path = Path(rospkg.RosPack().get_path("yolov5_ros"))
sys.path.append(str(path))
from yolov5.models.yolo import Detect, Model
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import colors


class YoloV5ObjectDetector:
    _bridge = cv_bridge.CvBridge()

    def __init__(
            self,
            config_path, weights_path, device,
            frame_rate=60, score_threshold=0.5, iou_threshold=0.45,
            image_sub_topic="/input/image/compressed", image_pub_topic="/output/image/compressed",
            bbox_pub_topic="/output/bounding_boxes", server_name="/check_for_objects",
    ):

        self._frame_rate = frame_rate

        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold

        self._frame_time = 1.0 / self._frame_rate
        self._before_time = 0

        self._device = device

        self._model, self._classes = self._init_model(config_path, weights_path, self._device)

        self._subscriber_kwargs = dict(name=image_sub_topic, data_class=sensor_msgs.CompressedImage, callback=self._image_callback)
        self._subscriber = None

        # Publisher
        self._image_publisher = rospy.Publisher(image_pub_topic, sensor_msgs.CompressedImage, queue_size=1)
        self._bboxes_publisher = rospy.Publisher(bbox_pub_topic, yolo_ros_msgs.BoundingBoxes, queue_size=1)

        # ActionServer
        self._server = actionlib.SimpleActionServer(server_name, yolo_ros_msgs.CheckForObjectsAction, self.action_call_back, auto_start=False)
        self._server.start()
        rospy.loginfo("Ready...")

    @staticmethod
    def _init_model(config, weight_path, device):
        """
        Args:
            config:
            weight_path:
            device:

        Returns:

        """
        rospy.loginfo(f"Loading model: {Path(config).name}, {Path(weight_path).name}")

        model = Model(cfg=config)
        load_pt = torch.load(weight_path)

        model.load_state_dict(load_pt["model"].state_dict())
        model.names = load_pt["model"].names

        for module in model.modules():
            if type(module) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
                module.inplace = True

        model.eval()
        model.requires_grad_(False)
        model = model.to(device)
        classes = model.names

        rospy.loginfo("Complete Loading model ...")
        return model, classes

    # ==================================================================================================
    #
    #   Main
    #
    # ==================================================================================================
    def check_subscriber(self):
        connection_exists = (self._image_publisher.get_num_connections() > 0) or (self._bboxes_publisher.get_num_connections() > 0)

        if connection_exists and (self._subscriber is None):
            self._subscriber = rospy.Subscriber(**self._subscriber_kwargs)
            rospy.loginfo(f"Register subscriber: {self._subscriber.resolved_name}")

        elif (not connection_exists) and (self._subscriber is not None):
            rospy.loginfo(f"Unregister subscriber: {self._subscriber.resolved_name}")
            self._subscriber.unregister()
            self._subscriber = None

    # ==================================================================================================
    #
    #   ROS Callback
    #
    # ==================================================================================================
    def _image_callback(self, msg):
        """
        Args:
            msg (sensor_msgs.CompressedImage):
        """
        now_time = time.time()
        if (now_time - self._before_time) < self._frame_time:
            return

        self._before_time = now_time
        bgr8_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        rgb8_image = self._bgr8_to_rgb8(bgr8_image)
        bounding_boxes = self._predict_image(rgb8_image=rgb8_image)

        self._publish_detection_image(rgb8_image, bounding_boxes, self._classes)
        self._publish_bounding_boxes(bounding_boxes, self._classes)

        elapsed_time = (time.time() - now_time)
        rospy.loginfo("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    def action_call_back(self, goal):
        """
        Args:
            goal (yolo_ros_msgs.CheckForObjectsGoal):
        """
        bgr8_image = self._bridge.compressed_imgmsg_to_cv2(goal.image)
        rgb8_image = self._bgr8_to_rgb8(bgr8_image)
        bounding_boxes = self._predict_image(rgb8_image=rgb8_image)
        self._publish_detection_image(rgb8_image, bounding_boxes, self._classes)

        # if not self._server.is_preempt_requested():
        result = yolo_ros_msgs.CheckForObjectsResult()
        result.bounding_boxes = self._generate_bounding_box_msg(bounding_boxes, self._classes)
        result.id = goal.id
        self._server.set_succeeded(result)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def _predict_image(self, rgb8_image):
        """
        Args:
            rgb8_image (np.ndarray):
        """
        batch_image = torch.from_numpy(rgb8_image).permute(2, 0, 1).unsqueeze(0).type(torch.float32).to(self._device) / 255.0
        result = self._model(batch_image)[0]
        result = non_max_suppression(result, conf_thres=self._score_threshold, iou_thres=self._iou_threshold)
        bounding_boxes = []
        for detections in result:
            detections = detections.detach().cpu().numpy()
            for detection in reversed(detections):
                (x0, y0, x1, y1), confidence, class_id = detection[:4].astype(np.int32), detection[4], int(detection[5])
                bounding_boxes.append((x0, y0, x1, y1, confidence, class_id))

        return bounding_boxes

    def _publish_detection_image(self, rgb8_image, bounding_boxes, classes):
        """
        Args:
            rgb8_image:
            bounding_boxes:
            classes:
        """
        image_with_bbox = self._generate_result_images(rgb8_image, bounding_boxes, classes)
        image_with_bbox_bgr8 = image_with_bbox[..., ::-1]
        compressed_image = self._bridge.cv2_to_compressed_imgmsg(image_with_bbox_bgr8)
        compressed_image.header.stamp = rospy.Time.now()
        self._image_publisher.publish(compressed_image)

    def _publish_bounding_boxes(self, bounding_boxes, classes):
        """
        Args:
            bounding_boxes:
            classes:
        """
        bounding_boxes_msg = self._generate_bounding_box_msg(bounding_boxes, classes)
        self._bboxes_publisher.publish(bounding_boxes_msg)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    @staticmethod
    def _generate_result_images(image, bounding_boxes, classes):
        """
        Args:
            image:
            bounding_boxes:
            classes:

        Returns:

        """
        output_image = image.copy()
        text_config = {'fontFace': cv2.FONT_HERSHEY_DUPLEX, 'fontScale': 0.6, 'thickness': 1}
        for i, bounding_box in enumerate(bounding_boxes):
            x0, y0, x1, y1, confidence, class_id = bounding_box
            color = colors(class_id, bgr=True)
            cv2.rectangle(output_image, (x0, y0), (x1, y1), color, thickness=2)
            label_name = classes[class_id]
            label_str = f"{label_name} {confidence * 100:.1f}"
            size, baseline = cv2.getTextSize(text=label_str, **text_config)
            cv2.rectangle(output_image, (x0, y0), (x0 + size[0], y0 + size[1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(output_image, org=(x0, y0 + size[1]), color=(255, 0, 0), text=label_str, **text_config)

        return output_image

    @staticmethod
    def _generate_bounding_box_msg(bounding_boxes, classes):
        """

        Args:
            bounding_boxes:
            classes:

        Returns:

        """
        msg = yolo_ros_msgs.BoundingBoxes()
        msg.header.stamp = rospy.Time.now()
        for i, bounding_box in enumerate(bounding_boxes):
            x0, y0, x1, y1, confidence, class_id = bounding_box

            bounding_box_msg = yolo_ros_msgs.BoundingBox()
            bounding_box_msg.xmin = x0
            bounding_box_msg.ymin = y0
            bounding_box_msg.xmax = x1
            bounding_box_msg.ymax = y1
            bounding_box_msg.probability = confidence
            bounding_box_msg.Class = classes[class_id]
            bounding_box_msg.id = i
            msg.bounding_boxes.append(bounding_box_msg)

        return msg

    @staticmethod
    def _bgr8_to_rgb8(bgr8):
        return cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB)
