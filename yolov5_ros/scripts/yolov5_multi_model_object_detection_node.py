#! /usr/bin/env python

import rospy

from modules.yolov5_object_detector import YoloV5ObjectDetector


class YoloMultiModelObjectDetectionNode:

    def __init__(self):
        device = rospy.get_param("~device")

        self._object_detectors = []
        for i in range(1, 9999):
            config_path = rospy.get_param(f"~config_{i}", default=None)
            weights_path = rospy.get_param(f"~weights_path_{i}", default=None)

            score_threshold = rospy.get_param(f"~score_threshold_{i}", default=None)
            iou_threshold = rospy.get_param(f"~iou_threshold_{i}", default=None)
            frame_rate = rospy.get_param(f"~frame_rate_{i}", default=None)
            if None not in [config_path, weights_path, score_threshold, iou_threshold, frame_rate]:
                object_detector = YoloV5ObjectDetector(
                    config_path=config_path,
                    weights_path=weights_path,
                    device=device,
                    frame_rate=frame_rate,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    image_sub_topic=f"/input/image/compressed_{i}",
                    image_pub_topic=f"/output/image/compressed_{i}",
                    bbox_pub_topic=f"/output/bounding_boxes_{i}",
                    server_name=f"/check_for_objects_{i}"
                )
                self._object_detectors.append(object_detector)
            else:
                break

        self._n_models = len(self._object_detectors)

    # ==================================================================================================
    #
    #   Main
    #
    # ==================================================================================================
    def main(self):
        while not rospy.is_shutdown():
            for object_detector in self._object_detectors:
                object_detector.check_subscriber()

            rospy.sleep(0.5)


if __name__ == "__main__":
    rospy.init_node("yolov5_multi_model_object_detection_node")
    YoloMultiModelObjectDetectionNode().main()
