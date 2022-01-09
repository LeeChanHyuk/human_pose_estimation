#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import head_pose_estimation_module.service as service
import cv2
import time
import os


def main(color=(224, 255, 255)):
    base_path = os.getcwd()
    fd = service.UltraLightFaceDetecion(os.path.join(base_path,"human_pose/head_pose_estimation_module/weights/RFB-320.tflite"),
                                        conf_threshold=0.95)

    fa = service.DepthFacialLandmarks(os.path.join(base_path, "human_pose/head_pose_estimation_module/weights/sparse_face.tflite"))
    face_detection = 
    handler = getattr(service, 'pose')
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)

        # raw copy for reconstruction
        feed = frame.copy()

        for results in fa.get_landmarks(feed, boxes):
            pitch, yaw, roll = handler(frame, results, color)
        #print('pitch = ', pitch, 'yaw = ', yaw, 'roll = ',roll)
        # cv2.imwrite(f'draft/gif/trans/img{counter:0>4}.jpg', frame)
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
