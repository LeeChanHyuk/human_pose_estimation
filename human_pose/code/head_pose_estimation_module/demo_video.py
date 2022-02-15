#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import service
import cv2
import time


def main(color=(224, 255, 255)):
    fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                        conf_threshold=0.95)

    fa = service.DepthFacialLandmarks("weights/sparse_face.tflite")

    handler = getattr(service, 'pose')
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)
        print(boxes)

        # raw copy for reconstruction
        feed = frame.copy()
        start_time2 = time.time()

        for results in fa.get_landmarks(feed, boxes):
            pitch, yaw, roll = handler(frame, results, color)
        #print('pitch = ', pitch, 'yaw = ', yaw, 'roll = ',roll)
        # cv2.imwrite(f'draft/gif/trans/img{counter:0>4}.jpg', frame)
        print(boxes[0][0])
        cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 3)
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        print(1/(time.time() - start_time2))


if __name__ == "__main__":
    main()
