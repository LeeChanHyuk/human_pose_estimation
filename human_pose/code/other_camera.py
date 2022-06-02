import cv2

cap_0 = cv2.VideoCapture(1)
#cap_1 = cv2.VideoCapture(1)

while 1:
    ret, frame_0 = cap_0.read()
    #ret, frame_1 = cap_1.read()
    cv2.imshow("frame_0", frame_0)
    #cv2.imshow("frame_1", frame_1)
    cv2.waitKey(1)