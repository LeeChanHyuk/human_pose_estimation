import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('save.avi', fourcc, 30.0, (640, 480))

while(True):
    ret, frame = cap.read()    # Read 결과와 frame
    start_time = time.time()
    if(ret) :
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()