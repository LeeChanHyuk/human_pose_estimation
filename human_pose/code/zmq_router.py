from cv2 import split
import zmq
import time

from collections import defaultdict

def router_function():
    line = ['1', '0 0 0', '0 0 0', 'standard'] # mode / eye_position / head_rotation / human_action
    context2 = zmq.Context()
    to_renderer = context2.socket(zmq.REP)
    to_renderer.bind("tcp://*:5558")
    line = ['0', '0 0 0', '0 0 0', 'standard'] # tracking_mode / eye_position / head_rotation / human action

    while True:
        # message read from txt file to communicate with tracker
        start_time = time.time()
        communication_read = open('communication.txt', 'r')
        for i in range(4):
            line[i] = communication_read.readline()
        communication_read.close()
        message = to_renderer.recv()
        message = str(message)[2]

        # message write to txt file to communicate with tracker
        communication_write = open('communication.txt', 'r+')
        communication_write.write(message)
        communication_write.close()
        #while line[0] == '':
        #    communication_read = open('communication.txt', 'r')
        #    for i in range(4):
        #        line[i] = communication_read.readline()
        #    communication_read.close()
        if line[0].strip() == '0':
            send_message = 'N'
        elif line[0].strip() == '1':
            send_message = 'D' + ' ' + line[1].strip()
        elif line[0].strip() == '2':
            send_message = 'E' + ' ' + line[1].strip() + ' ' + line[2].strip()
        elif line[0].strip() == '3':
            send_message = 'A' + ' ' + line[1].strip() + ' ' + line[2].strip() + ' ' + line[3].strip()
        to_renderer.send_string(send_message)
        print('fps = ', str(1/(time.time() - start_time)))
