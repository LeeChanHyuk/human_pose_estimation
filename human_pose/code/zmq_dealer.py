import random
import zmq
import time

context = zmq.Context()
worker = context.socket(zmq.DEALER)
worker.setsockopt_string(zmq.IDENTITY, str(random.randint(0, 1000)))
worker.connect("tcp://localhost:5558")
start = True
worker.send_string("Hello")
while True:
    if start:
        worker.send_string("recording data: %s" % random.randint(0,100))
        print('send')
    request = worker.recv()
    print('recv')
    if request == "START":
        start = True
    if request == "STOP":
        start = False
    if request == "END":
        print("A is finishing")
        break