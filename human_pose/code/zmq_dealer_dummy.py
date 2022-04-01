import random
import zmq
import time


def dummy_zmq_dealer():
    context = zmq.Context()
    worker = context.socket(zmq.DEALER)
    worker.setsockopt_string(zmq.IDENTITY, str(random.randint(0, 1000)))
    worker.connect("tcp://localhost:5557")
    start = True
    worker.send_string("Hello_from_dummy")
    count = 0
    while True:
        #count += 1
        #print(count)
        #time.sleep(0.01)
        if start:
            worker.send_string("recording data: %s" % random.randint(0,100))
        request = worker.recv()
        if request == "START":
            start = True
        if request == "STOP":
            start = False
        if request == "END":
            print("A is finishing")

        #print('fps is ' + str(1 / (time.time() - start_time)))

