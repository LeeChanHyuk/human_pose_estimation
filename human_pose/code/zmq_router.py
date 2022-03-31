from cv2 import split
import zmq
import time

from collections import defaultdict

def router_function():
    context = zmq.Context()
    context2 = zmq.Context()
    tracker = context.socket(zmq.ROUTER)
    tracker.bind("tcp://*:5557")
    to_renderer = context2.socket(zmq.REP)
    to_renderer.bind("tcp://*:5558")

    poll = zmq.Poller()
    poll.register(tracker, zmq.POLLIN)
    counter = defaultdict(int)
    count=0
    while True:
        # handle input
        sockets = dict(poll.poll(1000))

        if sockets:
            identity = tracker.recv()
            msg = tracker.recv()
            counter[identity] += 1
        # start recording
        for identity in counter.keys():
            tracker.send(identity, zmq.SNDMORE)
            tracker.send_string("START")
        
        message = to_renderer.recv()
        splited_list = list(str(msg).split(' '))
        if len(splited_list) > 5:
            to_renderer.send_string(str(msg))
        else:
            to_renderer.send_string('N')

