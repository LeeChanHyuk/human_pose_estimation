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
        sockets = dict(poll.poll(2))

        if sockets:
            identity = tracker.recv() # id
            msg = tracker.recv()
            send_message = msg_classification(msg)
            counter[identity] += 1 # id에 메세지가 하나 더 왔다.
        # start recording
        #for identity in counter.keys():
            #print('router send1')
            tracker.send(identity, zmq.SNDMORE)
            #print('router send2')
            tracker.send_string("START")
        
            message = to_renderer.recv()
            splited_list = list(str(msg).split(' '))
            to_renderer.send_string(send_message)

def msg_classification(msg):
    msg = list(str(msg).split(' '))
    if len(msg) == 1 or len(msg) == 3:
        send_message = 'N'
    elif len(msg) == 4: # only detection mode
        send_message = 'D' + ' ' + msg[1] + ' ' + msg[2] + ' ' + msg[3]
    elif len(msg) == 7: # estimation mode
        send_message = 'E' + ' ' + msg[1] + ' ' + msg[2] + ' ' + msg[3] + ' ' + msg[4] + ' ' + msg[5] + ' ' + msg[6]
    elif len(msg) == 8: # recognition mode
        send_message = 'A' + ' ' + msg[1]+ ' ' + msg[2] + ' ' + msg[3] + ' ' + msg[4] + ' ' + msg[5] + ' ' + msg[6]
    print('router ' + send_message)
    return send_message
            

