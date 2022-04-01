import os
from multiprocessing import Process
from pose_estimation_with_zeromq import main_function
from zmq_dealer_dummy import dummy_zmq_dealer
from zmq_router import router_function
from time import sleep
if __name__ == "__main__":
	p1 = Process(target=router_function)
	p2 = Process(target=dummy_zmq_dealer)
	p3 = Process(target=main_function)
	p1.start()
	print('p1 start')
	# 여기에 sleep 넣어볼까?
	p2.start()
	print('p2 start')
	p3.start()
	print('p3 start')
	p1.join()
	print('p1 join')
	p2.join()
	print('p2 join')
	p3.join()
	print('p3 join')