import os
from multiprocessing import Process
from pose_estimation_clean_version import main_function
from zmq_router import router_function
from time import sleep
if __name__ == "__main__":
	p1 = Process(target=router_function)
	p2 = Process(target=main_function)
	p1.start()
	print('p1 start')
	p2.start()
	print('p2 start')
	p1.join()
	print('p1 join')
	p2.join()
	print('p2 join')