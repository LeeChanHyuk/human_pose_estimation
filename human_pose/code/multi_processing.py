import os
from multiprocessing import Process
from pose_estimation_with_semi_final import main_function
from zmq_router import router_function
from time import sleep
if __name__ == "__main__":
	p1 = Process(target=router_function)
	p3 = Process(target=main_function)
	p1.start()
	print('p1 start')
	p3.start()
	print('p3 start')
	p1.join()
	print('p1 join')
	p3.join()
	print('p3 join')