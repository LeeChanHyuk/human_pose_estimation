import os
import numpy as np
import time

f = open('test.txt', 'r')
start_time = time.time()
while True:
	line = f.readline()
	if not line: break
print('fps = '+ str(time.time() - start_time))