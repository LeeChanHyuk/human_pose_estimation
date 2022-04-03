import os
import numpy as np
import time

f = open('test.txt', 'r')
start_time = time.time()
while True:
	line = f.readline()
	if not line: break
for i in range(1, 10000):
	f.write(str(i)+'\n')
print('fps = '+ str(time.time() - start_time))