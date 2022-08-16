import numpy
import matplotlib.pyplot as plt
import os
import subprocess

instances = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]

horizons = ["100", "400", "1600", "6400", "25600", "102400"]

file1 = open("outputData2.txt", "w") 

for i in range(len(instances)):
	x = []
	y = []
	c = 0.02
	while(c <= 0.3):
		x.append(c)
		sum_reg = 0
		for seed in range(0, 50):
			cmd = ['python3', 'bandit.py', '--instance',  instances[i], '--algorithm', 'ucb-t2', '--scale',  str(c),  '--threshold',  '0',  '--horizon', '10000', '--randomSeed', str(seed), '--epsilon',  '0.02']
			ret = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
			data = ret.stdout
			l = data.rstrip().split(", ")
			file1.write(data)
			print(data)
			reg = float(l[7])	
			sum_reg = sum_reg + reg
		y.append(sum_reg / 50)
		c = round(c + 0.02, 2)
	plt.plot(x, y, label = instances[i])
plt.xlabel("Scale")
plt.ylabel("Regret")
plt.title("Task-2")
plt.legend()
plt.savefig("task2.png", dpi=300, bbox_inches='tight')

file1.close()

