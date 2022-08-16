import numpy
import matplotlib.pyplot as plt
import os
import subprocess
import math

instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]

algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]

horizons = ["100", "400", "1600", "6400", "25600", "102400"]

file1 = open("outputData1.txt", "w") 

for i in range(len(instances)):
	plt.figure()
	for alg in algorithms:
		x = []
		y = []
		for hor in horizons:
			x.append(math.log(int(hor)))
			sum_reg = 0
			for seed in range(0, 50):
				cmd = ['python3', 'bandit.py', '--instance',  instances[i], '--algorithm', alg, '--scale',  '2',  '--threshold',  '0',  '--horizon', hor, '--randomSeed', str(seed), '--epsilon',  '0.02']
				ret = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
				data = ret.stdout
				l = data.rstrip().split(", ")
				file1.write(data)
				print(l)
				reg = float(l[7])
				sum_reg = sum_reg + reg
			y.append(sum_reg / 50)
		plt.plot(x, y, label = alg)
	plt.xlabel("Horizon(log scale)")
	plt.ylabel("Regret")
	plt.title(instances[i])
	plt.legend()
	plt.savefig("task1_" + str(i) + ".png", dpi=300, bbox_inches='tight')

file1.close()

