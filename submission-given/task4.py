import numpy
import matplotlib.pyplot as plt
import os
import subprocess
import math

instances = ["../instances/instances-task4/i-1.txt", "../instances/instances-task4/i-2.txt"]

algorithms = ["alg-t4"]

horizons = ["100", "400", "1600", "6400", "25600", "102400"]

threshold = ["0.2", "0.6"]

file1 = open("outputData4.txt", "w") 

for i in range(len(instances)):
	for alg in algorithms:
		for th in threshold:
			plt.figure()
			x = []
			y = []
			for hor in horizons:
				x.append(math.log(int(hor)))
				sum_reg = 0
				for seed in range(0, 50):
					cmd = ['python3', 'bandit.py', '--instance',  instances[i], '--algorithm', alg, '--scale',  '2',  '--threshold',  th,  '--horizon', hor, '--randomSeed', str(seed), '--epsilon',  '0.02']
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
			plt.ylabel("HIGHS-REGRET")
			plt.title(instances[i])
			plt.legend()
			plt.savefig("task4_" + th + "_" + str(i) + ".png", dpi=300, bbox_inches='tight')

file1.close()

