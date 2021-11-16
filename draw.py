import matplotlib.pyplot as plt
from cgp import *
import numpy as np


current_best = []
all_time_best = []
with open('results/Assault-ram-v0621414442.3691218', 'r') as f:
  for line in f:
    # remove linebreak which is the last character of the string
    l = line[:-1]
    l = l.split(',')
    current_best.append(float(l[1]))
    all_time_best.append(float(l[2]))
    # add item to the list
    print(l)

plt.plot(current_best)
plt.xlabel('Iteraatio')
plt.ylabel('Pistemäärä')
plt.yticks(np.arange(0, max(current_best), 50))
plt.show()