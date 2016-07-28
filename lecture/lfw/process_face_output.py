import numpy as np
import sys

def process_output(filename):
  lst = np.array([float(line.strip().split(' ')[-1]) for line in open(filename)])
  threshold = np.linspace(lst.min(), lst.max(), 1000)
  acc = np.array([((lst[:500] < th).mean() + (lst[500:] > th).mean())/2 for th in threshold])
  print acc.max()

process_output(sys.argv[1])
