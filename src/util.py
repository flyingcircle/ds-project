import numpy as np

def load_data(filename):
  return np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1)