import sys
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(BASE_DIR,'../')
DMP_DIR = os.path.join(ROOT_DIR,'dynamicMotionPrimitives')
sys.path.insert(0,DMP_DIR)
print(DMP_DIR)
from dmp_discrete import DMPs_discrete

if __name__ == "__main__":
  ### a straight line to target
  n_bfs=10
  w_init = np.random.uniform(size=(3,n_bfs)) * 1000.0
  dmp = DMPs_discrete(dt=.05, n_dmps=3, n_bfs=n_bfs, w=w_init, y0=[0.2,0.2,0.4],goal=[0.3,0.3,0.3])
  y_track, dy_track, ddy_track = dmp.rollout()
  print(y_track[0,:],y_track[-1,:])
  plt.figure(0)
  plt.subplot(311)
  plt.plot(y_track[:,0],lw=2)
  plt.subplot(312)
  plt.plot(y_track[:,1],lw=2)
  plt.subplot(313)
  plt.plot(y_track[:,2],lw=2)
  plt.show()
