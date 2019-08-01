from dmp import DMPs
import numpy as np

class DMPs_rhythmic(DMPs):
  """An implementation of discrete DMPs."""
  def __init__(self, **kwargs):
    super(DMPs_rhythmic, self).__init__(pattern='rhythmic',**kwargs)
    
    self.gen_centers()
 
    # set variance of Gaussian basis functions
    # trial and error to find this spacing
    self.h = np.ones(self.n_bfs) * self.n_bfs

    self.check_offset()

  def gen_centers(self):
    """Set the centre of the Gassian basis.
    functions be spaced evenly throughout run time."""
    c = np.linspace(0, 2*np.pi, self.n_bfs + 1)
    c = c[0:-1]
    self.c = c

  def gen_front_term(self,x,dmp_num):
    """Generate the front term on the forcing term.
    For rhythmic DMPs it's non-diminshing, so this
    function is just a placeholder to return 1
   
    x float: the current value of the canonical system
    dmp_num int: the index of the current dmp
    """
    if isinstance(x, np.ndarray):
      return np.ones(x.shape) 
    return 1

  def gen_goal(self,y_des):
    """Generate the goal for path imitation.
    For rhythmic DMPs the goal is the average of the 
    desired trajectory.
    y_des np.array: the desired trajectory to follow
    """
    goal = np.zeros(self.n_dmps)
    for n in range(self.n_dmps):
      num_idx = ~np.isnan(y_des[n])
      goal[n] = .5 * (y_des[n, num_idx].min() + 
                      y_des[n, num_idx].max())

    return goal

  def gen_psi(self, x):
    """Generate the activity of the basis functions for a given canonical system state or path.
    x float, array: the canonical system state or path
    """
    if isinstance(x, np.ndarray):
      x = x[:,None]
    return np.exp(self.h * (np.cos(x - self.c) - 1))

  def gen_weights(self, f_target):
    """Generate a set of weights over the basis functions such that the target forcing term 
    trajectory is matched.
    f_target np.array: the desired forcing term trajectory
    """
    # calculate x and pis
    x_track = self.cs.rollout()
    psi_track = self.gen_psi(x_track)

    # efficiently calculate BF weights using weighted linear regression
    for d in range(self.n_dmps):
      for b in range(self.n_bfs):
        self.w[d, b] = (np.dot(psi_track[:,b], f_target[:,d])/(np.sum(psi_track[:,b]) + 1e-10))

##### 
# Test code
########

if __name__ == "__main__":
  import matplotlib.pyplot as plt
 
  # test normal run
  dmp = DMPs_rhythmic(n_dmps=1, n_bfs=10, w=np.zeros((1,10))) 
  y_track, dy_track, ddy_track = dmp.rollout()

  plt.figure(1,figsize=(6,3))
  plt.plot(np.ones(len(y_track)) * dmp.goal, 'r--', lw=2)
  plt.plot(y_track,lw=2)
  plt.title('DMP system -no forcing term')
  plt.xlabel('time (ms)')
  plt.ylabel('system trajectory')
  plt.legend(['goal','system state'],loc='lower right')
  plt.tight_layout()

  # test imitation of path run
  plt.figure(2,figsize=(6, 4))
  n_bfs = [10, 30, 50, 100, 10000]

  # a straight line to target
  path1 = np.sin(np.arange(0, 2*np.pi, 0.01)*5)

  # a strange path to target
  path2 = np.zeros(path1.shape)
  path2[int(len(path2) / 2.):] = .5

  for ii, bfs in enumerate(n_bfs):
    dmp = DMPs_rhythmic(n_dmps=2,n_bfs=bfs)

    dmp.imitate_path(y_des=np.array([path1,path2]))
    y_track, dy_track, ddy_track = dmp.rollout()

    plt.figure(2)
    plt.subplot(211)
    plt.plot(y_track[:,0],lw=2)
    plt.subplot(212)
    plt.plot(y_track[:,1],lw=2)

  plt.subplot(211)
  a = plt.plot(path1,'r--',lw=2)
  plt.title('DMP imitate path')
  plt.xlabel('time (ms)')
  plt.legend([a[0]],['desired path'],loc='lower right')
  plt.subplot(212)
 
  b = plt.plot(path2,'r--',lw=2)
  plt.title('DMP imitate path')
  plt.xlabel('time(ms)')
  plt.ylabel('system trajectory')
  plt.legend(['%i FBs' % i for i in n_bfs],loc='lower right')

  plt.tight_layout()
  plt.show()
