import numpy as np

from .dmp_discrete import DMPs_discrete

class DMP:
    def __init__(self,opt):
        self.opt = opt
        self.start_pos = [0.4, -0.15, 0.34]
        self.end_pos = [0.4,-0.15,0.64]
        self.n_bfs = 10
        self.frame_num = 20
        self.w_init = np.random.uniform (size=(3, self.n_bfs)) * 100.0

    def set_start(self,start_pos):
        self.start_pos = start_pos
        self.dmp = DMPs_discrete (dt=1./self.frame_num, n_dmps=3, n_bfs=self.n_bfs, w=self.w_init,
                                  y0=self.start_pos, goal=self.end_pos)

    def set_goal(self,end_pos):
        self.end_pos = end_pos
        self.dmp = DMPs_discrete (dt=1./self.frame_num, n_dmps=3, n_bfs=self.n_bfs, w=self.w_init,
                                  y0=self.start_pos, goal=self.end_pos)

    def set_params(self,w_init):
        self.w_init = w_init
        self.dmp = DMPs_discrete (dt=1. / self.frame_num, n_dmps=3, n_bfs=self.n_bfs, w=self.w_init,
                                  y0=self.start_pos, goal=self.end_pos)

    def get_traj(self):
        y_track, dy_track, ddy_track = self.dmp.rollout ()
        return y_track