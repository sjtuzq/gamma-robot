import numpy as np
import os
import matplotlib.pyplot as plt
from .dmp_discrete import DMPs_discrete

class DMP:
    def __init__(self,opt):
        self.opt = opt
        self.start_pos = [0.4, -0.15, 0.34]
        self.end_pos = [0.4,-0.15,0.64]
        self.n_bfs = 200
        # self.frame_num = self.opt.cut_frame_num+1+8
        self.frame_num = self.opt.dmp_num + 1
        self.w_init = np.random.uniform (size=(3, self.n_bfs)) * 100.0
        self.dmp = DMPs_discrete (dt=1. / self.frame_num, n_dmps=3, n_bfs=self.n_bfs, w=self.w_init,
                                  y0=self.start_pos, goal=self.end_pos)


    def set_start(self,start_pos):
        self.start_pos = start_pos
        for i in range(3):
            self.dmp.y0[i] = self.start_pos[i]

    def set_goal(self,end_pos):
        self.end_pos = end_pos
        for i in range(3):
            self.dmp.goal[i] = self.end_pos[i]

    def get_traj(self):
        y_track, dy_track, ddy_track = self.dmp.rollout ()
        actions = []
        for i in range(y_track.shape[0]):
            if i==0:
                continue
            actions.append([x-y for x,y in zip(y_track[i],y_track[i-1])])
        return np.array(actions)[:self.opt.cut_frame_num]

    def imitate(self,trajectories):
        for traj in trajectories:
            traj = traj[:125,:]
            self.set_start(traj[0])
            self.set_goal(traj[-1])
            self.dmp.imitate_path (y_des=np.array ([traj[:,0], traj[:,1],traj[:,2]]))
            traj_path = os.path.join(self.opt.project_root,'scripts','Dmp','traj.npy')
            np.save(traj_path,traj)
