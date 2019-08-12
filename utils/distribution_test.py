"""
reward function distribution illustration
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self):
        self.sample_num = 1000000

    def sample_43(self,mu=0.5,sigma=0.5):
        # mu, sigma = 0.5, .5
        s = np.random.normal(loc=mu, scale=sigma, size=self.sample_num)
        # count, bins, _ = plt.hist(s, 100, normed=True)
        return s

    def sample_45(self,mu=-0.5,sigma=0.5):
        # mu, sigma = -0.5, .5
        s = np.random.normal(loc=mu, scale=sigma, size=self.sample_num)
        # count, bins, _ = plt.hist(s, 100, normed=True)
        return s

    def sample_47(self,mu=0.3,sigma=0.5):
        # mu, sigma = -0.5, .5
        s = np.random.normal(loc=mu, scale=sigma, size=self.sample_num)
        # count, bins, _ = plt.hist(s, 100, normed=True)
        return s

    def my_split(self,s,start,end,num):
        count = [0] * num
        gap = float(end-start)/num
        bins = []
        bins.append(start)
        for i in tqdm(range(num)):
            start_i = start+gap*i
            end_i = start_i + gap
            count[i] += ((s>start_i) * (s<=end_i)).sum()/s.shape[0]
            bins.append(end_i)
        return count,bins

    def show2(self,var=0.5,id=0):
        start = -3
        end = 3
        show_bin_dim = 300
        plt.cla ()
        s43 = self.sample_43 (sigma=var)
        s45 = self.sample_45 (sigma=var)
        s = np.concatenate ((s43, s45), 0)
        count,bins = self.my_split(s,start,end,show_bin_dim)
        plt.plot (bins[:-1], count)
        plt.xlim ([-2, 2])
        plt.ylim ([0, 0.02])
        # plt.show()
        plt.savefig('./logs/before/before_{}.jpg'.format(id))

        count43,bins = self.my_split (s43, start, end, show_bin_dim)
        count45,bins = self.my_split (s45, start, end, show_bin_dim)

        new_count43 = np.zeros_like(count43)
        new_count45 = np.zeros_like(count45)

        for i in range(show_bin_dim):
            new_count43[i] = count43[i]*abs(count43[i]-count45[i])
            new_count45[i] = count45[i]*abs(count45[i]-count43[i])

        new_count43_sum = new_count43.sum()
        new_count45_sum = new_count45.sum()
        for i in range(show_bin_dim):
            new_count43[i] /= new_count43_sum
            new_count45[i] /= new_count45_sum

        for repeat_t in range(1):
            tmp = []
            for i in range(show_bin_dim):
                tmp.append(new_count43[i]+new_count45[i])
            count = np.array(tmp)

            plt.cla()
            plt.plot(bins[:-1],count)
            plt.xlim ([-2, 2])
            plt.ylim ([0, 0.04])
            # plt.show()
            plt.savefig ('./logs/after/after_{}.jpg'.format (id))

            count43 = np.copy(new_count43)
            count45 = np.copy(new_count45)

            for i in range(show_bin_dim):
                new_count43[i] = count43[i] * abs (count43[i] - count45[i])
                new_count45[i] = count45[i] * abs (count45[i] - count43[i])

            new_count43_sum = new_count43.sum ()
            new_count45_sum = new_count45.sum ()
            for i in range (show_bin_dim):
                new_count43[i] /= new_count43_sum
                new_count45[i] /= new_count45_sum


    def show3(self):
        start = -3
        end = 3
        show_bin_dim = 300
        plt.cla ()
        s43 = self.sample_43 ()
        s45 = self.sample_45 ()
        s47 = self.sample_47 (mu=0)
        s = np.concatenate ((s43, s45, s47), 0)
        # count, bins, _ = plt.hist (s, show_bin_dim, normed=True)
        count,bins = self.my_split(s,start,end,show_bin_dim)
        plt.plot (bins[:-1], count)

        plt.show()

        # count43, bins43, _ = plt.hist (s43, show_bin_dim, normed=True)
        # count45, bins45, _ = plt.hist (s45, show_bin_dim, normed=True)
        # count47, bins47, _ = plt.hist (s47, show_bin_dim, normed=True)

        count43,bins = self.my_split (s43, start, end, show_bin_dim)
        count45,bins = self.my_split (s45, start, end, show_bin_dim)
        count47,bins = self.my_split (s47, start, end, show_bin_dim)

        new_count43 = np.zeros_like(count43)
        new_count45 = np.zeros_like(count45)
        new_count47 = np.zeros_like(count47)
        for i in range(show_bin_dim):
            new_count43[i] = count43[i]*abs(2*count43[i]-count45[i]-count47[i])
            new_count45[i] = count45[i]*abs(2*count45[i]-count43[i]-count47[i])
            new_count47[i] = count47[i]*abs(2*count47[i]-count43[i]-count45[i])

        new_count43_sum = new_count43.sum ()
        new_count45_sum = new_count45.sum ()
        new_count47_sum = new_count47.sum ()
        for i in range (show_bin_dim):
            new_count43[i] /= new_count43_sum
            new_count45[i] /= new_count45_sum
            new_count47[i] /= new_count47_sum

        for repeat_t in range(2):
            tmp = []
            for i in range(show_bin_dim):
                tmp.append(new_count43[i]+new_count45[i]+new_count47[i])
            count = np.array(tmp)

            plt.cla()
            plt.plot(bins[:-1],count)
            plt.show()

            count43 = np.copy(new_count43)
            count45 = np.copy(new_count45)
            count47 = np.copy(new_count47)
            for i in range(show_bin_dim):
                new_count43[i] = count43[i]*abs(2*count43[i]-count45[i]-count47[i])
                new_count45[i] = count45[i]*abs(2*count45[i]-count43[i]-count47[i])
                new_count47[i] = count47[i]*abs(2*count47[i]-count43[i]-count45[i])

            new_count43_sum = new_count43.sum ()
            new_count45_sum = new_count45.sum ()
            new_count47_sum = new_count47.sum ()
            for i in range (show_bin_dim):
                new_count43[i] /= new_count43_sum
                new_count45[i] /= new_count45_sum
                new_count47[i] /= new_count47_sum



if __name__ == '__main__':
    agent = Distribution()
    for i in range(50,150):
        agent.show2(var=0.005*i,id=i)
        print(i)
    # agent.show3()
