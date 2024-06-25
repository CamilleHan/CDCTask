# -*- coding: utf-8 -*-
import numpy as np
dtype = np.float32
from scipy import special

class Env():
    def __init__(self, fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, max_b, p_n, power_num,args, TCclients, device):
        self.fd = fd
        self.Ts = Ts
        self.n_x = n_x
        self.n_y = n_y
        self.L = L
        self.C = C
        self.maxM = maxM   # user number in one BS
        self.min_dis = min_dis #km
        self.max_dis = max_dis #km
        self.max_p = max_p #dBm
        self.max_b = max_b  # dBm
        self.p_n = p_n     #dBm
        self.power_num = power_num
        self.TCclients = TCclients
        self.c = 3*self.L*(self.L+1) + 1 # adjascent BS
        self.K = self.maxM * self.c # maximum adjascent users, including itself
        self.N = self.n_x * self.n_y # BS number
        self.M = self.N * self.maxM # maximum users
        self.state_num = 5 * self.C + 1  + 3  # C + 1
        self.W = np.ones((self.M), dtype = dtype)
        self.sigma2 = 1e-3*pow(10., self.p_n/10.)
        self.maxP = 1e-3*pow(10., self.max_p/10.)
        self.p_array, self.p_list = self.generate_environment()
        self.num_TCclient_update = args.num_TCclient_update
        self.TCclients = TCclients
        self.batch_size= max_b

    def get_power_set(self, min_p):
        power_set = np.hstack([np.zeros((1), dtype=dtype), 1e-3*pow(10.,
                                    np.linspace(min_p, self.max_p, self.power_num-1)/10.)])
        return power_set

    def get_batch_set(self, min_b):
        batch_set = np.hstack([np.zeros((1), dtype=int),(np.linspace(min_b, max_b, self.power_num-1))])
        return batch_set
        
    def generate_next_state(self, R, H2, p_matrix, g, batch_size, rate_matrix, reliability, delay):
        sinr_norm_inv = H2[:,1:] / np.tile(H2[:,0:1], [1,self.K-1])
        sinr_norm_inv = np.log2(1. + sinr_norm_inv)   # log representation
        indices1 = np.tile(np.expand_dims(np.linspace(0, p_matrix.shape[0]-1, num=p_matrix.shape[0], dtype=np.int32), axis=1),[1,self.C])
        indices2 = np.argsort(sinr_norm_inv, axis = 1)[:,-self.C:]
        sinr_norm_inv = sinr_norm_inv[indices1, indices2]
        p_last = np.hstack([p_matrix[:,0:1], p_matrix[indices1, indices2+1]])
        rate_last = np.hstack([rate_matrix[:,0:1], rate_matrix[indices1, indices2+1]])
        rate_last = 1 / (1 + np.exp(- rate_last))
        rate_last = np.log(rate_last + 1)
        bandwidth = 15e3
        dalay_u_last= R/(bandwidth*rate_last+1)
        for i,TCclient in enumerate(self.TCclients):
            dalay_c_last = R / ((self.TCclients[i].f)) / self.TCclients[i].c
        delay_last = dalay_u_last+ dalay_c_last
        reliability_last = rate_matrix[:,0:1]/(rate_matrix[indices1, indices2+1]+1)
        g = g.reshape(-1,1)
        s_actor_next = np.hstack([sinr_norm_inv, p_last, g, rate_last,reliability_last,delay_last])
        s_critic_next = H2
        return s_actor_next, s_critic_next

    def reset(self,device,R):
        self.count = 0
        self.H2_set = self.generate_H_set()
        P = np.ones([self.M], dtype=dtype)
        H2, p_matrix, rate_matrix, rate, sum_rate, reward_rate, delay_u, reliability= self.calculate_rate(P,R)
        reliability = np.clip(reliability, 1e-3, 1 - 1e-3)
        g,loss,delay_c= self.calculate_gradient(device,R)
        delay_c = np.array(delay_c)
        delay = delay_u + delay_c
        H2 = self.H2_set[:,:,self.count]
        batch_size = self.batch_size
        s_actor, s_critic = self.generate_next_state(R,H2, p_matrix,g,batch_size,rate_matrix ,reliability, delay)
        delay1 = delay_u + delay_c
        delay = delay1*1e-2
        delay2 = 1.5*delay_u + delay_c
        return s_actor, s_critic,g, loss, rate, reliability, delay

        
    def step(self, P, batch_size, device,R):
        H2,p_matrix, rate_matrix, rate , sum_rate ,reward_rate, delay_u, reliability = self.calculate_rate(P,R)
        g,loss,delay_c= self.calculate_gradient(device,R)
        self.count = self.count + 1
        H2_next = self.H2_set[:,:,self.count]
        # batch_size = self.batch_size
        delay = delay_c + delay_u
        s_actor_next, s_critic_next = self.generate_next_state(R,H2_next, p_matrix, g, batch_size, rate_matrix,reliability,delay)
        reliability = np.clip(reliability, 1e-3, 1 - 1e-3)
        delay1 = delay_u + delay_c
        delay = delay1*1e-2
        delay2 = 1.5*delay_u + delay_c
        return s_actor_next, s_critic_next, reward_rate, sum_rate,rate, reliability, delay,g,loss

    def calculate_gradient(self, device,R):
        local_loss = [0.0] * self.M
        g = [0.0] * self.M
        delay_c  = [0.0] * self.M
        for i,TCclient in enumerate(self.TCclients):
            local_loss[i], g[i] = self.TCclients[i].local_update(num_iter=self.num_TCclient_update,
                                                      device=device)
            delay_c[i] = R / ((self.TCclients[i].f)) / self.TCclients[i].c
        g = np.array(g)
        local_loss= np.array(local_loss)
        return g, local_loss,delay_c
        
    def calculate_rate(self, P,R):
        maxC = 1000.
        H2 = self.H2_set[:,:,self.count]
        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)
        p_matrix = p_extend[self.p_array]
        path_main = H2[:,0] * p_matrix[:,0]
        path_inter = np.sum(H2[:,1:] * p_matrix[:,1:], axis=1)
        sinr = np.minimum(path_main / (path_inter + self.sigma2), maxC)    #capped sinr
        rate = self.W * np.log2(1. + sinr)
        sinr_norm_inv = H2[:,1:] / np.tile(H2[:,0:1], [1,self.K-1])
        sinr_norm_inv = np.log2(1. + sinr_norm_inv)   # log representation
        rate_extend = np.concatenate([rate, np.zeros((1), dtype=dtype)], axis=0)
        rate_matrix = rate_extend[self.p_array]
        reliability = 1- path_main/(path_main+path_inter+1)
        sum_rate = np.mean(rate)
        reward_rate = rate + np.sum(rate_matrix, axis=1)
        bandwidth = 15e3
        delay_u =  R / (bandwidth*rate+1)
        rate = 1 / (1 + np.exp(- rate))
        rate = np.log(rate + 1)
        return H2,p_matrix, rate_matrix, rate, sum_rate,reward_rate,delay_u,reliability
    





