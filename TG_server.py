# The structure of the TGserver server
# THe TGserver should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the TCclient
# 3. Server sends the aggregated information back to TCclients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from processing import process_weights
from scipy import integrate
from sympy import *
import numpy as np
import math

class Server():

    def __init__(self, args,id,  cids, shared_layers):
        """
        id: TGserver id
        cids: ids of the TCclients under this TGserver
        receiver_buffer: buffer for the received updates from selected TCclients
        shared_state_dict: state dict for shared network
        id_registration: participated TCclients in this round of traning
        sample_registration: number of samples of the participated TCclients in this round of training
        all_trainsample_num: the training samples for all the TCclients under this TGserver
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the TGserver
        :param cids: Indexes of all the TCclients under this TGserver
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.g = []
        self.ec = np.array([np.random.uniform(0, 0.5)+ 0.0025*self.id*args.num_round, (1-np.random.uniform(0, 0.5))- 0.0025*self.id*args.num_round])
        self.gain =46
        self.d = np.random.uniform(0, 5)
        self.noise_power = -96
        self.SINR = self.gain *self.d ** (-2) / self.noise_power
        self.fading = 10 ** -14.71 * self.d ** -3.76  # large scaling fading in dB
        self.bandwidth = 20e+6  # allocated bandwidth in Hz
        self.rateu =  self.bandwidth *np.log2(1 + 10 ** (self.gain / 10) * self.fading / 10 ** (
                self.noise_power / 10))  # average distribution rate in bits/(sec*Hz)
        self.c = np.random.uniform(10, 20)
        self.f = np.random.uniform(1e+9, 2e+9)

    def refresh_TGserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def TCclient_register(self, TCclient):
        self.id_registration.append(TCclient.id)
        self.sample_registration[TCclient.id] = len(TCclient.train_loader.dataset)
        return None

    def receive_from_TCclient(self, TCclient_id, cshared_state_dict):
        self.receiver_buffer[TCclient_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict,self.g = process_weights(w = received_dict,
                                                 s_num= sample_num)
   
    def send_to_TCclient(self, TCclient):
        TCclient.receive_from_TGserver(copy.deepcopy(self.shared_state_dict))
        # self.Z = np.size(self.shared_state_dict)*16/1024/1024
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_TGserver(TGserver_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

