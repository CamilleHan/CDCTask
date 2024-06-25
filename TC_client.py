# The structure of the TCclient
# Should include following funcitons
# 1. TCclient intialization, dataloaders, model(include optimizer)
# 2. TCclient model update
# 3. TCclient send updates to server
# 4. TCclient receives updates from server
# 5. TCclient modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from NNmodels.initialize_model import initialize_model
import copy
import numpy as np
import math
class TCclient():
    def __init__(self, id, train_loader, test_loader, args, batch_size, device):
        self.id = id
        self.num_TCclient_update=args.num_TCclient_update
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        #record local update epoch
        self.epoch = 0
        self.clock = []
        self.cc =  np.array([np.random.uniform(0, 0.5)+ 0.0025*self.id*args.num_round, (1-np.random.uniform(0, 0.5))- 0.0025*self.id*args.num_round])
        self.c = np.random.uniform(1, 10)
        self.f = np.random.uniform(0.5e+9, 1e+9)

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        g=0.0
        end = False
        for epoch in range(self.num_TCclient_update):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss0, g = self.model.optimize_model(input_batch=inputs, label_batch=labels)
                loss += loss0
                g = np.array(g)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch = self.epoch)
        loss /= num_iter
        g = np.abs(np.linalg.norm(g))
        return loss, g

    def prox_local_update(self, args,num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        for epoch in range(self.num_TCclient_update):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                batchloss = self.model.optimize_proxmodel(args,input_batch=inputs,
                                            label_batch=labels)
                loss +=batchloss
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    self.epoch += 1
                    self.model.step_lr_scheduler(epoch=self.epoch)
                    break
            if end: break
            self.epoch += 1
            self.model.step_lr_scheduler(epoch=self.epoch)

        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_TGserver(self, TGserver):
        TGserver.receive_from_TCclient(TCclient_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        received_dict = [dict for dict in self.receiver_buffer.values()]
        return received_dict,self.model.shared_layers.state_dict()

    def receive_from_TGserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None
    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

    def sync_with_TGserver(self):
        self.model.update_model(self.receiver_buffer)
        return None

