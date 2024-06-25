import random
import matplotlib.pyplot as plt
import networkx as nx
import math
import torch
import copy
import numpy as np
import time
from tqdm import tqdm
from options import args_parser
from TC_client import TCclient
from TG_server import Server
from datasets.get_data import get_dataloaders
from NNmodels.cifar_lenet import LeNet
from NNmodels.mnist_logistic import LogisticRegression
from NNmodels.mnist_lenet import mnistlenet


# Initialize seed for reproducibility
random.seed(10)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_TCclient_class(args, TCclients):
    """Classify TC clients' data distribution."""
    TCclient_class = []
    TCclient_class_dis = [[] for _ in range(10)]
    for TCclient in TCclients:
        train_loader = TCclient.train_loader

def get_TGserver_class(args, TGservers, TCclients):
    """Classify TG servers' data distribution."""
    TGserver_class = [[] for _ in range(len(TGservers))]
    for i, TGserver in enumerate(TGservers):
        for cid in TGserver.cids:
            TCclient = TCclients[cid]
            train_loader = TCclient.train_loader

def initialize_TGservers_iid(num_TGservers, TCclients, args, TCclient_class_dis):
    """Initialize TG servers with IID data distribution."""
    TGservers = []
    p_TCclients = [0.0] * num_TGservers
    for eid in range(num_TGservers):
        if eid == num_TGservers - 1:
            break
        assigned_TCclients_idxes = []
        for label in range(10):
            assigned_TCclient_idx = np.random.choice(TCclient_class_dis[label], 1, replace=False)
            for idx in assigned_TCclient_idx:
                assigned_TCclients_idxes.append(idx)
            TCclient_class_dis[label] = list(set(TCclient_class_dis[label]) - set(assigned_TCclient_idx))
        TGservers.append(Server(id=eid, cids=assigned_TCclients_idxes, shared_layers=copy.deepcopy(TCclients[0].model.shared_layers)))
        [TGservers[eid].TCclient_register(TCclients[TCclient]) for TCclient in assigned_TCclients_idxes]
        TGservers[eid].all_trainsample_num = sum(TGservers[eid].sample_registration.values())
        p_TCclients[eid] = [sample / float(TGservers[eid].all_trainsample_num) for sample in list(TGservers[eid].sample_registration.values())]
        TGservers[eid].refresh_TGserver()
    # Initialize the last TG server
    eid = num_TGservers - 1
    assigned_TCclients_idxes = []
    for label in range(10):
        if not TCclient_class_dis[label]:
            print(f"label {label} is empty")
        else:
            assigned_TCclient_idx = TCclient_class_dis[label]
            for idx in assigned_TCclient_idx:
                assigned_TCclients_idxes.append(idx)
            TCclient_class_dis[label] = list(set(TCclient_class_dis[label]) - set(assigned_TCclient_idx))
    TGservers.append(Server(id=eid, cids=assigned_TCclients_idxes, shared_layers=copy.deepcopy(TCclients[0].model.shared_layers)))
    [TGservers[eid].TCclient_register(TCclients[TCclient]) for TCclient in assigned_TCclients_idxes]
    TGservers[eid].all_trainsample_num = sum(TGservers[eid].sample_registration.values())
    p_TCclients[eid] = [sample / float(TGservers[eid].all_trainsample_num) for sample in list(TGservers[eid].sample_registration.values())]
    TGservers[eid].refresh_TGserver()
    return TGservers, p_TCclients

def all_TCclients_test(server, TCclients, cids, device):
    """Test on TC clients."""
    [server.send_to_TCclient(TCclients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_TCclient(TCclients[cid])
        TCclients[cid].sync_with_TGserver()
    correct_TGserver = 0.0
    total_TGserver = 0.0
    for cid in cids:
        correct, total = TCclients[cid].test_model(device)
        correct_TGserver += correct
        total_TGserver += total
    return correct_TGserver, total_TGserver

def fast_all_TCclients_test(v_test_loader, aggregated_nn, device):
    """Quickly test on TC clients."""
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = aggregated_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def initialize_aggregated_nn(args):
    """Initialize the neural network based on dataset and model type."""
    if args.dataset == 'mnist':
        if args.model == 'logistic':
            aggregated_nn = LogisticRegression(input_dim=1, output_dim=10)
        elif args.model == 'mnistlenet':
            aggregated_nn = mnistlenet(args)
        else:
            raise ValueError(f"Model {args.model} not implemented for MNIST")
    elif args.dataset == 'cifar10':
        if args.model == 'lenet':
            aggregated_nn = LeNet(args)
        else:
            raise ValueError(f"Model {args.model} not implemented for CIFAR-10")
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")
    return aggregated_nn

def CDCtask(args):
    """Execute the CDC task based on the provided arguments."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    aggregated_nn = initialize_aggregated_nn(args)
    if args.cuda:
        aggregated_nn = aggregated_nn.cuda(device)

    TGservers = [[] for _ in range(args.num_TGservers)]
    TCclients = [[] for _ in range(args.num_TCclients)]
    bat = 50
    fd = 10
    Ts = 20e-3
    n_x = 1
    n_y = 5
    L = 2
    C = 16
    maxM = 20  # user number
    min_dis = 0.01  # km
    max_dis = 1.0  # km
    max_p = 38 # dBm
    max_b=512
    p_n = -114.0  # dBm
    power_num = maxM  # action_num

    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)
    Agents_list = []
    reward_lists_of_list = []
    mean_reward_lists_of_list = []
    critic_loss_list = []
    actor_loss_list = []
    target_list = []
    critic_list = []
     
    Ns = args.num_round * args.num_TGserver_update

    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args, bat)
    for i in range(args.num_TCclients):
        TCclients[i] = TCclient(id=i, train_loader=train_loaders[i], test_loader=test_loaders[i], args=args, batch_size=bat, device=device)

    env = Env(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p,  max_b, p_n, power_num, args, TCclients, device)
    env.set_Ns(Ns)

    state_num = env.state_num
    action_num = env.power_num

    for n in range(args.num_TGservers):
        Agents_list.append(agent(state_dim=state_num, action_dim=action_num, max_action=action_num, batch_size=max_b))
        reward_lists_of_list.append([])
        mean_reward_lists_of_list.append([])
        critic_loss_list.append([])
        actor_loss_list.append([])
        target_list.append([])
        critic_list.append([])
    glob = np.ones(args.num_TGservers) * 1e-5
    execution_time_cost = 0

    for num_comm in tqdm(range(args.num_round)):
        avg_acc = np.zeros(args.num_TGservers)
        num = int(args.num_TCclients/args.num_TGservers)
        a = np.ones(args.num_TCclients,dtype=int)
        p = np.ones((args.num_TGservers,num))
        b = np.ones((args.num_TGservers,num),dtype=int)
        reward_hist = np.zeros((args.num_TCclients, args.num_TGservers))
        R = bat * 28 * 28
        s_actor, _, g, loss, rate, reliability, delay = env.reset(device, R)
        for num_TGserveragg in range(args.num_TGserver_update):
            TGserver_sample = [0] * args.num_TGservers
            correct_all = 0.0
            total_all = 0.0
            selected_cids = [[] for _ in range(args.num_TGservers)]
            cids = [[] for _ in range(args.num_TGservers)]
            rewardlist = np.zeros((maxM, args.num_TGservers))
            s_actor_t = s_actor.copy()

            for i, TGserver in enumerate(TGservers):
                rr = 0.4 - i * 1e-4
                rl = 0.9 - i * 0.02
                rd = 1.8e+3 - i * 1e2
                c = np.zeros(maxM)
                s_actor_agent = s_actor_t[i * maxM:(i + 1) * maxM]
                tg = g[i * maxM: (i + 1) * maxM]

                trate = rate[i * maxM: (i + 1) * maxM]
                treliability = reliability[i * maxM: (i + 1) * maxM]
                tdelay = delay[i * maxM: (i + 1) * maxM]

                for j in range(len(s_actor_agent)):
                    o = tg * (np.log(sigmoid(trate * treliability * tdelay)) + 1) ** 2
                    if trate[j] >= rr and treliability[j] >= rl and tdelay[j] <= rd and (np.abs(c[j] - glob[i])) <= 2.5:
                        s_actor_t[j] = s_actor_t[j]
                    else:
                        s_actor_t[j] = -1
                    rewardlist[j, i] = args.mu * (rd - tdelay[j]) / 5e2 + (1 - args.mu) * (o[j] / 30)
                agent = Agents_list[i]
                a_agent, a_agent_1 = agent.choose_action(s_actor_agent, i + 1)
                selected_cids[i] = (a_agent_1 + i * maxM).tolist()
                cids[i] = a_agent_1
                p[i] = np.squeeze(a_agent* (max_p / 10))
                b[i] = np.squeeze(a_agent  * (max_b / (2**(10))))
                s_actor_t[a[i * maxM:(i + 1) * maxM]] = -1
            mp = p.reshape(-1)
            mb = b.reshape(-1)
            s_actor_next, s_critic_next, _, _, rate, reliability, delay, g, loss = env.step(mp, mb, device, R)


            for i in range(args.num_TGservers):
                TGservers[i] = Server(args, id=i, cids=selected_cids[i], shared_layers=copy.deepcopy(TCclients[0].model.shared_layers))
                [TGservers[i].TCclient_register(TCclients[cid]) for cid in selected_cids[i]]
                TGservers[i].all_trainsample_num = sum(TGservers[i].sample_registration.values())

            for i, TGserver in enumerate(TGservers):
                reward_hist[i * maxM:(i + 1) * maxM, :] = rewardlist
            for i, TGserver in enumerate(TGservers):
                agent = Agents_list[i]
                TGserver.erate = np.sum(rate[cid] for cid in selected_cids[i]) / len(selected_cids[i])
                TGserver.ereliability = np.sum(reliability[cid] for cid in selected_cids[i]) / len(selected_cids[i])
                TGserver.edelay = np.sum(delay[cid] for cid in selected_cids[i]) / len(selected_cids[i])
                TGserver.reward = reward_hist[selected_cids[i], i]
                TGserver.reward = np.asarray(TGserver.reward)
                TGserver.sumreward = np.sum(TGserver.reward) / len(selected_cids[i])
                agent.store_transition(s_actor[cids[i]], cids[i], TGserver.reward, s_actor_next[cids[i]])
            s_actor = s_actor_next

            for i, TGserver in enumerate(TGservers):
                for selected_cid in TGserver.cids:
                    TGserver.TCclient_register(TCclients[selected_cid])
                    TCclients[selected_cid].send_to_TGserver(TGserver)
                TGserver.aggregate(args)

            for i, TGserver in enumerate(TGservers):
                TGserver_sample[i] = sum(TGserver.sample_registration.values())
                correct, total = all_TCclients_test(TGserver, TCclients, TGserver.cids, device)
                correct_all += correct
                total_all += total

                for selected_cid in TGserver.cids:
                    TGserver.send_to_TCclient(TCclients[selected_cid])
                    TCclients[selected_cid].sync_with_TGserver()
                acc = correct_all / total_all
                avg_acc[i] = acc

            erate = 0
            ereliability = 0
            edelay = 0
            egradient = 0
            ereward = 0

            for i, TGserver in enumerate(TGservers):
                agent = Agents_list[i]
                agent.learn()
                ereward += TGserver.sumreward
                erate += TGserver.erate
                ereliability += TGserver.ereliability
                edelay += TGserver.edelay
                egradient += (3e2*acc/TGserver.g)



def main():
    args = args_parser()
    CDCtask(args)

if __name__ == '__main__':
    main()
