import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def gen_ran_sum(_sum, num_users):
    base = 100*np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100*num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users

def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iid_esize_split(dataset, args, kwargs, batch_size , is_shuffle = True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(dataset)
    num_samples_per_TCclient = int(sum_samples / args.num_TCclients)
    # change from dict to list
    data_loaders = [0] * args.num_TCclients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_TCclients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_TCclient, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def iid_nesize_split(dataset, args, kwargs, batch_size , is_shuffle = True):
    sum_samples = len(dataset)
    num_samples_per_TCclient = gen_ran_sum(sum_samples, args.num_TCclients)
    # change from dict to list
    data_loaders = [0] * args.num_TCclients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_TCclient) in enumerate(num_samples_per_TCclient):
        dict_users[i] = np.random.choice(all_idxs, num_samples_TCclient, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def niid_esize_split(dataset, args, kwargs, batch_size , is_shuffle = True):
    data_loaders = [0] * args.num_TCclients
    # each TCclient has only two classes of the network
    num_shards = 2* args.num_TCclients
    # the number of images in one shard
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    for i in range(args.num_TCclients):
        rand_set = set(np.random.choice(idx_shard, 2, replace= False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = batch_size,
                                    shuffle = is_shuffle, **kwargs)
    return data_loaders

def niid_esize_split_train(dataset, args, kwargs, batch_size , is_shuffle = True):
    data_loaders = [0]* args.num_TCclients
    num_shards = args.classes_per_TCclient * args.num_TCclients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    split_pattern = {i: [] for i in range(args.num_TCclients)}
    for i in range(args.num_TCclients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test(dataset, args, kwargs, split_pattern, batch_size , is_shuffle = False):
    data_loaders = [0] * args.num_TCclients
    num_shards = args.classes_per_TCclient * args.num_TCclients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    for i in range(args.num_TCclients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_train_large(dataset, args, kwargs, batch_size , is_shuffle = True):
    data_loaders = [0]* args.num_TCclients
    num_shards = args.classes_per_TCclient * args.num_TCclients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_TCclients)}
    for i in range(args.num_TCclients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, batch_size , is_shuffle = False):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each TCclient
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_TCclients
    num_shards = 10
    num_imgs = int (len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(len(dataset))
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    for i in range(args.num_TCclients):
        rand_set = split_pattern[i]
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_oneclass(dataset, args, kwargs, batch_size , is_shuffle = True):
    data_loaders = [0] * args.num_TCclients
    num_shards = args.num_TCclients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_TCclients)}
    idxs = np.arange(num_shards * num_imgs)
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    for i in range(args.num_TCclients):
        rand_set = set(np.random.choice(idx_shard, 1, replace = False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand+1)*num_imgs]), axis = 0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                            batch_size = batch_size,
                            shuffle = is_shuffle, **kwargs)
    return data_loaders

def split_data(dataset, args, kwargs, batch_size , is_shuffle = True):
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, batch_size, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, batch_size, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, batch_size, is_shuffle)
    elif args.iid == -2:
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, batch_size, is_shuffle)
    else :
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders

def get_dataset(dataset_root, dataset, args, batch_size):
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args,batch_size)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args,batch_size)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader

def get_mnist(dataset_root, args, batch_size):
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = True,
                            download = True, transform = transform)
    test =  datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = False,
                            download = True, transform = transform)
    train_loaders = split_data(train, args, kwargs, batch_size = batch_size,is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs,batch_size = batch_size, is_shuffle = False)
    v_train_loader = DataLoader(train, batch_size = batch_size* args.num_TCclients,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = batch_size* args.num_TCclients,
                                shuffle = False, **kwargs)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader

def get_cifar10(dataset_root, args,batch_size):
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory':True} if is_cuda else{}
    # if args.model == 'cnn_complex':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train = True,
                        download = True, transform = transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root,'cifar10'), train = False,
                        download = True, transform = transform_test)
    v_train_loader = DataLoader(train, batch_size = batch_size* args.num_TCclients,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = batch_size* args.num_TCclients,
                                shuffle = False, **kwargs)
    train_loaders = split_data(train, args, kwargs, batch_size = batch_size,is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, batch_size = batch_size,is_shuffle = False)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    if args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'cifar10':
        labels =np.array(dataloader.dataset.dataset.targets)

    num_samples = len(dataloader.dataset)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)

    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution1 = np.array(distribution)
    distribution2 = distribution1 / num_samples
    return distribution1,distribution2
if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_TCclients} TCclients/tasks in an iid = {args.iid} way")
    for i in range(args.num_TCclients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)
