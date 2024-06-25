# Interface between the dataset and client
# For artificially partitioned dataset, params include num_clients, dataset

from datasets.cifar_mnist import get_dataset, show_distribution
def get_dataloaders(args, batch):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10','FMNIST','svhn']:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(dataset_root='data',
                                                                                       dataset=args.dataset,
                                                                                       args = args, batch_size=batch)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, v_train_loader, v_test_loader
