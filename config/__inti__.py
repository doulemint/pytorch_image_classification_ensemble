from .defaults import get_default_config
import torch


def update_config(config):
    
    if config.optim == "adam":  # clw note: not stable
        config.lr_scheduler = "cosine_change_per_epoch"  # lr scheduler method: "step", "cosine_change_per_epoch", "cosine_change_per_batch", "adjust","on_loss","on_acc",    adjust不需要配置这里的epoch和lr
        #lr = 1e-4  # adam: 1e-4, 3e-4, 5e-4
        config.lr = 1e-4
        config.epochs = 10
    elif config.optim == "sgd":
        config.lr_scheduler = "step"
        #lr_scheduler = "cosine_change_per_epoch"
        if "vit" in config.model_name:
            config.lr = 1e-2
        elif "resnet50" in config.model_name or "resnext" in config.model_name:
            config.lr = 2e-2
        else:
            config.lr = 1e-1
    else:
        config.lr = 1e-3
        config.lr_scheduler = None
        config.epochs = 15
        config.lr_scheduler = "cosine_change_per_epoch"
    # if config.dataset.name in ['CIFAR10', 'CIFAR100']:
    #     dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
    #     config.dataset.dataset_dir = dataset_dir
    #     config.dataset.image_size = 32
    #     config.dataset.n_channels = 3
    #     config.dataset.n_classes = int(config.dataset.name[5:])
    # elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
    #     dataset_dir = '~/.torch/datasets'
    #     config.dataset.dataset_dir = dataset_dir
    #     config.dataset.image_size = 28
    #     config.dataset.n_channels = 1
    #     config.dataset.n_classes = 10

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
