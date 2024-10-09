import torch
import torchvision
from torchvision.transforms import v2

import hydra
import torch.nn as nn
from torchinfo import summary

from train import train_model
from models.reskanet import reskanet_18x32p
from models.reskanet import resqkanet_18x32p


def get_data():
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.AutoAugment(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_val = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms_train)
    # Load and transform the CIFAR100 validation dataset
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms_val)
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="./configs/", config_name="cafar10-qreskanet.yaml")
def main(cfg):
    model = resqkanet_18x32p(4,
                            10,
                            groups=cfg.model.groups,
                            spline_order=int(cfg.model.spline_order),
                            grid_size=int(cfg.model.grid_size),
                            base_activation=nn.SiLU,
                            dropout=cfg.model.dropout,
                            l1_decay=cfg.model.l1_decay,
                            dropout_linear=cfg.model.dropout_linear)
    summary(model, (64, 3, 32, 32), device='cpu')
    dataset_train, dataset_test = get_data()
    loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

    train_model(model, dataset_train, dataset_test, loss_func, cfg)


if __name__ == '__main__':
    main()
