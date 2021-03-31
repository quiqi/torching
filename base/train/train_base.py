import logging
import torch
import torchvision
from torch.utils.data import DataLoader
import random
import logging

H_ARG = {
    'n_epochs': 3,
    'batch_size_train': 64,
    'batch_size_test': 1000,
    'learning_rate': 0.01,
    'momentum': 0.5,
    'random_seed': random.randint(0,1000)
}

class TrainBase:
    """
    训练器父类，所有的训练器都要继承该类
    用于提供一个训练方案的实例，以管理参数配置训练过程,如果你什么都不修改，他将运行一个MNIST手写数字识别的实现
    """
    def __init__(self, name='TrainBase', info='Simple training program.', h_args=H_ARG):
        if h_args is None:
            h_args = H_ARG
        self.name = name
        self.info = info
        self.h_args = h_args
        torch.manual_seed(self.h_args['random_seed'])

        logging.info('==============init train====================')
        logging.info('init {}: {}'.format(self.name, self.info))
        logging.info('Hyperparameters:{}'.format(self.h_args))
        logging.info('============================================')

    def __str__(self):
        return "{}:{}".format(self.name, self.info)

    def get_train_load(self):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.h_args['batch_size_train'], shuffle=True)
        return train_loader

    def get_test_loader(self):
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.h_args['batch_size_test'], shuffle=True)
        return test_loader


