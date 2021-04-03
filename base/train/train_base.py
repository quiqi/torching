import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import random
import logging
from base.network.lenet import LeNet

H_ARG = {
    'n_epochs': 3,
    'batch_size_train': 64,
    'batch_size_test': 1000,
    'learning_rate': 0.01,
    'momentum': 0.5,
    'random_seed': random.randint(0, 1000),
    'data_root': './data/'
}


class TrainBase:
    """
    训练器父类，所有的训练器都要继承该类
    用于提供一个训练方案的实例，以管理参数配置训练过程,如果你什么都不修改，他将运行一个MNIST手写数字识别的实现
    """
    def __init__(self, name='TrainBase', info='Simple training program.', h_args=None):
        """
        训练器初始化函数
        Args:
            name: 训练器名称
            info: 训练器介绍
            h_args: 训练器中要用到的超参数，你可以替换成自己需要的
        """
        if h_args is None:
            h_args = H_ARG
        self.name = name
        self.info = info
        self.h_args = h_args
        self.save_root = "output/save_pth/"
        torch.manual_seed(self.h_args['random_seed'])
        self.model = self.get_model()

        logging.info('==============init train====================')
        logging.info('init {}: {}'.format(self.name, self.info))
        logging.info('Hyperparameters:{}'.format(self.h_args))
        logging.info('============================================')

    def __str__(self):
        return "{}:{}".format(self.name, self.info)

    def get_train_loader(self):
        """
        训练数据读取函数，可以通过继承该类替换成自己需要的
        Returns:训练集迭代器
        """
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.h_args['data_root'], train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.h_args['batch_size_train'], shuffle=True)
        return train_loader

    def get_test_loader(self):
        """
        测试数据读取函数，可以通过继承该类替换成自己喜欢的
        Returns:测试集迭代器
        """
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.h_args['data_root'], train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.h_args['batch_size_test'], shuffle=True)
        return test_loader

    def preprocessing(self, x, label):
        """
        在数据放入迭代器之前，如果你需要进行修改，可以使用这个函数，默认不修改
        Args:
            x: 待放入神经网络的数据
            label: x的标签

        Returns:修改后的x和label
        """
        return x, label

    def get_model(self):
        """
        获得需要训练的模型
        Returns:需要训练的模型
        """
        return LeNet()

    def save_model(self, wight_name):
        torch.save(self.model.state_dict(), self.save_root+wight_name)

    def loss_fun(self):
        return nn.CrossEntropyLoss()

    def optim_fun(self):
        return optim.SGD(self.model.parameters(), lr=self.h_args['learning_rate'], momentum=self.h_args['momentum'])

    def run(self):
        cuda = torch.cuda.is_available()
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()
        self.model.train()
        if cuda:
            self.model.cuda()

        criterion = self.loss_fun()
        optimizer = self.optim_fun()
        all_loss_list = []
        test_loss_list = []
        for e in range(self.h_args['n_epochs']):
            all_loss = 0
            # train
            for batch_idx, (x, label) in enumerate(train_loader):
                x, label = self.preprocessing(x, label)
                if cuda:
                    x, label = x.cuda(), label.cuda()
                output = self.model(x)
                loss = criterion(output, label)
                all_loss += loss.item() * len(label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info('epoch:{}/{}\tbatch_idx:{}/{}, loss:{:.6f}'.format(
                    e, self.h_args['n_epochs'], batch_idx, len(train_loader), loss
                ))
            logging.info('\nepoch-{}-end\tall_loss:{:.6f}'.format(e, all_loss))
            all_loss_list.append(all_loss)

            # testing
            test_loss = 0
            for x, label in test_loader:
                x, label = self.preprocessing(x, label)
                if cuda:
                    x = x.cuda()
                    label = label.cuda()
                output = self.model(x)
                loss = criterion(output, label)
                test_loss += loss.item() * len(label)
            print('epoch-{}-test:loss:{:.6f}'.format(e, test_loss))
            test_loss_list.append(test_loss)
            self.save_model('epoch:{}_test-loss:{:.6f}.pth'.format(
                e, test_loss
            ))











