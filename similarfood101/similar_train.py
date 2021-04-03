from base.train.train_base import *
from similarfood101.similar_loss_fun import *
from similarfood101.similar_net import *
from similarfood101.network.lenet import LeNet
import similarfood101.tools.creat_pairs as scp
import matplotlib.image as mpimg


class SimilarTrain(TrainBase):
    def __init__(self):
        super().__init__(name='SimilarTrain', info='Scheme for training similarity networks')
        self.h_args['data_root'] = '/home/quqi/projects/毕业设计/big_dataset/Food101/archive/'
        self.h_args['n_pairs'] = 10000
        self.h_args['rate_pairs'] = 0.5

    def get_train_loader(self):
        xs = []
        with open(self.h_args['data_root']+'meta/meta/train.txt', 'r') as f:
            line = f.readline()
            while line:
                xs.append(line[:-1])
                line = f.readline()
        labels = []
        for x in xs:
            labels.append(x.split('/')[0])
        pairs, ys = scp.create_pairs(xs, labels, self.h_args['n_pairs'], self.h_args['rate_pairs'])
        return pairs, ys

    def get_test_loader(self):
        xs = []
        with open(self.h_args['data_root'] + 'meta/meta/text.txt', 'r') as f:
            line = f.readline()
            while line:
                xs.append(line[:-1])
                line = f.readline()
        labels = []
        for x in xs:
            labels.append(x.split('/')[0])
        return xs, labels

    def preprocessing(self, xs, labels):
        pics = []
        for x in xs:
            pic1 = mpimg.imread(self.h_args['data_root']+'images/'+x[0]+'.jpg')
            pic2 = mpimg.imread(self.h_args['data_root']+'images/'+x[1]+'.jpg')
            pics.append([pic1, pic2])
        return pics, labels

    def get_model(self):
        return SiameseTrainNet(LeNet())

    def save_model(self, wight_name):
        torch.save(self.model.net.state_dict(), self.save_root+'siamese'+wight_name)
