from similarfood101.similar_train import SimilarTrain
import matplotlib.pyplot as plt


class SimilarFood101TrainTask:
    @staticmethod
    def run():
        train = SimilarTrain()
        pairs, ys = train.get_train_loader()
        for pair, y in zip(pairs, ys):
            print(pair, y)
            pair, y = train.preprocessing([pair], [y])
            print(pair[0][0].shape)
            plt.imshow(pair[0][0])
            plt.imshow(pair[0][1])
            plt.show()

