import sys
import task
import logging
import time


class ArgError(Exception):
    def __init__(self):
        self.info = '请按如下格式输入参数：python main.py <task> [log_path], ' \
                    '若log_path不给出，程序将以:当地时间_task.txt为文件名存储于 ./logfile 文件夹'

    def __str__(self):
        return self.info


if __name__ == '__main__':
    # sys.argv = ['main.py', 'task.similar_food101_train_task.SimilarFood101TrainTask']
    # 参数检测
    if len(sys.argv) < 2:
        raise ArgError

    if len(sys.argv) >= 3:
        log_path = sys.argv[2]
    else:
        log_path = 'output/logfile/{}_{}.txt'.format(
            time.strftime('%Y-%m-%d_%H-%M-%S'),     # 当地时间
            sys.argv[1]                             # task
        )
    print('log_path:{}'.format(log_path))
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode='a'
    )
    logging.info("{} is run at {}".format(
        time.strftime('%Y/%m/%d %H:%M:%S'),
        sys.argv[1]
    ))
    exec(sys.argv[1] + '.run()')
