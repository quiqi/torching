import random


def sorting(keys: list):
    """
    将索引按key值分类
    Args:
        keys: 要分类的key值列表

    Returns:

    """
    dic = {}
    for i, key in enumerate(keys):
        if key in dic.keys():
            dic[key].append(i)
        else:
            dic[key] = [i]
    index_list = []
    for key in dic:
        index_list.append(dic[key])
    return index_list


def create_pairs(xs: list, labels: list, n: int, r: float):
    """
    生成n对数据，其中有比例为r的正例
    Args:
        xs: 训练样本
        labels: 样本标签
        n: 生成n对数据
        r: 生成数据中正例的比例
    Returns: 一个(pairs, ys)元组，pairs表示生成的数据对列表， ys为该数据对的标签。
    """
    index_list = sorting(labels)
    pairs = []
    ys = []
    for i in range(n):
        if random.random() < r:
            # 选择一个类别
            class_id = random.randint(0, len(index_list)-1)

            # 选择该类别两个不同的样本
            index1 = random.randint(0, len(index_list[class_id])-1)
            index2 = random.randint(0, len(index_list[class_id])-1)
            while index1 == index2:
                index2 = random.randint(0, len(index_list[class_id]) - 1)
            index1 = index_list[class_id][index1]
            index2 = index_list[class_id][index2]
        else:
            # 选择两个不同的类别
            class_id1 = random.randint(0, len(index_list)-1)
            class_id2 = random.randint(0, len(index_list)-1)
            while class_id1 == class_id2:
                class_id2 = random.randint(0, len(index_list)-1)

            # 在每个类别中分别选择一个个体
            index1 = random.randint(0, len(index_list[class_id1])-1)
            index2 = random.randint(0, len(index_list[class_id2])-1)
            index1 = index_list[class_id1][index1]
            index2 = index_list[class_id2][index2]

        x1 = xs[index1]
        x2 = xs[index2]
        pairs.append([x1, x2])

        y1 = labels[index1]
        y2 = labels[index2]
        ys.append([y1, y2])

    return pairs, ys
