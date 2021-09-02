from sklearn import metrics
import numpy as np


def evaluate_multi_cls(pred, label):
    '''
    :param pred: numpy  [n]
    :param label: numpy  [n]
    :return:
    '''
    # metrics 中 0 为不进行预测  所以 pred label 均加1
    # pred = pred + 1
    # label = label + 1
    # print('pred = ', pred.shape, pred)
    # print('label = ', label.shape, label)
    acc = np.mean((pred == label).astype(int))
    micro_f1 = metrics.f1_score(label, pred, average='micro')
    macro_f1 = metrics.f1_score(label, pred, average='macro')
    micro_p = metrics.precision_score(label, pred, average='macro')
    micro_r = metrics.recall_score(label, pred, average='macro')

    return acc, micro_f1, macro_f1, micro_p, micro_r
