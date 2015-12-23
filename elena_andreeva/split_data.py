# -*- coding: utf8 -*-

import sys
import os


def k_fold_cross_validation(X, K, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.
    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise: from random import shuffle; X = list(X); shuffle(X)
    for k in xrange(0, K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation, k


if __name__ == '__main__':
    source = sys.argv[1]

    ds_name = source.split(".")[0]
    os.mkdir(ds_name)

    with open(source, 'r') as data:
        for training, validation, k in k_fold_cross_validation(data.readlines(), K=10, randomise=True):
            tf = open("%s/train_%d.csv" % (ds_name, k + 1), 'w')
            tf.writelines(training)
            tv = open("%s/test_%d.csv" % (ds_name, k + 1), 'w')
            tv.writelines(validation)
            tf.close()
            tv.close()
