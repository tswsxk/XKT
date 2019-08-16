# coding: utf-8
# create by tongshiwei on 2019-8-15

import mxnet as mx
from mxnet import gluon

if __name__ == '__main__':
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    print(loss(mx.nd.ones((16, 20, 100)), mx.nd.ones((16, 20))))