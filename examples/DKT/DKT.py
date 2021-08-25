# coding: utf-8
# 2021/5/26 @ tongshiwei
import mxnet as mx
from XKT.DKT import etl

from XKT import DKT

batch_size = 32
train = etl("../../data/a0910c/train.json", batch_size=batch_size)
valid = etl("../../data/a0910c/valid.json", batch_size=batch_size)
test = etl("../../data/a0910c/test.json", batch_size=batch_size)

model = DKT(hyper_params=dict(ku_num=146, hidden_num=100))
model.train(train, valid, end_epoch=2)
model.save("dkt")

model = DKT.from_pretrained("dkt")
print(model.eval(test))

inputs = mx.nd.ones((2, 3))
outputs, _ = model(inputs)
print(outputs)
