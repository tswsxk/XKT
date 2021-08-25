# coding: utf-8
# 2021/5/26 @ tongshiwei
import mxnet as mx
from XKT.DKVMN import etl

from XKT import DKVMN

batch_size = 32
train = etl("../../data/a0910c/train.json", batch_size=batch_size)
valid = etl("../../data/a0910c/valid.json", batch_size=batch_size)
test = etl("../../data/a0910c/test.json", batch_size=batch_size)

model = DKVMN(
    hyper_params=dict(
        ku_num=146,
        key_embedding_dim=10,
        value_embedding_dim=10,
        key_memory_size=20,
        hidden_num=100
    )
)
model.train(train, valid, end_epoch=2)
model.save("dkvmn")

model = DKVMN.from_pretrained("dkvmn")
print(model.eval(test))


inputs = mx.nd.ones((2, 3))
outputs, _ = model(inputs)
print(outputs)