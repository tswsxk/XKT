# coding: utf-8
# 2021/5/26 @ tongshiwei
from XKT.GKT import etl

from XKT import MGKT

batch_size = 16
train = etl("../../data/assistment_2009_2010/train.json", batch_size=batch_size)
valid = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)
test = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)

model = MGKT(
    hyper_params=dict(
        ku_num=124,
        graph="../../data/assistment_2009_2010/transition_graph.json",
        hidden_num=5,
    )
)
model.train(train, valid, end_epoch=2)
model.save("mgkt")

model = MGKT.from_pretrained("mgkt")
print(model.eval(test))
