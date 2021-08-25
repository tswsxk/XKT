# coding: utf-8
# 2021/5/26 @ tongshiwei
from XKT.SKT import etl

from XKT import MSKT

batch_size = 16
train = etl("../../data/assistment_2009_2010/train.json", batch_size=batch_size)
valid = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)
test = etl("../../data/assistment_2009_2010/test.json", batch_size=batch_size)

model = MSKT(
    hyper_params=dict(
        ku_num=124,
        graph_params=[
            ['../../data/assistment_2009_2010/correct_transition_graph.json', True],
            ['../../data/assistment_2009_2010/ctrans_sim.json', False]
        ],
        hidden_num=5,
    )
)
model.train(train, valid, end_epoch=2)
model.save("mskt")

model = MSKT.from_pretrained("mskt")
print(model.eval(test))
