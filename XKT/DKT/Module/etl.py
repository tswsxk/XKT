# coding: utf-8
# create by tongshiwei on 2019/4/12

import mxnet as mx
from gluonnlp.data import FixedBucketSampler, PadSequence
from tqdm import tqdm

from XKT.shared.etl import *


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    num_buckets = params.num_buckets
    batch_size = params.batch_size

    responses = raw_data

    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def index(r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct

    for batch_idx in tqdm(batch_idxes, "batchify"):
        batch_rs = []
        batch_pick_index = []
        batch_labels = []
        for idx in batch_idx:
            batch_rs.append([index(r) for r in responses[idx]])
            if len(responses[idx]) <= 1:
                pick_index, labels = [], []
            else:
                pick_index, labels = zip(*[(r[0], 0 if r[1] <= 0 else 1) for r in responses[idx][1:]])
            batch_pick_index.append(list(pick_index))
            batch_labels.append(list(labels))

        max_len = max([len(rs) for rs in batch_rs])
        padder = PadSequence(max_len, pad_val=0)
        batch_rs, data_mask = zip(*[(padder(rs), len(rs)) for rs in batch_rs])

        max_len = max([len(rs) for rs in batch_labels])
        padder = PadSequence(max_len, pad_val=0)
        batch_labels, label_mask = zip(*[(padder(labels), len(labels)) for labels in batch_labels])
        batch_pick_index = [padder(pick_index) for pick_index in batch_pick_index]
        batch.append(
            [mx.nd.array(batch_rs), mx.nd.array(data_mask), mx.nd.array(batch_labels),
             mx.nd.array(batch_pick_index),
             mx.nd.array(label_mask)])

    return batch


def etl(data_src, params):
    raw_data = extract(data_src)
    return transform(raw_data, params)


def pseudo_data_iter(_cfg):
    return transform(pseudo_data_generation(_cfg), _cfg)


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    import os

    filename = "../../../data/junyi/data/test"

    print(os.path.abspath(filename))

    for data in tqdm(extract(filename)):
        pass

    parameters = AttrDict({"batch_size": 128, "num_buckets": 100})
    for data in tqdm(etl(filename, params=parameters)):
        pass
