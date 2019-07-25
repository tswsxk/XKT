# coding: utf-8
# create by tongshiwei on 2019-7-13

import json

import mxnet as mx
from tqdm import tqdm
from gluonnlp.data import FixedBucketSampler, PadSequence

__all__ = ["extract", "transform", "etl"]


def extract(data_src):
    responses = []
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)[:200]
            if len(data) < 2:
                continue
            responses.append(data)

    return responses


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    num_buckets = params.num_buckets
    batch_size = params.batch_size

    responses = raw_data

    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def one_hot(r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct

    for batch_idx in tqdm(batch_idxes, "batchify"):
        batch_rs = []
        batch_pick_index = []
        batch_labels = []
        for idx in batch_idx:
            batch_rs.append([one_hot(r) for r in responses[idx]])
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
