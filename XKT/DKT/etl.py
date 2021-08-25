# coding: utf-8
# 2021/8/22 @ tongshiwei

import mxnet.ndarray as nd
from tqdm import tqdm
from XKT.utils import extract
from baize.utils import FixedBucketSampler, PadSequence


def transform(raw_data, batch_size, num_buckets=100):
    # 定义数据转换接口
    # raw_data --> batch_data

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
            if len(responses[idx]) <= 1:  # pragma: no cover
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
        # Load
        batch.append(
            [nd.array(batch_rs), nd.array(data_mask), nd.array(batch_labels),
             nd.array(batch_pick_index),
             nd.array(label_mask)]
        )

    return batch


def etl(data_src, cfg=None, batch_size=None, **kwargs):  # pragma: no cover
    batch_size = batch_size if batch_size is not None else cfg.batch_size
    raw_data = extract(data_src)
    return transform(raw_data, batch_size, **kwargs)
