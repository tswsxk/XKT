# coding: utf-8
# create by tongshiwei on 2019-7-13

import json

from tqdm import tqdm

__all__ = ["extract", "extract_iter"]


def extract(data_src, max_step=200):
    responses = []
    step = max_step
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            if step is not None:
                for i in range(0, len(data), step):
                    if len(data[i: i + step]) < 2:
                        continue
                    responses.append(data[i: i + step])
            else:
                responses.append(data)

    return responses


def extract_iter(data_src):
    step = 200
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            for i in range(0, len(data), step):
                if len(data[i: i + step]) < 2:
                    continue
                yield data[i: i + step]
