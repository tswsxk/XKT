# coding: utf-8
# create by tongshiwei on 2019-7-13

import json

from tqdm import tqdm

__all__ = ["extract"]


def extract(data_src):
    responses = []
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)[:200]
            if len(data) < 2:
                continue
            responses.append(data)

    return responses
