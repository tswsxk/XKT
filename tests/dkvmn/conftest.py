# coding: utf-8
# 2021/8/23 @ tongshiwei
import pytest

import json
from baize import as_out_io, path_append
from XKT.utils.tests import pseudo_data_generation
from XKT.DKVMN.etl import transform


@pytest.fixture(scope="package")
def conf():
    item_num = 10
    hidden_num = 10
    batch_size = 32
    return item_num, hidden_num, batch_size


@pytest.fixture(scope="package")
def pseudo_data(conf):
    ques_num, *_ = conf
    return pseudo_data_generation(ques_num)


@pytest.fixture(scope="package")
def data(pseudo_data, conf):
    *_, batch_size = conf
    return transform(pseudo_data, batch_size)


@pytest.fixture(scope="package")
def train_file(pseudo_data, tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")
    filepath = path_append(data_dir, "data.json", to_str=True)
    with as_out_io(filepath) as wf:
        for line in pseudo_data:
            print(json.dumps(line), file=wf)
    return filepath
