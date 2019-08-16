# coding: utf-8
# create by tongshiwei on 2019-7-13

from longling.lib.structure import AttrDict
from tqdm import tqdm

from XKT import extract, etl


def test_extract():
    assert extract("../data/junyi/student_log_kt.json.train") is not None
    assert extract("../data/junyi/student_log_kt.json.test") is not None


def test_etl():
    params = AttrDict({
        "num_buckets": 100,
        "batch_size": 64,
    })
    for _ in tqdm(etl("../data/junyi/student_log_kt.json.train", params)):
        pass
    for _ in tqdm(etl("../data/junyi/student_log_kt.json.test", params)):
        pass
    assert True



