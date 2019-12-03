# coding: utf-8
# create by tongshiwei on 2019-7-13


from XKT import extract


def test_extract(train_dataset, test_dataset):
    for _ in extract(train_dataset):
        pass
    else:
        assert True
    for _ in extract(test_dataset):
        pass
    else:
        assert True
