# coding: utf-8
# 2019/12/3 @ tongshiwei

from EduData import get_data
from longling import path_append
import functools
import pytest

test_url_dict = {
    "tests":
        "http://base.ustc.edu.cn/data/ktbd/assistment_2009_2010/",
}

get_data = functools.partial(get_data, url_dict=test_url_dict)


@pytest.fixture(scope="session")
def root_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def dataset_dir(root_dir):
    dataset_dir = path_append(root_dir, "ktbd")
    dataset_dir = get_data("tests", dataset_dir)
    yield path_append(dataset_dir, "assistment_2009_2010")


@pytest.fixture(scope="session")
def train_dataset(dataset_dir):
    return path_append(dataset_dir, "train.json", to_str=True)


@pytest.fixture(scope="session")
def test_dataset(dataset_dir):
    return path_append(dataset_dir, "test.json", to_str=True)
