# coding: utf-8
# 2019/12/3 @ tongshiwei

from EduData import get_data
from longling import path_append
import functools
import pytest

test_url_dict = {
    "tests":
        "http://base.ustc.edu.cn/data/ktbd/synthetic/",
}

get_data = functools.partial(get_data, url_dict=test_url_dict)


@pytest.fixture(scope="session")
def root_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def dataset():
    return "synthetic"


@pytest.fixture(scope="session")
def root_data_dir(root_dir, dataset):
    dataset_dir = path_append(root_dir, "ktbd")
    dataset_dir = get_data("tests", dataset_dir)
    yield path_append(dataset_dir, dataset, to_str=True)


@pytest.fixture(scope="session")
def train_dataset(root_data_dir):
    return path_append(root_data_dir, "train.json", to_str=True)


@pytest.fixture(scope="session")
def test_dataset(root_data_dir):
    return path_append(root_data_dir, "test.json", to_str=True)
