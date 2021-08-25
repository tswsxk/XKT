# coding: utf-8
# 2021/5/26 @ tongshiwei
from fire import Fire
from baize import path_append
from EduData import get_data
from XKT import DKVMN

DATASET = {
    "a0910c": (
        "ktbd-a0910c",
        "a0910c/train.json",
        "a0910c/valid.json",
        "a0910c/test.json",
        dict(
            ku_num=146,
            key_embedding_dim=10,
            value_embedding_dim=10,
            key_memory_size=20,
            hidden_num=100
        )
    )
}


def get_dataset_and_config(dataset, train=None, valid=None, test=None, hyper_params=None):
    if dataset in DATASET:
        dataset, train, valid, test, hyper_params = DATASET[dataset]

        data_dir = "../../data"
    else:
        data_dir = dataset
        hyper_params = {} if hyper_params is None else hyper_params

    get_data(dataset, data_dir)

    train = path_append(data_dir, train)
    valid = path_append(data_dir, valid)
    test = path_append(data_dir, test)
    return train, valid, test, hyper_params


def run(mode, model, dataset, epoch, train_path=None, valid_path=None, test_path=None, *args, **kwargs):
    train, valid, test, hyper_params = get_dataset_and_config(dataset, train_path, valid_path, test_path)
    loss_params = {}
    if mode in {"hs", "train"}:
        DKVMN.benchmark_train(
            train,
            valid,
            enable_hyper_search=True if mode == "hs" else False,
            end_epoch=epoch,
            loss_params=loss_params,
            hyper_params=hyper_params,
            save=False if mode == "hs" else True,
            model_dir=model,
            model_name=model,
            *args, **kwargs
        )
    elif mode == "test":
        print(DKVMN.benchmark_eval(test, model, epoch))
    else:
        raise ValueError("unknown mode %s" % mode)


if __name__ == '__main__':
    Fire(run)

    # run("train", "dkvmn", "a0910c", 2)
