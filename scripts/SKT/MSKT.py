# coding: utf-8
# 2021/8/25 @ tongshiwei

from fire import Fire
from baize import path_append
from EduData import get_data
from XKT import MSKT

DATASET = {
    "a0910": (
        "ktbd-a0910",
        "assistment_2009_2010/train.json",
        "assistment_2009_2010/valid.json",
        "assistment_2009_2010/test.json",
        [
            ['assistment_2009_2010/correct_transition_graph.json', True],
            ['assistment_2009_2010/ctrans_sim.json', False]
        ],
        dict(
            ku_num=124,
            hidden_num=16,
        )
    )
}


def get_dataset_and_config(dataset, train=None, valid=None, test=None, hyper_params=None):
    if dataset in DATASET:
        dataset, train, valid, test, graphs, hyper_params = DATASET[dataset]

        data_dir = "../../data"
        for graph in graphs:
            graph[0] = path_append(data_dir, graph[0], to_str=True)
    else:
        data_dir = dataset
        assert hyper_params
        graphs = None

    get_data(dataset, data_dir)

    if graphs is not None:
        hyper_params["graph_params"] = graphs

    train = path_append(data_dir, train)
    valid = path_append(data_dir, valid)
    test = path_append(data_dir, test)

    return train, valid, test, hyper_params


def run(mode, model, dataset, epoch, train_path=None, valid_path=None, test_path=None, *args, **kwargs):
    train, valid, test, hyper_params = get_dataset_and_config(dataset, train_path, valid_path, test_path)
    loss_params = {}
    if mode in {"hs", "train"}:
        hyper_params.update({"net_type": model.upper()})
        MSKT.benchmark_train(
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
        print(MSKT.benchmark_eval(test, model, epoch))
    else:
        raise ValueError("unknown mode %s" % mode)


if __name__ == '__main__':
    Fire(run)

    # run("train", "dkvmn", "a0910c", 2)
