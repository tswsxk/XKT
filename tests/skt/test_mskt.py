# coding: utf-8
# 2019/12/6 @ tongshiwei

import pytest
from XKT import MSKT


@pytest.mark.parametrize("net_type", ["SKT", "SKT_TE", "SKTPart", "SKTSync"])
def test_benchmark(train_file, graphs, conf, tmpdir, net_type):
    ku_num, hidden_num, batch_size = conf
    model_dir = str(tmpdir.mkdir("mskt"))
    MSKT.benchmark_train(
        train_path=train_file,
        valid_path=train_file,
        enable_hyper_search=False,
        save=True,
        model_dir=model_dir,
        end_epoch=1,
        batch_size=batch_size,
        hyper_params={
            "ku_num": ku_num,
            "hidden_num": hidden_num,
            "graph_params": graphs,
            "net_type": net_type
        },
    )
    MSKT.benchmark_eval(train_file, model_dir, best_epoch=0)
