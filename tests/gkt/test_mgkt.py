# coding: utf-8
# 2019/12/6 @ tongshiwei

from XKT import MGKT


def test_benchmark(train_file, graph_file, conf, tmpdir):
    ku_num, hidden_num, batch_size = conf
    model_dir = str(tmpdir.mkdir("mgkt"))
    MGKT.benchmark_train(
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
            "graph": graph_file
        },
    )
    MGKT.benchmark_eval(train_file, model_dir, best_epoch=0)
