# coding: utf-8
# 2019/12/6 @ tongshiwei
import pytest
from XKT import DKT


@pytest.mark.parametrize("add_embedding_layer", [True, False])
def test_train(data, conf, tmpdir, add_embedding_layer):
    ku_num, hidden_num, batch_size = conf
    model = DKT(
        batch_size=batch_size,
        hyper_params={
            "ku_num": ku_num,
            "hidden_num": hidden_num,
            "add_embedding_layer": add_embedding_layer,
            "embedding_dim": hidden_num
        }
    )
    print(model.cfg)
    model.train(data, valid_data=data, end_epoch=1)
    filepath = tmpdir.mkdir("dkt")
    model.save(filepath)
    model = DKT.from_pretrained(filepath)
    model.eval(data)


@pytest.mark.parametrize("rnn_type", ["rnn", "lstm", "gru"])
def test_benchmark(train_file, conf, tmpdir, rnn_type):
    ku_num, hidden_num, batch_size = conf
    model_dir = str(tmpdir.mkdir("dkt"))
    DKT.benchmark_train(
        train_path=train_file,
        valid_path=train_file,
        enable_hyper_search=False,
        save=True,
        model_dir=model_dir,
        end_epoch=1,
        batch_size=batch_size,
        hyper_params={
            "rnn_type": rnn_type,
            "ku_num": ku_num,
            "hidden_num": hidden_num,
        },

    )
    DKT.benchmark_eval(train_file, model_dir, best_epoch=0)


def test_exception():
    with pytest.raises(TypeError):
        DKT(hyper_params={"ku_num": 10, "rnn_type": "error", "hidden_num": 10})
