# coding: utf-8
# 2019/12/6 @ tongshiwei
import pytest
from XKT import DKVMN


def test_train(data, conf, tmpdir):
    ku_num, hidden_num, batch_size = conf
    model = DKVMN(
        batch_size=batch_size,
        hyper_params=dict(
            ku_num=ku_num,
            key_embedding_dim=hidden_num,
            value_embedding_dim=hidden_num,
            hidden_num=hidden_num,
            key_memory_size=hidden_num)
    )
    print(model.cfg)
    model.train(data, valid_data=data, end_epoch=1)
    filepath = tmpdir.mkdir("dkvmn")
    model.save(filepath)
    model = DKVMN.from_pretrained(filepath)
    model.eval(data)


def test_benchmark(train_file, conf, tmpdir):
    ku_num, hidden_num, batch_size = conf
    model_dir = str(tmpdir.mkdir("dkvmn"))
    DKVMN.benchmark_train(
        train_path=train_file,
        valid_path=train_file,
        enable_hyper_search=False,
        save=True,
        model_dir=model_dir,
        end_epoch=1,
        batch_size=batch_size,
        hyper_params=dict(
            ku_num=ku_num,
            key_embedding_dim=hidden_num,
            value_embedding_dim=hidden_num,
            hidden_num=hidden_num,
            key_memory_size=hidden_num
        )
    )
    DKVMN.benchmark_eval(train_file, model_dir, best_epoch=0)
