# coding: utf-8
# 2019/12/6 @ tongshiwei

from XKT.DKT.DKT import DKT


def test_train(root_data_dir, dataset):
    DKT.run(
        [
            "train", "$data_dir/train.json", "$data_dir/test.json",
            "--workspace", "DKT",
            "--hyper_params",
            "nettype=DKT;ku_num=int(50);hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)",
            # "--loss_params", "lw2=float(1e-100)",
            "--end_epoch", "int(1)",
            "--ctx", "cpu(0)",
            "--dataset", dataset,
            "--root_data_dir", root_data_dir,
            "--data_dir", "$root_data_dir"
        ]
    )
