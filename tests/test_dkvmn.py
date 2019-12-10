# coding: utf-8
# 2019/12/10 @ tongshiwei


from XKT.DKVMN import DKVMN


def test_train(root_data_dir, dataset):
    DKVMN.run(
        [
            "train", "$data_dir/train.json", "$data_dir/test.json",
            "--workspace", "DKVMN",
            "--hyper_params",
            "nettype=DKVMN;ku_num=int(50);hidden_num=int(100);key_embedding_dim=int(50);value_embedding_dim=int(200);"
            "key_memory_size=int(5);value_memory_size=int(5);"
            "key_memory_state_dim=int(50);value_memory_state_dim=int(200);"
            "dropout=float(0.5)",
            "--end_epoch", "int(1)",
            "--ctx", "cpu(0)",
            "--dataset", dataset,
            "--root_data_dir", root_data_dir,
            "--data_dir", "$root_data_dir"
        ]
    )
