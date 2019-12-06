# coding: utf-8
# Copyright @tongshiwei

from __future__ import absolute_import
from __future__ import print_function

import datetime
import pathlib

import longling.ML.MxnetHelper.glue.parser as parser
from longling.ML.MxnetHelper.glue.parser import path_append, var2exp, eval_var
from longling.ML.MxnetHelper.toolkit.select_exp import all_params as _select
from longling.lib.utilog import config_logging, LogLevel
from mxnet import cpu


class Configuration(parser.Configuration):
    # 目录配置
    model_name = str(pathlib.Path(__file__).parents[1].name)

    # root = pathlib.Path(__file__).parents[3]
    root = "./"
    dataset = ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    workspace = ""

    root_data_dir = "$root/data/$dataset" if dataset else "$root/data"
    data_dir = "$root_data_dir/data"
    root_model_dir = "$root_data_dir/model/$model_name"
    model_dir = "$root_model_dir/$workspace" if workspace else root_model_dir
    cfg_path = "$model_dir/configuration.json"

    root = str(root)
    root_data_dir = str(root_data_dir)
    root_model_dir = str(root_model_dir)

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 20
    batch_size = 16
    save_epoch = 1

    # 优化器设置
    # optimizer, optimizer_params = get_optimizer_cfg(name="base")
    optimizer = "adam"
    optimizer_params = {
        "learning_rate": 1e-3,
    }
    lr_params = None
    # {
    #     "learning_rate": 10e-3,
    #     "step": 100,
    #     "max_update_steps": get_update_steps(
    #         update_epoch=10,
    #         batches_per_epoch=1000,
    #     ),
    # }

    # 更新保存参数，一般需要保持一致
    train_select = _select
    save_select = train_select

    # 运行设备
    ctx = cpu(0)

    # 工具包参数
    toolbox_params = {}

    # 用户变量
    num_buckets = 100
    # 超参数
    hyper_params = {
    }
    loss_params = {
    }
    # 说明
    caption = ""

    def __init__(self, params_json=None, **kwargs):
        """
        Configuration File, including categories:

        * directory setting
        * optimizer setting
        * training parameters
        * equipment
        * parameters saving setting
        * user parameters

        Parameters
        ----------
        params_json: str
            The path to configuration file which is in json format
        kwargs:
            Parameters to be reset.
        """
        super(Configuration, self).__init__(
            logger=config_logging(
                logger=self.model_name,
                console_log_level=LogLevel.INFO
            )
        )

        params = self.class_var
        if params_json:
            params.update(self.load_cfg(params_json=params_json))
        params.update(**kwargs)

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        # set dataset
        if kwargs.get("dataset") and not kwargs.get("root_data_dir"):
            kwargs["root_data_dir"] = "$root/data/$dataset"
        # set workspace
        if kwargs.get("workspace") and not kwargs.get("root_model_dir"):
            kwargs["model_dir"] = "$root_model_dir/$workspace"

        # rebuild relevant directory or file path according to the kwargs
        _dirs = [
            "workspace", "root_data_dir", "data_dir", "root_model_dir",
            "model_dir"
        ]
        for _dir in _dirs:
            exp = var2exp(
                kwargs.get(_dir, getattr(self, _dir)),
                env_wrap=lambda x: "self.%s" % x
            )
            setattr(self, _dir, eval(exp))

        _vars = [
            "ctx"
        ]
        for _var in _vars:
            if _var in kwargs:
                try:
                    setattr(self, _var, eval_var(kwargs[_var]))
                except TypeError:
                    pass

        self.validation_result_file = path_append(
            self.model_dir, "result.json", to_str=True
        )
        self.cfg_path = path_append(
            self.model_dir, "configuration.json", to_str=True
        )

    def dump(self, cfg_path=None, override=False):
        cfg_path = self.cfg_path if cfg_path is None else cfg_path
        super(Configuration, self).dump(cfg_path, override)

    @staticmethod
    def load(cfg_path, **kwargs):
        return Configuration(Configuration.load_cfg(cfg_path, **kwargs))

    def var2val(self, var):
        return eval(var2exp(
            var,
            env_wrap=lambda x: "self.%s" % x
        ))


class ConfigurationParser(parser.ConfigurationParser):
    pass


def directory_check(class_obj):
    print("data_dir", class_obj.data_dir)
    print("model_dir", class_obj.model_dir)


if __name__ == '__main__':
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the paramters configuation file
    # to check the details (step 2)

    # step 1
    directory_check(Configuration)

    # 命令行参数配置
    _kwargs = ConfigurationParser.get_cli_cfg(Configuration)

    cfg = Configuration(
        **_kwargs
    )
    print(_kwargs)
    print(cfg)

    # # step 2
    cfg.dump(override=True)
    try:
        logger = cfg.logger
        cfg.load_cfg(cfg.cfg_path)
        cfg.logger = logger
        cfg.logger.info('format check done')
    except Exception as e:
        print("parameters format error, may contain illegal data type")
        raise e
