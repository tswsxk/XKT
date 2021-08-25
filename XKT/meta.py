# coding: utf-8
# 2021/8/22 @ tongshiwei
from baize import path_append, get_params_filepath
from baize.const import CFG_JSON
from baize.mxnet import load_net, save_params, Configuration


class KTM(object):
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.net = None

    def __call__(self, *args, **kwargs):
        assert self.net

        return self.net(*args, **kwargs)

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, model_dir=None, *args, **kwargs) -> ...:
        model_dir = model_dir if model_dir is not None else self.cfg.model_dir
        select = kwargs.get("select", self.cfg.save_select)
        save_params(get_params_filepath(self.cfg.model_name, model_dir), self.net, select)
        self.cfg.dump(path_append(model_dir, CFG_JSON, to_str=True))
        return model_dir

    def load(self, model_path, *args, **kwargs) -> ...:
        load_net(model_path, self.net)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> ...:
        raise NotImplementedError

    @classmethod
    def benchmark_train(cls, *args, **kwargs) -> ...:
        raise NotImplementedError

    @classmethod
    def benchmark_eval(cls, *args, **kwargs) -> ...:
        raise NotImplementedError
