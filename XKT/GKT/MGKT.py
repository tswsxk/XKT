# coding: utf-8
# 2021/8/22 @ tongshiwei

from tqdm import tqdm
import mxnet as mx
from XKT.meta import KTM
from XKT.utils import SLMLoss
from baize import get_params_filepath, get_epoch_params_filepath, path_append
from baize.const import CFG_JSON
from baize.mxnet import light_module as lm, Configuration, fit_wrapper, split_and_load
from baize.metrics import classification_report
from .etl import etl
from .net import get_net


@fit_wrapper
def fit(net, batch_data, loss_function, *args, **kwargs):
    item_id, data, data_mask, label, next_item_id, label_mask = batch_data
    output, _ = net(item_id, data, data_mask)
    loss = loss_function(output, next_item_id, label, label_mask)
    return sum(loss)


def evaluation(net, test_data, ctx=mx.cpu(), *args, **kwargs):
    ground_truth = []
    prediction = []
    pred_labels = []

    for batch_data in tqdm(test_data, "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (item_id, data, data_mask, label, next_item_id, label_mask) in ctx_data:
            output, _ = net(item_id, data, data_mask)
            output = mx.nd.slice(output, (None, None), (None, -1))
            output = mx.nd.pick(output, next_item_id)
            pred = output.asnumpy().tolist()
            label = label.asnumpy().tolist()
            for i, length in enumerate(label_mask.asnumpy().tolist()):
                length = int(length)
                ground_truth.extend(label[i][:length])
                prediction.extend(pred[i][:length])
                pred_labels.extend([0 if p < 0.5 else 1 for p in pred[i][:length]])

    return classification_report(ground_truth, y_pred=pred_labels, y_score=prediction)


class MGKT(KTM):
    def __init__(self, init_net=True, cfg_path=None, *args, **kwargs):
        super(MGKT, self).__init__(Configuration(params_path=cfg_path, *args, **kwargs))
        if init_net:
            self.net = get_net(**self.cfg.hyper_params)

    def train(self, train_data, valid_data=None, re_init_net=False, enable_hyper_search=False,
              save=False, *args, **kwargs) -> ...:
        self.cfg.update(**kwargs)

        if not enable_hyper_search:
            print(self.cfg)

        lm.train(
            net=self.net,
            cfg=self.cfg,
            get_net=get_net if re_init_net is True else None,
            fit_f=fit,
            eval_f=evaluation,
            trainer=None,
            loss_function=SLMLoss(**self.cfg.loss_params),
            train_data=train_data,
            test_data=valid_data,
            enable_hyper_search=enable_hyper_search,
            dump_result=save,
            params_save=save,
            primary_key="macro_auc",
        )

    def eval(self, test_data, *args, **kwargs) -> ...:
        return evaluation(self.net, test_data, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_dir, best_epoch=None, *args, **kwargs):
        cfg_path = path_append(model_dir, CFG_JSON)
        model = MGKT(init_net=True, cfg_path=cfg_path, model_dir=model_dir)
        cfg = model.cfg
        model.load(
            get_epoch_params_filepath(cfg.model_name, best_epoch, cfg.model_dir)
            if best_epoch is not None else get_params_filepath(cfg.model_name, cfg.model_dir)
        )
        return model

    @classmethod
    def benchmark_train(cls, train_path, valid_path=None, enable_hyper_search=False,
                        save=False, *args, **kwargs):
        dkt = MGKT(init_net=not enable_hyper_search, *args, **kwargs)
        train_data = etl(train_path, dkt.cfg)
        valid_data = etl(valid_path, dkt.cfg) if valid_path is not None else None
        dkt.train(train_data, valid_data, re_init_net=enable_hyper_search, enable_hyper_search=enable_hyper_search,
                  save=save)

    @classmethod
    def benchmark_eval(cls, test_path, model_path, best_epoch, *args, **kwargs):
        dkt = MGKT.from_pretrained(model_path, best_epoch)
        test_data = etl(test_path, dkt.cfg)
        return dkt.eval(test_data)
