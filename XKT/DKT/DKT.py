# coding: utf-8
# 2021/8/22 @ tongshiwei

from tqdm import tqdm
import mxnet as mx
from mxnet import gluon
from XKT.meta import KTM
from XKT.utils import SLMLoss
from baize import get_params_filepath, get_epoch_params_filepath, path_append
from baize.const import CFG_JSON
from baize.mxnet import light_module as lm, Configuration, fit_wrapper, split_and_load
from baize.metrics import classification_report
from .etl import etl


class DKTNet(gluon.HybridBlock):
    def __init__(self, ku_num, hidden_num,
                 add_embedding_layer=False, embedding_dim=None, embedding_dropout=None,
                 dropout=0.0, rnn_type=None,
                 prefix=None, params=None, **kwargs):
        """
        Deep Knowledge Tracing Model

        Parameters
        ----------
        ku_num: int
            Number of knowledge units
        hidden_num : int
            Number of units in output symbol of rnn
        add_embedding_layer: bool
            Whether add embedding layer
        embedding_dim: int or None
            When embedding_dim is None, the embedding_dim will be equal to hidden_num
        embedding_dropout: float or None
            When not set, be equal to dropout
        dropout: float
            Fraction of the input units to drop. Must be a number between 0 and 1.
        rnn_type: str or None
            rnn, lstm or gru
        prefix : str
            Prefix for name of `Block`s
        params : Parameter or None
            Container for weight sharing between cells.
            Created if `None`.
        """
        super(DKTNet, self).__init__(prefix, params)

        self.length = None
        self.ku_num = ku_num
        self.hidden_dim = hidden_num
        self.add_embedding_layer = add_embedding_layer

        with self.name_scope():
            if add_embedding_layer is True:
                embedding_dim = self.hidden_dim if embedding_dim is None else embedding_dim
                embedding_dropout = dropout if embedding_dropout is None else embedding_dropout
                self.embedding = gluon.nn.HybridSequential()
                self.embedding.add(
                    gluon.nn.Embedding(ku_num * 2, embedding_dim),
                    gluon.nn.Dropout(embedding_dropout)
                )
                cell = gluon.rnn.LSTMCell
            else:
                self.embedding = lambda x, F: F.one_hot(x, ku_num * 2)
                cell = gluon.rnn.RNNCell

            if rnn_type is not None:
                if rnn_type in {"elman", "rnn"}:
                    cell = gluon.rnn.RNNCell
                elif rnn_type == "lstm":
                    cell = gluon.rnn.LSTMCell
                elif rnn_type == "gru":
                    cell = gluon.rnn.GRUCell
                else:
                    raise TypeError("unknown rnn type: %s" % rnn_type)

            self.rnn = gluon.rnn.HybridSequentialRNNCell()
            self.rnn.add(
                cell(hidden_num),
            )
            self.dropout = gluon.nn.Dropout(dropout)
            self.nn = gluon.nn.HybridSequential()
            self.nn.add(
                gluon.nn.Dense(ku_num, flatten=False)
            )

    def hybrid_forward(self, F, responses, mask=None, begin_state=None, *args, **kwargs):
        length = self.length if self.length else len(responses[0])

        if self.add_embedding_layer:
            input_data = self.embedding(responses)
        else:
            input_data = self.embedding(responses, F)

        outputs, states = self.rnn.unroll(length, input_data, begin_state=begin_state, merge_outputs=True,
                                          valid_length=mask)

        output = self.nn(self.dropout(outputs))
        output = F.sigmoid(output)
        return output, states


@fit_wrapper
def fit(net, batch_data, loss_function, *args, **kwargs):
    data, data_mask, label, pick_index, label_mask = batch_data
    output, _ = net(data, data_mask)
    loss = loss_function(output, pick_index, label, label_mask)
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
        for (data, data_mask, label, pick_index, label_mask) in ctx_data:
            output, _ = net(data, data_mask)
            output = mx.nd.slice(output, (None, None), (None, -1))
            output = mx.nd.pick(output, pick_index)
            pred = output.asnumpy().tolist()
            label = label.asnumpy().tolist()
            for i, length in enumerate(label_mask.asnumpy().tolist()):
                length = int(length)
                ground_truth.extend(label[i][:length])
                prediction.extend(pred[i][:length])
                pred_labels.extend([0 if p < 0.5 else 1 for p in pred[i][:length]])

    return classification_report(ground_truth, y_pred=pred_labels, y_score=prediction)


def get_net(**kwargs):

    return DKTNet(**kwargs)


class DKT(KTM):
    """
    Examples
    --------
    >>> import mxnet as mx
    >>> model = DKT(init_net=True, hyper_params={"ku_num": 3, "hidden_num": 5})
    >>> model.net.initialize()
    >>> inputs = mx.nd.ones((2, 4))
    >>> outputs, (states, *_) = model(inputs)
    >>> outputs.shape
    (2, 4, 3)
    >>> states.shape
    (2, 5)
    >>> outputs, (states, *_) = model(inputs, begin_state=[states])
    >>> outputs.shape
    (2, 4, 3)
    >>> states.shape
    (2, 5)
    """
    def __init__(self, init_net=True, cfg_path=None, *args, **kwargs):
        super(DKT, self).__init__(Configuration(params_path=cfg_path, *args, **kwargs))
        if init_net:
            self.net = get_net(**self.cfg.hyper_params)

    def __call__(self, x, mask=None, begin_state=None):
        return super(DKT, self).__call__(x, mask, begin_state)

    def train(self, train_data, valid_data=None, re_init_net=False, enable_hyper_search=False,
              save=False, *args, **kwargs) -> ...:
        self.cfg.update(**kwargs)

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
            primary_key="macro_auc"
        )

    def eval(self, test_data, *args, **kwargs) -> ...:
        return evaluation(self.net, test_data, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_dir, best_epoch=None, *args, **kwargs):
        cfg_path = path_append(model_dir, CFG_JSON)
        model = DKT(init_net=True, cfg_path=cfg_path, model_dir=model_dir)
        cfg = model.cfg
        model.load(
            get_epoch_params_filepath(cfg.model_name, best_epoch, cfg.model_dir)
            if best_epoch is not None else get_params_filepath(cfg.model_name, cfg.model_dir)
        )
        return model

    @classmethod
    def benchmark_train(cls, train_path, valid_path=None, enable_hyper_search=False,
                        save=False, *args, **kwargs):
        dkt = DKT(init_net=not enable_hyper_search, *args, **kwargs)
        train_data = etl(train_path, dkt.cfg)
        valid_data = etl(valid_path, dkt.cfg) if valid_path is not None else None
        dkt.train(train_data, valid_data, re_init_net=enable_hyper_search, enable_hyper_search=enable_hyper_search,
                  save=save)

    @classmethod
    def benchmark_eval(cls, test_path, model_path, best_epoch, *args, **kwargs):
        dkt = DKT.from_pretrained(model_path, best_epoch)
        test_data = etl(test_path, dkt.cfg)
        return dkt.eval(test_data)
