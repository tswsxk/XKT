# coding: utf-8
# create by tongshiwei on 2019-9-1
import mxnet as mx
import mxnet.ndarray as nd
from longling.ML.MxnetHelper.toolkit.ctx import split_and_load
from mxnet import autograd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm


def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    keys, values, data_mask, label, label_mask = _data
    output, _ = _net(keys, values, mask=data_mask)
    bp_loss = None
    for name, func in loss_function.items():
        loss = func(output, label, label_mask)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = nd.mean(loss).asscalar()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def eval_f(_net, test_data, ctx=mx.cpu()):
    ground_truth = []
    prediction = []
    pred_labels = []

    for batch_data in tqdm(test_data, "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (keys, values, data_mask, label, label_mask) in ctx_data:
            output, _ = _net(keys, values, mask=data_mask)
            pred = output.asnumpy().tolist()
            label = label.asnumpy().tolist()
            for i, length in enumerate(label_mask.asnumpy().tolist()):
                length = int(length)
                ground_truth.extend(label[i][:length])
                prediction.extend(pred[i][:length])
                pred_labels.extend([0 if p < 0.5 else 1 for p in pred[i][:length]])

    auc = roc_auc_score(ground_truth, prediction)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, pred_labels)

    evaluation_result = {}

    evaluation_result.update(
        {"precision_%d" % i: precision[i] for i in range(len(precision))}
    )
    evaluation_result.update(
        {"recall_%d" % i: recall[i] for i in range(len(recall))}
    )
    evaluation_result.update(
        {"f1_%d" % i: f1[i] for i in range(len(f1))}
    )

    evaluation_result.update({"auc": auc})
    return evaluation_result


def fit_f(net, batch_size, batch_data,
          trainer, bp_loss_f, loss_function, loss_monitor=None,
          ctx=mx.cpu()
          ):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_size: int
            The size of each batch
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    bp_loss_f: dict with only one value and one key
        The function to compute the loss for the procession
        of back propagation
    loss_function: dict of function
        Some other measurement in addition to bp_loss_f
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.

    Returns
    -------

    """
    # 此函数定义训练过程
    ctx_data = split_and_load(
        ctx, *batch_data,
        even_split=False
    )

    with autograd.record():
        for _data in ctx_data:
            bp_loss = _fit_f(
                net, _data, bp_loss_f, loss_function, loss_monitor
            )
            assert bp_loss is not None
            bp_loss.backward()
    trainer.step(1)
