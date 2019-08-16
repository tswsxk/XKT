# coding: utf-8
# create by tongshiwei on 2019-7-30

__all__ = ["SequenceLogisticMaskLoss", "LogisticMaskLoss"]

from mxnet import gluon


class SequenceLogisticMaskLoss(gluon.HybridBlock):
    """
    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """

    def __init__(self, lwr=0.0, lw1=0.0, lw2=0.0, **kwargs):
        super(SequenceLogisticMaskLoss, self).__init__(**kwargs)
        self.lwr = lwr
        self.lw1 = lw1
        self.lw2 = lw2
        with self.name_scope():
            self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    def hybrid_forward(self, F, pred_rs, pick_index, label, label_mask, *args, **kwargs):
        if self.lw1 > 0.0 or self.lw2 > 0.0:
            post_pred_rs = F.slice(pred_rs, (None, 1), (None, None))
            pre_pred_rs = F.slice(pred_rs, (None, None), (None, -1))
            diff = post_pred_rs - pre_pred_rs
            _weight_mask = F.squeeze(
                F.SequenceMask(F.expand_dims(F.ones_like(pre_pred_rs), -1), sequence_length=label_mask,
                               use_sequence_length=True, axis=1)
            )
            diff = _weight_mask * diff
            w1 = F.mean(F.norm(diff, 1, -1)) / diff.shape[-1]
            w2 = F.mean(F.norm(diff, 2, -1)) / diff.shape[-1]
            # w2 = F.mean(F.sqrt(diff ** 2))
            w1 = w1 * self.lw1 if self.lw1 > 0.0 else 0.0
            w2 = w2 * self.lw2 if self.lw2 > 0.0 else 0.0
        else:
            w1 = 0.0
            w2 = 0.0

        if self.lwr > 0.0:
            re_pred_rs = F.slice(pred_rs, (None, 1), (None, None))
            re_pred_rs = F.pick(re_pred_rs, pick_index)
            re_weight_mask = F.squeeze(
                F.SequenceMask(F.expand_dims(F.ones_like(re_pred_rs), -1), sequence_length=label_mask,
                               use_sequence_length=True, axis=1)
            )
            wr = self.loss(re_pred_rs, label, re_weight_mask)
            wr = F.mean(wr) * self.lwr
        else:
            wr = 0.0

        pred_rs = F.slice(pred_rs, (None, None), (None, -1))
        pred_rs = F.pick(pred_rs, pick_index)
        weight_mask = F.squeeze(
            F.SequenceMask(F.expand_dims(F.ones_like(pred_rs), -1), sequence_length=label_mask,
                           use_sequence_length=True, axis=1)
        )
        loss = self.loss(pred_rs, label, )
        # loss = F.sum(loss, axis=-1)
        loss = F.mean(loss) + w1 + w2 + wr
        return loss


class LogisticMaskLoss(gluon.HybridBlock):
    """
        Notes
        -----
        The loss has been average, so when call the step method of trainer, batch_size should be 1
        """

    def __init__(self, **kwargs):
        super(LogisticMaskLoss, self).__init__(**kwargs)

        with self.name_scope():
            self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    def hybrid_forward(self, F, pred_rs, label, label_mask, *args, **kwargs):
        weight_mask = F.squeeze(
            F.SequenceMask(F.expand_dims(F.ones_like(pred_rs), -1), sequence_length=label_mask,
                           use_sequence_length=True, axis=1)
        )
        loss = self.loss(pred_rs, label, weight_mask)
        loss = F.mean(loss)
        return loss
