# coding: utf-8
# create by tongshiwei on 2019-7-30

__all__ = ["SequenceLogisticMaskLoss"]

from mxnet import gluon

class SequenceLogisticMaskLoss(gluon.HybridBlock):
    """
    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """

    def __init__(self, **kwargs):
        super(SequenceLogisticMaskLoss, self).__init__(**kwargs)

        with self.name_scope():
            self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    def hybrid_forward(self, F, pred_rs, pick_index, label, label_mask, *args, **kwargs):
        pred_rs = F.slice(pred_rs, (None, None), (None, -1))
        pred_rs = F.pick(pred_rs, pick_index)
        weight_mask = F.squeeze(
            F.SequenceMask(F.expand_dims(F.ones_like(pred_rs), -1), sequence_length=label_mask,
                           use_sequence_length=True, axis=1)
        )
        loss = self.loss(pred_rs, label, weight_mask)
        # loss = F.sum(loss, axis=-1)
        loss = F.mean(loss)
        return loss
