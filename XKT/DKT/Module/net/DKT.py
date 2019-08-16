# coding: utf-8
# create by tongshiwei on 2019-7-30

__all__ = ["DKTNet"]

from mxnet import gluon


class DKTNet(gluon.HybridBlock):
    def __init__(self, ku_num, hidden_num, nettype="DKT", dropout=0.0, **kwargs):
        super(DKTNet, self).__init__(kwargs.get("prefix"), kwargs.get("params"))

        self.length = None
        self.nettype = nettype

        with self.name_scope():
            if nettype == "EmbedDKT":
                latent_dim = kwargs["latent_dim"]
                embedding_dropout = kwargs.get("embedding_dropout", 0.2)
                self.embedding = gluon.nn.Embedding(2 * ku_num, latent_dim)
                self.embedding_dropout = gluon.nn.Dropout(embedding_dropout)

            self.lstm = gluon.rnn.HybridSequentialRNNCell()
            self.lstm.add(
                gluon.rnn.LSTMCell(hidden_num),
            )
            self.dropout = gluon.nn.Dropout(dropout)
            self.nn = gluon.nn.HybridSequential()
            self.nn.add(
                gluon.nn.Dense(ku_num, flatten=False)
            )

    def hybrid_forward(self, F, responses, mask=None, begin_state=None, *args, **kwargs):
        length = self.length if self.length else len(responses[0])

        if self.nettype == "EmbedDKT":
            input_data = self.embedding_dropout(self.embedding(responses))
        else:
            input_data = responses

        outputs, states = self.lstm.unroll(length, input_data, begin_state=begin_state, merge_outputs=True,
                                           valid_length=mask)

        output = self.nn(self.dropout(outputs))
        output = F.sigmoid(output)
        return output, states
