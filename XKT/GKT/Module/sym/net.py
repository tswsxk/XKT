# coding: utf-8
# create by tongshiwei on 2019-7-30

__all__ = ["get_net", "get_bp_loss"]

import networkx as nx
from longling import as_list
from longling.ML.MxnetHelper.gallery.layer import format_sequence, mask_sequence_variable_length
from longling.ML.MxnetHelper.gallery.layer.attention import MultiHeadAttentionCell, DotProductAttentionCell
from mxnet import gluon
import mxnet as mx

from XKT.shared import SLMLoss


def get_bp_loss(**kwargs):
    return {"SLMLoss": SLMLoss(**kwargs)}


def begin_states(shapes, prefix, func=mx.nd.zeros):
    states = []
    for i, shape in enumerate(as_list(shapes)):
        state = func(name='%sbegin_state_%d' % (prefix, i), shape=shape)
        states.append(state)
    return states


def get_states(indexes, states):
    if isinstance(indexes, mx.nd.NDArray):
        indexes = indexes.asnumpy().tolist()
    if isinstance(indexes, list):
        return mx.nd.stack(*[get_states(index, state) for (index, state) in zip(indexes, states)])
    elif isinstance(indexes, (int, float)):
        return states[int(indexes)]
    else:
        raise TypeError("cannot handle %s" % type(indexes))


def expand_states(indexes, states, expand_axis):
    mask = mx.nd.expand_dims(indexes, )



def get_net(ku_num, graph=None, gl_type="MHA", **kwargs):
    return GKT(ku_num, graph, gl_type, **kwargs)


class GKT(gluon.Block):
    def __init__(self, ku_num, graph, gl_type="MHA", latent_dim=None,
                 hidden_num=None, prefix=None, params=None, K=2, **kwargs):
        super(GKT, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.ku_num if latent_dim is None else int(latent_dim)
        self.graph = nx.DiGraph() if not isinstance(graph, nx.DiGraph) else graph
        self.graph.add_nodes_from(range(ku_num))
        # self.graph.add_edges_from([(i, i + 1) for i in range((ku_num - 1))])
        self.gl_type = gl_type

        with self.name_scope():
            self.rnn = gluon.rnn.GRUCell(self.hidden_num)
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.concept_embedding = gluon.nn.Embedding(self.ku_num, self.latent_dim)
            self.f_self = gluon.nn.Dense(self.hidden_num, flatten=False)
            self.out = gluon.nn.Dense(1, flatten=False)

            if self.gl_type == "PAM":
                raise NotImplementedError
            elif self.gl_type == "MHA":
                self.K = K
                units = self.hidden_num * K
                self._multi_attention = MultiHeadAttentionCell(
                    DotProductAttentionCell(),
                    units, units, units,
                    self.K,
                )
                self.gl = self._multi_attention
            elif self.gl_type == "VAE":
                raise NotImplementedError
            else:
                raise TypeError("gl_type must be PAM, MHA or VAE, now is %s" % self.gl_type)

    def neighbors(self, x, ordinal=True):
        if isinstance(x, mx.nd.NDArray):
            x = x.asnumpy().tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                return list(self.graph.neighbors(int(x)))
            else:
                _ret = [0] * self.hidden_num
                for i in self.graph.neighbors(int(x)):
                    _ret[i] = 1
                return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def forward(self, questions, answers, mask=None, states=None, layout='NTC', *args, **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        from tqdm import tqdm
        for i in tqdm(range(length)):
            # neighbors - aggregate
            _neighbors = self.neighbors(inputs[i])
            _neighbors_mask = mx.nd.expand_dims(mx.nd.array(_neighbors), -1)
            _neighbors_ea_mask = 1 - mx.nd.broadcast_to(_neighbors_mask, (0, 0, self.hidden_num))
            _neighbors_mask = mx.nd.broadcast_to(_neighbors_mask, (0, 0, self.hidden_num + self.latent_dim))

            # get concept embedding
            concept_embeddings = self.concept_embedding.weight.data(ctx)
            concept_embeddings = mx.nd.expand_dims(mx.nd.array(concept_embeddings), 0)
            concept_embeddings = mx.nd.broadcast_to(concept_embeddings, (batch_size, 0, 0))

            agg_states = mx.nd.concat(concept_embeddings, states, dim=-1)

            # aggregate
            _neighbors_states = _neighbors_mask * agg_states

            # self - aggregate
            _concept_embedding = get_states(inputs[i], states)
            _self_hidden_states = mx.nd.concat(_concept_embedding, self.response_embedding(answers[i]), dim=-1)

            _self_mask = mx.nd.one_hot(inputs[i], self.ku_num)
            _self_mask = mx.nd.expand_dims(_self_mask, -1)
            _self_mask = mx.nd.broadcast_to(_self_mask, (0, 0, self.hidden_num + self.latent_dim))

            self_hidden_states = mx.nd.expand_dims(_self_hidden_states, 1)
            self_hidden_states = mx.nd.broadcast_to(self_hidden_states, (0, self.ku_num, 0))

            self_hidden_states = _self_mask * self_hidden_states

            # aggregate
            _hidden_states = _neighbors_states + self_hidden_states

            # neighbors - update
            context_vec, _ = self.gl(_hidden_states, _hidden_states)
            next_neighbors_states = mx.nd.mean(mx.nd.reshape(context_vec, (0, 0, self.ku_num, -1)),
                                               -1) + _neighbors_ea_mask * states

            # self - update
            next_self_states = self.f_self(_self_hidden_states)
            next_self_states = mx.nd.expand_dims(next_self_states, 1)
            next_self_states = mx.nd.broadcast_to(next_self_states, (0, self.ku_num, 0))
            next_self_states = _self_mask * next_self_states

            next_states = next_neighbors_states + next_self_states

            _next_states, _, _, _ = format_sequence(self.ku_num, next_states, layout, False)
            _states, _, _, _ = format_sequence(self.ku_num, states, layout, False)
            next_states = mx.nd.stack(*[self.rnn(_next_states[j], [_states[j]])[1][0] for j in range(self.ku_num)],
                                      axis=1)
            states = next_states
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(states)))
        return output, states
