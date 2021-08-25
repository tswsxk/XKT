# coding: utf-8
# 2021/8/22 @ tongshiwei


__all__ = ["get_net"]

import json
import networkx as nx
from baize.mxnet.utils import format_sequence, mask_sequence_variable_length
from mxnet import gluon
import mxnet as mx
from XKT.utils.nn import GRUCell, begin_states, get_states, expand_tensor


def get_net(ku_num, graph, **kwargs):
    return GKT(ku_num, graph, **kwargs)


class GKT(gluon.Block):
    def __init__(self, ku_num, graph, latent_dim=None,
                 hidden_num=None, dropout=0.0, prefix=None, params=None):
        super(GKT, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.ku_num if latent_dim is None else int(latent_dim)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(ku_num))
        try:
            with open(graph) as f:
                self.graph.add_weighted_edges_from(json.load(f))
        except ValueError:
            with open(graph) as f:
                self.graph.add_weighted_edges_from([e + [1.0] for e in json.load(f)])

        with self.name_scope():
            self.rnn = GRUCell(self.hidden_num)
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.concept_embedding = gluon.nn.Embedding(self.ku_num, self.latent_dim)
            self.f_self = gluon.nn.Dense(self.hidden_num, flatten=False)
            self.n_out = gluon.nn.Dense(self.hidden_num, flatten=False)
            self.n_in = gluon.nn.Dense(self.hidden_num, flatten=False)
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(1, flatten=False)

    def in_weight(self, x, ordinal=True, with_weight=True):
        if isinstance(x, mx.nd.NDArray):
            x = x.asnumpy().tolist()
        if isinstance(x, list):
            return [self.in_weight(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                return list(self.graph.predecessors(int(x)))
            else:
                _ret = [0] * self.ku_num
                for i in self.graph.predecessors(int(x)):
                    if with_weight:
                        _ret[i] = self.graph[i][x]['weight']
                    else:
                        _ret[i] = 1
                return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def out_weight(self, x, ordinal=True, with_weight=True):
        if isinstance(x, mx.nd.NDArray):
            x = x.asnumpy().tolist()
        if isinstance(x, list):
            return [self.out_weight(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                return list(self.graph.successors(int(x)))
            else:
                _ret = [0] * self.ku_num
                for i in self.graph.successors(int(x)):
                    if with_weight:
                        _ret[i] = self.graph[x][i]['weight']
                    else:
                        _ret[i] = 1
                return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def neighbors(self, x, ordinal=True, with_weight=False):
        if isinstance(x, mx.nd.NDArray):
            x = x.asnumpy().tolist()
        if isinstance(x, list):
            return [self.neighbors(_x) for _x in x]
        elif isinstance(x, (int, float)):
            if not ordinal:
                return list(self.graph.neighbors(int(x)))
            else:
                _ret = [0] * self.ku_num
                for i in self.graph.neighbors(int(x)):
                    if with_weight:
                        _ret[i] = self.graph[x][i]['weight']
                    else:
                        _ret[i] = 1
                return _ret
        else:
            raise TypeError("cannot handle %s" % type(x))

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True, *args,
                **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        outputs = []
        all_states = []
        for i in range(length):
            # neighbors - aggregate
            _neighbors = self.neighbors(inputs[i])
            neighbors_mask = expand_tensor(mx.nd.array(_neighbors, ctx=ctx), -1, self.hidden_num)
            _neighbors_mask = expand_tensor(mx.nd.array(_neighbors, ctx=ctx), -1, self.hidden_num + self.latent_dim)

            # get concept embedding
            concept_embeddings = self.concept_embedding.weight.data(ctx)
            concept_embeddings = expand_tensor(concept_embeddings, 0, batch_size)

            agg_states = mx.nd.concat(concept_embeddings, states, dim=-1)

            # aggregate
            _neighbors_states = _neighbors_mask * agg_states

            # self - aggregate
            _concept_embedding = get_states(inputs[i], states)
            _self_hidden_states = mx.nd.concat(_concept_embedding, self.response_embedding(answers[i]), dim=-1)

            _self_mask = mx.nd.one_hot(inputs[i], self.ku_num)
            _self_mask = expand_tensor(_self_mask, -1, self.hidden_num)

            self_hidden_states = expand_tensor(_self_hidden_states, 1, self.ku_num)

            # aggregate
            _hidden_states = mx.nd.concat(_neighbors_states, self_hidden_states, dim=-1)

            _in_state = self.n_in(_hidden_states)
            _out_state = self.n_out(_hidden_states)
            in_weight = expand_tensor(mx.nd.array(self.in_weight(inputs[i]), ctx=ctx), -1, self.hidden_num)
            out_weight = expand_tensor(mx.nd.array(self.out_weight(inputs[i]), ctx=ctx), -1, self.hidden_num)

            next_neighbors_states = in_weight * _in_state + out_weight * _out_state

            # self - update
            next_self_states = self.f_self(_self_hidden_states)
            next_self_states = expand_tensor(next_self_states, 1, self.ku_num)
            next_self_states = _self_mask * next_self_states

            next_states = neighbors_mask * next_neighbors_states + next_self_states

            next_states, _ = self.rnn(next_states, [states])
            next_states = (_self_mask + neighbors_mask) * next_states + (1 - _self_mask - neighbors_mask) * states

            states = self.dropout(next_states)
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(states), axis=-1))
            outputs.append(output)
            if valid_length is not None and not compressed_out:
                all_states.append([states])

        if valid_length is not None:
            if compressed_out:
                states = None
            else:
                states = [mx.nd.SequenceLast(mx.nd.stack(*ele_list, axis=0),
                                             sequence_length=valid_length,
                                             use_sequence_length=True,
                                             axis=0)
                          for ele_list in zip(*all_states)]
            outputs = mask_sequence_variable_length(mx.nd, outputs, length, valid_length, axis, True)
        outputs, _, _, _ = format_sequence(length, outputs, layout, merge=True)

        return outputs, states
