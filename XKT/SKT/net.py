# coding: utf-8
# 2021/8/22 @ tongshiwei
from baize.mxnet.utils import format_sequence, mask_sequence_variable_length
from mxnet import gluon
import mxnet as mx
from XKT.utils.nn import GRUCell, begin_states, get_states, expand_tensor
from .utils import Graph


def get_net(ku_num, graph_params=None, net_type="SKT", k=2, **kwargs):
    if net_type == "SKT":
        return SKT(ku_num, graph_params, **kwargs)
    elif net_type == "SKT_TE":
        return SKT_TE(ku_num, **kwargs)
    elif net_type == "SKTPart":
        return SKTPart(ku_num, graph_params, **kwargs)
    elif net_type == "SKTSync":
        return SKTSync(ku_num, graph_params, **kwargs)
    else:
        raise NotImplementedError


class SKT(gluon.Block):
    def __init__(self, ku_num, graph_params=None,
                 alpha=0.5,
                 latent_dim=None, activation=None,
                 hidden_num=90, concept_dim=None,
                 # dropout=0.5, self_dropout=0.0,
                 dropout=0.0, self_dropout=0.5,
                 # dropout=0.0, self_dropout=0.0,
                 sync_activation="relu", sync_dropout=0.0,
                 prop_activation="relu", prop_dropout=0.0,
                 agg_activation="relu", agg_dropout=0.0,
                 prefix=None, params=None):
        super(SKT, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.hidden_num if latent_dim is None else int(latent_dim)
        self.concept_dim = self.hidden_num if concept_dim is None else int(concept_dim)
        graph_params = graph_params if graph_params is not None else []
        self.graph = Graph.from_file(ku_num, graph_params)
        self.alpha = alpha

        sync_activation = sync_activation if activation is None else activation
        prop_activation = prop_activation if activation is None else activation
        agg_activation = agg_activation if activation is None else activation

        with self.name_scope():
            self.rnn = GRUCell(self.hidden_num)
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.concept_embedding = gluon.nn.Embedding(self.ku_num, self.concept_dim)
            self.f_self = gluon.rnn.GRUCell(self.hidden_num)
            # self.f_self = gluon.nn.Sequential()
            # self.f_self.add(
            #     gluon.nn.Dense(self.hidden_num),
            #     gluon.nn.Activation("relu")
            # )
            self.self_dropout = gluon.nn.Dropout(self_dropout)
            self.f_prop = gluon.nn.Sequential()
            self.f_prop.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                gluon.nn.Activation(prop_activation),
                gluon.nn.Dropout(prop_dropout),
            )
            self.f_sync = gluon.nn.Sequential()
            self.f_sync.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                gluon.nn.Activation(sync_activation),
                gluon.nn.Dropout(sync_dropout),
            )
            self.f_agg = gluon.nn.Sequential()
            self.f_agg.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                # gluon.nn.InstanceNorm(),
                # gluon.nn.LayerNorm(),
                # gluon.nn.BatchNorm(),
                gluon.nn.Activation(agg_activation),
                gluon.nn.Dropout(agg_dropout),
            )
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(1, flatten=False)

    def neighbors(self, x, ordinal=True):
        return self.graph.neighbors(x, ordinal)

    def successors(self, x, ordinal=True):
        return self.graph.successors(x, ordinal)

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True,
                *args, **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        if states is None:
            states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        outputs = []
        all_states = []
        for i in range(length):
            # self - influence
            _self_state = get_states(inputs[i], states)
            # fc
            # _next_self_state = self.f_self(mx.nd.concat(_self_state, self.response_embedding(answers[i]), dim=-1))
            # gru
            _next_self_state, _ = self.f_self(self.response_embedding(answers[i]), [_self_state])
            # _next_self_state = self.f_self(mx.nd.concat(_self_hidden_states, _self_state))
            # _next_self_state, _ = self.f_self(_self_hidden_states, [_self_state])
            _next_self_state = self.self_dropout(_next_self_state)

            # get self mask
            _self_mask = mx.nd.expand_dims(mx.nd.one_hot(inputs[i], self.ku_num), -1)
            _self_mask = mx.nd.broadcast_to(_self_mask, (0, 0, self.hidden_num))

            # find neighbors
            _neighbors = self.neighbors(inputs[i])
            _neighbors_mask = mx.nd.expand_dims(mx.nd.array(_neighbors, ctx=ctx), -1)
            _neighbors_mask = mx.nd.broadcast_to(_neighbors_mask, (0, 0, self.hidden_num))

            # synchronization
            _broadcast_next_self_states = mx.nd.expand_dims(_next_self_state, 1)
            _broadcast_next_self_states = mx.nd.broadcast_to(_broadcast_next_self_states, (0, self.ku_num, 0))
            # _sync_diff = mx.nd.concat(states, _broadcast_next_self_states, concept_embeddings, dim=-1)
            _sync_diff = mx.nd.concat(states, _broadcast_next_self_states, dim=-1)
            _sync_inf = _neighbors_mask * self.f_sync(_sync_diff)

            # reflection on current vertex
            _reflec_inf = mx.nd.sum(_sync_inf, axis=1)
            _reflec_inf = mx.nd.broadcast_to(mx.nd.expand_dims(_reflec_inf, 1), (0, self.ku_num, 0))
            _sync_inf = _sync_inf + _self_mask * _reflec_inf

            # find successors
            _successors = self.successors(inputs[i])
            _successors_mask = mx.nd.expand_dims(mx.nd.array(_successors, ctx=ctx), -1)
            _successors_mask = mx.nd.broadcast_to(_successors_mask, (0, 0, self.hidden_num))

            # propagation
            # _prop_diff = mx.nd.concat(_next_self_state - _self_state, self.concept_embedding(inputs[i]), dim=-1)
            _prop_diff = _next_self_state - _self_state

            # 1
            _prop_inf = self.f_prop(_prop_diff)
            _prop_inf = _successors_mask * mx.nd.broadcast_to(mx.nd.expand_dims(_prop_inf, axis=1), (0, self.ku_num, 0))
            # 2
            # _broadcast_diff = mx.nd.broadcast_to(mx.nd.expand_dims(_prop_diff, axis=1), (0, self.ku_num, 0))
            # _pro_inf = _successors_mask * self.f_prop(
            #     mx.nd.concat(_broadcast_diff, concept_embeddings, dim=-1)
            # )
            # _pro_inf = _successors_mask * self.f_prop(
            #     _broadcast_diff
            # )
            # concept embedding
            concept_embeddings = self.concept_embedding.weight.data(ctx)
            concept_embeddings = expand_tensor(concept_embeddings, 0, batch_size)
            # concept_embeddings = (_self_mask + _successors_mask + _neighbors_mask) * concept_embeddings

            # aggregate
            _inf = self.f_agg(self.alpha * _sync_inf + (1 - self.alpha) * _prop_inf)
            # next_states, _ = self.rnn(_inf, [states])
            next_states, _ = self.rnn(mx.nd.concat(_inf, concept_embeddings, dim=-1), [states])
            # states = (1 - _self_mask) * next_states + _self_mask * _broadcast_next_self_states
            states = next_states
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(self.dropout(states)), axis=-1))
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


class SKTPart(SKT):
    def __init__(self, ku_num, graph_params=None,
                 latent_dim=None, activation=None,
                 hidden_num=90, concept_dim=None,
                 dropout=0.0, self_dropout=0.0,
                 prop_activation="relu", prop_dropout=0.0,
                 prefix=None, params=None):
        super(SKT, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.hidden_num if latent_dim is None else int(latent_dim)
        self.concept_dim = self.hidden_num if concept_dim is None else int(concept_dim)
        graph_params = graph_params if graph_params is not None else []
        self.graph = Graph.from_file(ku_num, graph_params)

        prop_activation = prop_activation if activation is None else activation

        with self.name_scope():
            self.rnn = GRUCell(self.hidden_num)
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.concept_embedding = gluon.nn.Embedding(self.ku_num, self.concept_dim)
            self.f_self = gluon.rnn.GRUCell(self.hidden_num)
            # self.f_self = gluon.nn.Sequential()
            # self.f_self.add(
            #     gluon.nn.Dense(self.hidden_num),
            #     gluon.nn.Activation("relu")
            # )
            self.self_dropout = gluon.nn.Dropout(self_dropout)
            self.f_prop = gluon.nn.Sequential()
            self.f_prop.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                gluon.nn.Activation(prop_activation),
                gluon.nn.Dropout(prop_dropout),
            )
            # self.f_sync = gluon.nn.Sequential()
            # self.f_sync.add(
            #     gluon.nn.Dense(self.hidden_num, flatten=False),
            #     gluon.nn.Activation(sync_activation),
            #     gluon.nn.Dropout(sync_dropout),
            # )
            # self.f_reflec = gluon.nn.Sequential()
            # self.f_reflec.add(
            #     gluon.nn.Dense(self.hidden_num, flatten=False),
            #     gluon.nn.Activation(sync_activation),
            #     gluon.nn.Dropout(sync_dropout),
            # )
            # self.f_agg = gluon.nn.Sequential()
            # self.f_agg.add(
            #     gluon.nn.Dense(self.hidden_num, flatten=False),
            #     # gluon.nn.InstanceNorm(),
            #     # gluon.nn.LayerNorm(),
            #     # gluon.nn.BatchNorm(),
            #     gluon.nn.Activation(agg_activation),
            #     gluon.nn.Dropout(agg_dropout),
            # )
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(1, flatten=False)

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True,
                *args, **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        if states is None:
            states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        outputs = []
        all_states = []
        for i in range(length):
            # self - influence
            _self_state = get_states(inputs[i], states)
            # fc
            # _next_self_state = self.f_self(mx.nd.concat(_self_state, self.response_embedding(answers[i]), dim=-1))
            # gru
            _next_self_state, _ = self.f_self(self.response_embedding(answers[i]), [_self_state])
            # _next_self_state = self.f_self(mx.nd.concat(_self_hidden_states, _self_state))
            # _next_self_state, _ = self.f_self(_self_hidden_states, [_self_state])
            _next_self_state = self.self_dropout(_next_self_state)

            # get self mask
            _self_mask = mx.nd.expand_dims(mx.nd.one_hot(inputs[i], self.ku_num), -1)
            _self_mask = mx.nd.broadcast_to(_self_mask, (0, 0, self.hidden_num))
            # self-concept embedding
            # _self_concept_embedding = self.concept_embedding(inputs[i])
            # _broadcast_self_concept_embedding = mx.nd.expand_dims(_self_concept_embedding, dim=1)
            # _broadcast_self_concept_embedding = mx.nd.broadcast_to(_broadcast_self_concept_embedding,
            #                                                        (0, self.ku_num, 0))
            # concept embedding
            concept_embeddings = self.concept_embedding.weight.data(ctx)
            concept_embeddings = expand_tensor(concept_embeddings, 0, batch_size)
            # concept_embeddings = (_self_mask + _successors_mask + _neighbors_mask) * concept_embeddings

            # find successors
            _successors = self.successors(inputs[i])
            _successors_mask = mx.nd.expand_dims(mx.nd.array(_successors, ctx=ctx), -1)
            _successors_mask = mx.nd.broadcast_to(_successors_mask, (0, 0, self.hidden_num))

            _broadcast_next_self_states = mx.nd.expand_dims(_next_self_state, 1)
            _broadcast_next_self_states = mx.nd.broadcast_to(_broadcast_next_self_states, (0, self.ku_num, 0))

            # propagation
            # _prop_diff = mx.nd.concat(_next_self_state - _self_state, self.concept_embedding(inputs[i]), dim=-1)
            _prop_diff = _next_self_state - _self_state

            # 1
            _prop_inf = self.f_prop(
                mx.nd.concat(mx.nd.broadcast_to(mx.nd.expand_dims(_prop_diff, axis=1), (0, self.ku_num, 0)),
                             concept_embeddings, dim=-1))
            _prop_inf = _successors_mask * _prop_inf

            # aggregate
            # _inf = self.f_agg(_prop_inf)
            _inf = _prop_inf
            # next_states, _ = self.rnn(_inf, [states])
            next_states, _ = self.rnn(_inf, [states])
            updated = _successors_mask * next_states + _self_mask * _broadcast_next_self_states
            states = updated + (1 - _successors_mask - _self_mask) * states
            # states = next_states
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(self.dropout(states)), axis=-1))
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


class SKT_TE(gluon.Block):
    def __init__(self, ku_num,
                 latent_dim=None,
                 hidden_num=90, concept_dim=None,
                 dropout=0.0, self_dropout=0.5,
                 prefix=None, params=None):
        super(SKT_TE, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.hidden_num if latent_dim is None else int(latent_dim)
        self.concept_dim = self.hidden_num if concept_dim is None else int(concept_dim)

        with self.name_scope():
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.f_self = gluon.rnn.GRUCell(self.hidden_num)
            self.self_dropout = gluon.nn.Dropout(self_dropout)
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(1, flatten=False)

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True,
                *args, **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        if states is None:
            states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        outputs = []
        all_states = []
        for i in range(length):
            # self - influence
            _self_state = get_states(inputs[i], states)
            # fc
            # _next_self_state = self.f_self(mx.nd.concat(_self_state, self.response_embedding(answers[i]), dim=-1))
            # gru
            _next_self_state, _ = self.f_self(self.response_embedding(answers[i]), [_self_state])
            # _next_self_state = self.f_self(mx.nd.concat(_self_hidden_states, _self_state))
            # _next_self_state, _ = self.f_self(_self_hidden_states, [_self_state])
            _next_self_state = self.self_dropout(_next_self_state)

            # get self mask
            _self_mask = mx.nd.expand_dims(mx.nd.one_hot(inputs[i], self.ku_num), -1)
            _self_mask = mx.nd.broadcast_to(_self_mask, (0, 0, self.hidden_num))

            _broadcast_next_self_states = mx.nd.expand_dims(_next_self_state, 1)
            _broadcast_next_self_states = mx.nd.broadcast_to(_broadcast_next_self_states, (0, self.ku_num, 0))

            states = (1 - _self_mask) * states + _self_mask * _broadcast_next_self_states
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(self.dropout(states)), axis=-1))
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


class SKTSync(SKT):
    def __init__(self, ku_num, graph_params=None,
                 alpha=0.5,
                 latent_dim=None, activation=None,
                 hidden_num=90, concept_dim=None,
                 dropout=0.0, self_dropout=0.0,
                 sync_activation="relu", sync_dropout=0.0,
                 prop_activation="relu", prop_dropout=0.0,
                 agg_activation="relu", agg_dropout=0.0,
                 prefix=None, params=None):
        super(SKT, self).__init__(prefix=prefix, params=params)
        self.ku_num = int(ku_num)
        self.hidden_num = self.ku_num if hidden_num is None else int(hidden_num)
        self.latent_dim = self.hidden_num if latent_dim is None else int(latent_dim)
        self.concept_dim = self.hidden_num if concept_dim is None else int(concept_dim)
        graph_params = graph_params if graph_params is not None else []
        self.graph = Graph.from_file(ku_num, graph_params)
        self.alpha = alpha

        sync_activation = sync_activation if activation is None else activation

        with self.name_scope():
            self.rnn = GRUCell(self.hidden_num)
            self.response_embedding = gluon.nn.Embedding(2 * self.ku_num, self.latent_dim)
            self.concept_embedding = gluon.nn.Embedding(self.ku_num, self.concept_dim)
            self.f_self = gluon.rnn.GRUCell(self.hidden_num)
            # self.f_self = gluon.nn.Sequential()
            # self.f_self.add(
            #     gluon.nn.Dense(self.hidden_num),
            #     gluon.nn.Activation("relu")
            # )
            self.self_dropout = gluon.nn.Dropout(self_dropout)
            self.f_sync = gluon.nn.Sequential()
            self.f_sync.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                gluon.nn.Activation(sync_activation),
                gluon.nn.Dropout(sync_dropout),
            )
            self.f_reflec = gluon.nn.Sequential()
            self.f_reflec.add(
                gluon.nn.Dense(self.hidden_num, flatten=False),
                gluon.nn.Activation(sync_activation),
                gluon.nn.Dropout(sync_dropout),
            )
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(1, flatten=False)

    def forward(self, questions, answers, valid_length=None, states=None, layout='NTC', compressed_out=True,
                *args, **kwargs):
        ctx = questions.context
        length = questions.shape[1]

        inputs, axis, F, batch_size = format_sequence(length, questions, layout, False)
        answers, _, _, _ = format_sequence(length, answers, layout, False)
        if states is None:
            states = begin_states([(batch_size, self.ku_num, self.hidden_num)], self.prefix)[0]
        states = states.as_in_context(ctx)
        outputs = []
        all_states = []
        for i in range(length):
            # self - influence
            _self_state = get_states(inputs[i], states)
            # fc
            # _next_self_state = self.f_self(mx.nd.concat(_self_state, self.response_embedding(answers[i]), dim=-1))
            # gru
            _next_self_state, _ = self.f_self(self.response_embedding(answers[i]), [_self_state])
            # _next_self_state = self.f_self(mx.nd.concat(_self_hidden_states, _self_state))
            # _next_self_state, _ = self.f_self(_self_hidden_states, [_self_state])
            _next_self_state = self.self_dropout(_next_self_state)

            # get self mask
            _self_mask = mx.nd.expand_dims(mx.nd.one_hot(inputs[i], self.ku_num), -1)
            _self_mask = mx.nd.broadcast_to(_self_mask, (0, 0, self.hidden_num))
            # self-concept embedding
            _self_concept_embedding = self.concept_embedding(inputs[i])
            # _broadcast_self_concept_embedding = mx.nd.expand_dims(_self_concept_embedding, dim=1)
            # _broadcast_self_concept_embedding = mx.nd.broadcast_to(_broadcast_self_concept_embedding,
            #                                                        (0, self.ku_num, 0))
            # concept embedding
            concept_embeddings = self.concept_embedding.weight.data(ctx)
            concept_embeddings = expand_tensor(concept_embeddings, 0, batch_size)
            # concept_embeddings = (_self_mask + _successors_mask + _neighbors_mask) * concept_embeddings

            # find neighbors
            _neighbors = self.neighbors(inputs[i])
            _neighbors_mask = mx.nd.expand_dims(mx.nd.array(_neighbors, ctx=ctx), -1)
            _neighbors_mask = mx.nd.broadcast_to(_neighbors_mask, (0, 0, self.hidden_num))

            # synchronization
            _broadcast_next_self_states = mx.nd.expand_dims(_next_self_state, 1)
            _broadcast_next_self_states = mx.nd.broadcast_to(_broadcast_next_self_states, (0, self.ku_num, 0))
            # _sync_diff = mx.nd.concat(states, _broadcast_next_self_states, concept_embeddings, dim=-1)
            _sync_diff = mx.nd.concat(states, _broadcast_next_self_states, dim=-1)
            _sync_inf = _neighbors_mask * self.f_sync(
                mx.nd.concat(_sync_diff, concept_embeddings, dim=-1)
            )

            # reflection on current vertex
            _reflec_diff = mx.nd.concat(mx.nd.sum(_neighbors_mask * states, axis=1) + _next_self_state,
                                        _self_concept_embedding, dim=-1)
            # _reflec_diff = mx.nd.concat(mx.nd.sum(_neighbors_mask * states, axis=1), _next_self_state,
            #                             _self_concept_embedding, dim=-1)
            _reflec_inf = self.f_reflec(_reflec_diff)
            _reflec_inf = mx.nd.broadcast_to(mx.nd.expand_dims(_reflec_inf, 1), (0, self.ku_num, 0))
            _sync_inf = _sync_inf + _self_mask * _reflec_inf

            # aggregate
            _inf = _sync_inf
            next_states, _ = self.rnn(_inf, [states])
            states = (_neighbors_mask + _self_mask) * next_states + (1 - _neighbors_mask - _self_mask) * states
            # states = next_states
            output = mx.nd.sigmoid(mx.nd.squeeze(self.out(self.dropout(states)), axis=-1))
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
