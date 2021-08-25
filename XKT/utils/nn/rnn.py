# coding: utf-8
# 2021/8/23 @ tongshiwei
import mxnet as mx
from baize import as_list
from mxnet import gluon

__all__ = ["GRUCell", "begin_states", "get_states", "expand_tensor", "expand_states"]


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


def expand_states(indexes, states, expand_num):
    if isinstance(indexes, mx.nd.NDArray):
        indexes = indexes.asnumpy().tolist()
    if isinstance(indexes, list):
        return mx.nd.stack(*[expand_states(index, state, expand_num) for (index, state) in zip(indexes, states)])
    elif isinstance(indexes, (int, float)):
        _expand_state = mx.nd.broadcast_to(mx.nd.expand_dims(states, 0), (expand_num, 0))
        _mask = mx.nd.array([[0] * len(states) for _ in range(expand_num)], ctx=states.context)
        return _expand_state * _mask
    else:
        raise TypeError("cannot handle %s" % type(indexes))


def expand_tensor(tensor, expand_axis, expand_num, ctx=None, dtype=None) -> mx.nd.NDArray:
    if not isinstance(tensor, mx.nd.NDArray):
        tensor = mx.nd.array(tensor, ctx, dtype)
    assert len(tensor.shape) == 2

    _tensor = mx.nd.expand_dims(tensor, expand_axis)

    shape = [0] * 3
    shape[expand_axis] = expand_num

    _tensor = mx.nd.broadcast_to(_tensor, tuple(shape))

    return _tensor


class GRUCell(gluon.nn.Block):
    def __init__(self, hidden_num, prefix=None, params=None):
        super(GRUCell, self).__init__(prefix, params)
        with self.name_scope():
            self.i2h = gluon.nn.Dense(3 * hidden_num, flatten=False)
            self.h2h = gluon.nn.Dense(3 * hidden_num, flatten=False)
            self.reset_act = gluon.nn.Activation("sigmoid")
            self.update_act = gluon.nn.Activation("sigmoid")
            self.act = gluon.nn.Activation("tanh")

    def forward(self, inputs, states):
        prev_state_h = states[0]

        i2h = self.i2h(inputs)
        h2h = self.h2h(prev_state_h)
        i2h_r, i2h_z, i2h = mx.nd.SliceChannel(i2h, 3, axis=-1)
        h2h_r, h2h_z, h2h = mx.nd.SliceChannel(h2h, 3, axis=-1)

        reset_gate = self.reset_act(i2h_r + h2h_r)
        update_gate = self.update_act(i2h_z + h2h_z)
        next_h_tmp = self.act(i2h + reset_gate * h2h)
        ones = mx.nd.ones_like(update_gate)
        next_h = (ones - update_gate) * next_h_tmp + update_gate * prev_state_h

        return next_h, [next_h]
