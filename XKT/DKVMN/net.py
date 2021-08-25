# coding: utf-8
# 2021/8/22 @ tongshiwei

from baize.mxnet.utils import format_sequence, mask_sequence_variable_length
from mxnet import gluon
from mxnet import ndarray


def get_net(ku_num, key_embedding_dim, value_embedding_dim, hidden_num,
            key_memory_size,
            nettype="DKVMN", dropout=0.0, **kwargs):
    return DKVMN(
        ku_num=ku_num,
        key_embedding_dim=key_embedding_dim,
        value_embedding_dim=value_embedding_dim,
        hidden_num=hidden_num,
        key_memory_size=key_memory_size,
        nettype=nettype,
        dropout=dropout,
        **kwargs
    )


class KVMNCell(gluon.HybridBlock):
    def __init__(self, memory_state_dim, memory_size, input_size=0, prefix=None, params=None, *args, **kwargs):
        super(KVMNCell, self).__init__(prefix=prefix, params=params)

        self._input_size = input_size
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

    def addressing(self, F, control_input, memory):
        """

        Parameters
        ----------
        F
        control_input: Shape (batch_size, control_state_dim)
        memory: Shape (memory_size, memory_state_dim)

        Returns
        -------
        correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = F.FullyConnected(data=control_input,
                                            num_hidden=self.memory_size,
                                            weight=memory,
                                            no_bias=True,
                                            name="similarity_score")
        correlation_weight = F.SoftmaxActivation(similarity_score)  # Shape: (batch_size, memory_size)
        return correlation_weight

    def reset(self):
        pass

    def hybrid_forward(self, F, control_input, memory, *args, **kwargs):
        return self.addressing(F, control_input, memory)


class KVMNReadCell(KVMNCell):
    def __init__(self, memory_state_dim, memory_size, input_size=0, prefix=None, params=None):
        super(KVMNReadCell, self).__init__(memory_state_dim, memory_size, input_size, prefix, params)

    def read(self, memory, control_input=None, read_weight=None):
        return self(memory, control_input, read_weight)

    def hybrid_forward(self, F, memory, control_input=None, read_weight=None):
        """

        Parameters
        ----------
        F
        control_input: Shape (batch_size, control_state_dim)
        memory: Shape (batch_size, memory_size, memory_state_dim)
        read_weight: Shape (batch_size, memory_size)

        Returns
        -------
        read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(F, control_input=control_input, memory=memory)
        read_weight = F.Reshape(read_weight, shape=(-1, 1, self.memory_size))
        read_content = F.Reshape(data=F.batch_dot(read_weight, memory),
                                 # Shape (batch_size, 1, memory_state_dim)
                                 shape=(-1, self.memory_state_dim))  # Shape (batch_size, memory_state_dim)
        return read_content


class KVMNWriteCell(KVMNCell):
    def __init__(self, memory_state_dim, memory_size, input_size=0,
                 erase_signal_weight_initializer=None, erase_signal_bias_initializer=None,
                 add_signal_weight_initializer=None, add_signal_bias_initializer=None,
                 prefix=None, params=None):
        super(KVMNWriteCell, self).__init__(memory_state_dim, memory_size, input_size, prefix, params)
        with self.name_scope():
            self.erase_signal_weight = self.params.get('erase_signal_weight', shape=(memory_state_dim, input_size),
                                                       init=erase_signal_weight_initializer,
                                                       allow_deferred_init=True)

            self.erase_signal_bias = self.params.get('erase_signal_bias', shape=(memory_state_dim,),
                                                     init=erase_signal_bias_initializer,
                                                     allow_deferred_init=True)

            self.add_signal_weight = self.params.get('add_signal_weight', shape=(memory_state_dim, input_size),
                                                     init=add_signal_weight_initializer,
                                                     allow_deferred_init=True)

            self.add_signal_bias = self.params.get('add_signal_bias', shape=(memory_state_dim,),
                                                   init=add_signal_bias_initializer,
                                                   allow_deferred_init=True)

    def read(self, F, memory, control_input=None, read_weight=None):
        if read_weight is None:
            read_weight = self.addressing(F, control_input=control_input, memory=memory)
        read_weight = F.Reshape(read_weight, shape=(-1, 1, self.memory_size))
        read_content = F.Reshape(data=F.batch_dot(read_weight, memory, name=self.name + "read_content_batch_dot"),
                                 # Shape (batch_size, 1, memory_state_dim)
                                 shape=(-1, self.memory_state_dim))  # Shape (batch_size, memory_state_dim)
        return read_content

    def write(self, memory, control_input, write_weight):
        return self(memory, control_input, write_weight)

    def hybrid_forward(self, F, memory, control_input, write_weight,
                       erase_signal_weight, erase_signal_bias, add_signal_weight, add_signal_bias,
                       ):
        if write_weight is None:
            write_weight = self.addressing(
                F, control_input=control_input, memory=memory
            )  # Shape Shape (batch_size, memory_size)

        # erase_signal  Shape (batch_size, memory_state_dim)
        erase_signal = F.FullyConnected(data=control_input,
                                        num_hidden=self.memory_state_dim,
                                        weight=erase_signal_weight,
                                        bias=erase_signal_bias)
        erase_signal = F.Activation(data=erase_signal, act_type='sigmoid', name=self.name + "_erase_signal")
        # add_signal  Shape (batch_size, memory_state_dim)
        add_signal = F.FullyConnected(data=control_input,
                                      num_hidden=self.memory_state_dim,
                                      weight=add_signal_weight,
                                      bias=add_signal_bias)
        add_signal = F.Activation(data=add_signal, act_type='tanh', name=self.name + "_add_signal")
        # erase_mult  Shape (batch_size, memory_size, memory_state_dim)
        erase_mult = 1 - F.batch_dot(F.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                     F.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)),
                                     name=self.name + "_erase_mult")

        aggre_add_signal = F.batch_dot(F.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                       F.Reshape(add_signal, shape=(-1, 1, self.memory_state_dim)),
                                       name=self.name + "_aggre_add_signal")
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory


class DKVMNCell(gluon.HybridBlock):
    def __init__(self, key_memory_size, key_memory_state_dim, value_memory_size, value_memory_state_dim,
                 prefix=None, params=None):
        super(DKVMNCell, self).__init__(prefix, params)
        self._modified = False
        self.reset()

        with self.name_scope():
            self.key_head = KVMNReadCell(
                memory_size=key_memory_size,
                memory_state_dim=key_memory_state_dim,
                prefix=self.prefix + "->key_head"
            )
            self.value_head = KVMNWriteCell(
                memory_size=value_memory_size,
                memory_state_dim=value_memory_state_dim,
                prefix=self.prefix + "->value_head"
            )

        self.key_memory_size = key_memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_size = value_memory_size
        self.value_memory_state_dim = value_memory_state_dim

    def forward(self, *args):
        """Unrolls the recurrent cell for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            Input symbol, 2D, of shape (batch_size * num_units).
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
            This can be used as an input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        # pylint: disable= arguments-differ
        self._counter += 1
        return super(DKVMNCell, self).forward(*args)

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1
        for cell in self._children.values():
            cell.reset()

    def begin_state(self, batch_size=0, func=ndarray.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var etc`. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.
        batch_size: int, default 0
            Only required for NDArray API. Size of the batch ('N' in layout)
            dimension of input.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. ZoneoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        states = []
        for info in self.state_info(batch_size):
            self._init_counter += 1
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter),
                         **info)
            states.append(state)
        return states

    def state_info(self, batch_size=0):
        return [
            {'shape': (batch_size, self.key_memory_size, self.key_memory_state_dim), '__layout__': 'NC'},
            {'shape': (batch_size, self.value_memory_size, self.key_memory_state_dim), '__layout__': 'NC'}
        ]

    def _alias(self):
        return 'dkvmn_cell'

    def attention(self, F, control_input, memory):
        correlation_weight = self.key_head.addressing(F, control_input=control_input, memory=memory)
        return correlation_weight  # (batch_size, memory_size)

    def read(self, F, read_weight, memory):
        read_content = self.value_head.read(F, memory=memory, read_weight=read_weight)
        return read_content  # (batch_size, memory_state_dim)

    def write(self, F, write_weight, control_input, memory):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=memory,
                                             write_weight=write_weight)
        return memory_value

    def hybrid_forward(self, F, keys, values, key_memory, value_memory):
        # Attention
        correlation_weight = self.attention(F, keys, key_memory)

        # Read Process
        read_content = self.read(F, correlation_weight, value_memory)

        # Write Process
        next_value_memory = self.write(F, correlation_weight, values, value_memory)

        return read_content, [key_memory, next_value_memory]

    def unroll(self, length, keys, values, key_memory, value_memory, layout='NTC', merge_outputs=None,
               valid_length=None):
        """Unrolls an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if `layout` is 'NTC',
            or (length, batch_size, ...) if `layout` is 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_memory : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if `None`.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If `False`, returns outputs as a list of Symbols.
            If `True`, concatenates output across time steps
            and returns a single symbol with shape
            (batch_size, length, ...) if layout is 'NTC',
            or (length, batch_size, ...) if layout is 'TNC'.
            If `None`, output whatever is faster.
        valid_length : Symbol, NDArray or None
            `valid_length` specifies the length of the sequences in the batch without padding.
            This option is especially useful for building sequence-to-sequence models where
            the input and output sequences would potentially be padded.
            If `valid_length` is None, all sequences are assumed to have the same length.
            If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
            The ith element will be the length of the ith sequence in the batch.
            The last valid state will be return and the padded outputs will be masked with 0.
            Note that `valid_length` must be smaller or equal to `length`.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
        """
        # pylint: disable=too-many-locals
        self.reset()

        keys, axis, F, batch_size = format_sequence(length, keys, layout, False)
        values, axis, F, batch_size = format_sequence(length, values, layout, False)

        states = F.broadcast_to(F.expand_dims(value_memory, axis=0),
                                shape=(batch_size, self.value_memory_size, self.value_memory_state_dim))
        outputs = []
        all_states = []
        for i in range(length):
            output, [_, new_states] = self(keys[i], values[i], key_memory, states)
            states = new_states
            outputs.append(output)
            if valid_length is not None:
                all_states.append(states)
        if valid_length is not None:
            states = [F.SequenceLast(F.stack(*ele_list, axis=0),
                                     sequence_length=valid_length,
                                     use_sequence_length=True,
                                     axis=0)
                      for ele_list in zip(*all_states)]
            outputs = mask_sequence_variable_length(F, outputs, length, valid_length, axis, True)

        # all_read_value_content = F.Concat(*outputs, num_args=length, dim=0)
        outputs, _, _, _ = format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states


class DKVMN(gluon.HybridBlock):
    def __init__(self, ku_num, key_embedding_dim, value_embedding_dim, hidden_num,
                 key_memory_size, value_memory_size=None, key_memory_state_dim=None, value_memory_state_dim=None,
                 nettype="DKVMN", dropout=0.0,
                 key_memory_initializer=None, value_memory_initializer=None,
                 **kwargs):
        super(DKVMN, self).__init__(kwargs.get("prefix"), kwargs.get("params"))

        ku_num = int(ku_num)
        key_embedding_dim = int(key_embedding_dim)
        value_embedding_dim = int(value_embedding_dim)
        hidden_num = int(hidden_num)
        key_memory_size = int(key_memory_size)
        value_memory_size = int(value_memory_size) if value_memory_size is not None else key_memory_size

        self.length = None
        self.nettype = nettype
        self._mask = None

        key_memory_state_dim = int(key_memory_state_dim) if key_memory_state_dim else key_embedding_dim
        value_memory_state_dim = int(value_memory_state_dim) if value_memory_state_dim else value_embedding_dim

        with self.name_scope():
            self.key_memory = self.params.get(
                'key_memory', shape=(key_memory_size, key_memory_state_dim),
                init=key_memory_initializer,
            )

            self.value_memory = self.params.get(
                'value_memory', shape=(value_memory_size, value_memory_state_dim),
                init=value_memory_initializer,
            )

            embedding_dropout = kwargs.get("embedding_dropout", 0.2)
            self.key_embedding = gluon.nn.Embedding(ku_num, key_embedding_dim)
            self.value_embedding = gluon.nn.Embedding(2 * ku_num, value_embedding_dim)
            self.embedding_dropout = gluon.nn.Dropout(embedding_dropout)

            self.dkvmn = DKVMNCell(key_memory_size, key_memory_state_dim, value_memory_size, value_memory_state_dim)
            self.input_nn = gluon.nn.Dense(50, flatten=False)  # 50 is set by the paper authors
            self.input_act = gluon.nn.Activation('tanh')
            self.read_content_nn = gluon.nn.Dense(hidden_num, flatten=False)
            self.read_content_act = gluon.nn.Activation('tanh')
            self.dropout = gluon.nn.Dropout(dropout)
            self.nn = gluon.nn.HybridSequential()
            self.nn.add(
                gluon.nn.Dense(ku_num, activation="tanh", flatten=False),
                self.dropout,
                gluon.nn.Dense(1, flatten=False),
            )

    def __call__(self, *args, mask=None):
        self._mask = mask
        result = super(DKVMN, self).__call__(*args)
        self._mask = None
        return result

    def hybrid_forward(self, F, questions, responses, key_memory, value_memory, *args, **kwargs):
        length = self.length if self.length else len(responses[0])

        q_data = self.embedding_dropout(self.key_embedding(questions))
        r_data = self.embedding_dropout(self.value_embedding(responses))

        read_contents, states = self.dkvmn.unroll(
            length, q_data, r_data, key_memory, value_memory, merge_outputs=True
        )

        input_embed_content = self.input_act(self.input_nn(q_data))
        read_content_embed = self.read_content_act(
            self.read_content_nn(
                F.Concat(read_contents, input_embed_content, num_args=2, dim=2)
            )
        )

        output = self.nn(read_content_embed)
        output = F.sigmoid(output)
        output = F.squeeze(output, axis=2)
        return output, states
