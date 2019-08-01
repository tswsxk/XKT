# coding: utf-8
# create by tongshiwei on 2019-7-30

__all__ = ["DKVMN"]

from longling.ML.MxnetHelper.gallery.layer import get_begin_state, format_sequence, mask_sequence_variable_length
from mxnet import gluon


class KVMNCell(gluon.rnn.HybridRecurrentCell):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, memory_state_dim, memory_size,
                 erase_signal_weight_initializer=None, erase_signal_bias_initializer=None,
                 add_signal_weight_initializer=None, add_signal_bias_initializer=None,
                 input_size=0, prefix=None, params=None):
        super(KVMNCell, self).__init__(prefix=prefix, params=params)

        self._input_size = input_size

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

    def read(self, F, memory, control_input=None, read_weight=None):
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

    def write(self, F, memory, control_input, write_weight=None):
        if write_weight is None:
            write_weight = self.addressing(
                F, control_input=control_input, memory=memory
            )  # Shape Shape (batch_size, memory_size)

        # erase_signal  Shape (batch_size, memory_state_dim)
        erase_signal = F.FullyConnected(data=control_input,
                                        num_hidden=self.memory_state_dim,
                                        weight=self.erase_signal_weight,
                                        bias=self.erase_signal_bias)
        erase_signal = F.Activation(data=erase_signal, act_type='sigmoid', name=self.name + "_erase_signal")
        # add_signal  Shape (batch_size, memory_state_dim)
        add_signal = F.FullyConnected(data=control_input,
                                      num_hidden=self.memory_state_dim,
                                      weight=self.add_signal_weight,
                                      bias=self.add_signal_bias)
        add_signal = F.Activation(data=add_signal, act_type='tanh', name=self.name + "_add_signal")
        # erase_mult  Shape (batch_size, memory_size, memory_state_dim)
        erase_mult = 1 - F.batch_dot(F.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                     F.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)))

        aggre_add_signal = F.batch_dot(F.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                       F.Reshape(add_signal, shape=(-1, 1, self.memory_state_dim)))
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory

    def hybrid_forward(self, F, memory, control_input, read_weight=None, write_weight=None):
        read_content = self.read(F, memory=memory, control_input=control_input, read_weight=read_weight)
        next_memory = self.write(F, memory=memory, control_input=control_input, write_weight=write_weight)
        return read_content, [next_memory, ]


class DKVMNCell(gluon.rnn.HybridRecurrentCell):
    def __init__(self, key_memory_size, key_memory_state_dim, value_memory_size, value_memory_state_dim,
                 prefix=None, params=None):
        super(DKVMNCell, self).__init__(prefix, params)

        with self.name_scope():
            self.key_head = KVMNCell(
                memory_size=key_memory_size,
                memory_state_dim=key_memory_state_dim,
                prefix=self.prefix + "->key_head"
            )
            self.value_head = KVMNCell(
                memory_size=value_memory_size,
                memory_state_dim=value_memory_state_dim,
                prefix=self.prefix + "->value_head"
            )

        self.key_memory_size = key_memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_size = value_memory_size
        self.value_memory_state_dim = value_memory_state_dim

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
        memory_value = self.value_head.write(F, control_input=control_input,
                                             memory=memory,
                                             write_weight=write_weight)
        return memory_value

    def hybrid_forward(self, F, keys, values, memories):
        key_memory, value_memory = memories
        # Attention
        correlation_weight = self.attention(F, keys, key_memory)

        # Read Process
        read_content = self.read(F, correlation_weight, value_memory)

        # Write Process
        next_value_memory = self.write(F, correlation_weight, values, value_memory)

        return read_content, [key_memory, next_value_memory]

    def unroll(self, length, keys, values, begin_memory=None, layout='NTC', merge_outputs=None,
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
        begin_memory = get_begin_state(self, F, begin_memory, keys, batch_size)

        states = begin_memory
        outputs = []
        all_states = []
        for i in range(length):
            output, states = self(keys[i], values[i], states)
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
        outputs, _, _, _ = format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states


class DKVMN(gluon.HybridBlock):
    def __init__(self, ku_num, key_embedding_dim, value_embedding_dim, hidden_num,
                 key_memory_size, key_memory_state_dim, value_memory_size, value_memory_state_dim,
                 nettype="DKVMN", dropout=0.0, **kwargs):
        super(DKVMN, self).__init__(kwargs.get("prefix"), kwargs.get("params"))

        self.length = None
        self.nettype = nettype

        with self.name_scope():
            embedding_dropout = kwargs.get("embedding_dropout", 0.2)
            self.key_embedding = gluon.nn.Embedding(ku_num, key_embedding_dim)
            self.value_embedding = gluon.nn.Embedding(2 * ku_num, value_embedding_dim)
            self.embedding_dropout = gluon.nn.Dropout(embedding_dropout)

            self.dkvmn = DKVMNCell(key_memory_size, key_memory_state_dim, value_memory_size, value_memory_state_dim)
            self.input_nn = gluon.nn.Dense(50, flatten=False)
            self.input_act = gluon.nn.Activation('tanh')
            self.read_content_nn = gluon.nn.Dense(hidden_num, flatten=False)
            self.read_content_act = gluon.nn.Activation('tanh')
            self.dropout = gluon.nn.Dropout(dropout)
            self.nn = gluon.nn.Dense(ku_num, flatten=False)

    def hybrid_forward(self, F, questions, responses, mask=None, begin_state=None, *args, **kwargs):
        length = self.length if self.length else len(responses[0])

        q_data = self.embedding_dropout(self.key_embedding(questions))
        r_data = self.embedding_dropout(self.value_embedding(responses))

        read_contents, states = self.dkvmn.unroll(length, q_data, r_data, begin_memory=begin_state, merge_outputs=True)

        input_embed_content = self.input_act(self.input_nn(q_data))
        read_content_embed = self.read_content_act(
            self.read_content_nn(
                F.Concat(read_contents, input_embed_content, num_args=2, dim=1)
            )
        )

        output = self.nn(self.dropout(read_content_embed))
        output = F.sigmoid(output)

        return output, states
