class RepeatVector(Layer):
    """Repeats the input n times.
    # Example
    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension
        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```
    # Arguments
        n: integer, repetition factor.
    # Input shape
        2D tensor of shape `(num_samples, features)`.
    # Output shape
        3D tensor of shape `(num_samples, n, features)`.
    """

    def __init__(self, n, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, inputs):
        return K.repeat(inputs, self.n)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))