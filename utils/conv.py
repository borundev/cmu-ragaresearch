from tensorflow import keras
import numpy as np
import tensorflow as tf

class PeriodicPaddingConv2D(keras.layers.Conv2D):

    # Note this ONLY works now for filter size 3 and stride of 1
    # Based on  https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding

    def __init__(self, *args, **kwargs):
        padding_val = kwargs.pop('padding', None)
        if padding_val == 'same':
            print('For PeriodPaddingConv2D padding must be valid so setting it to valid.')
            kwargs['padding'] = 'valid'
        super().__init__(*args, **kwargs)
        stride_1 = self.kernel_size[0]
        self.l = list(range(12))
        self.l = self.l[-stride_1 // 2 + 1:] + self.l + self.l[:stride_1 // 2]
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(0, stride_1 // 2))

    def call(self, x):
        pre = tf.constant(np.diag(np.ones(12)).take(self.l, axis=0), dtype=np.float32)
        x = self.zero_padding(x)
        x = tf.transpose(tf.tensordot(pre, x, axes=[1, 1]), (1, 0, 2, 3))
        return super().call(x)


class OnlyTimeConvolution(tf.keras.layers.Layer):

    def __init__(self, num_filters, filter_size, *args, **kwargs):
        name = kwargs.pop('name', None)
        super().__init__(name=name)
        self.tr1 = keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))
        self.tr2 = keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))
        self.c = tf.keras.layers.Conv1D(num_filters,
                                        filter_size,
                                        data_format='channels_last', *args, **kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[1]

    def call(self, x):
        x = self.tr1(x)
        x = tf.stack([self.c(x[:, :, i]) for i in range(self.num_channels)], 2)
        x = self.tr2(x)
        return x