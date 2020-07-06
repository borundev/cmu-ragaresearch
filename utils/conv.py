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


