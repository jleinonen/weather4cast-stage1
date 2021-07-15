from tensorflow.keras.layers import Layer
import tensorflow as tf
from blocks import GRUResBlock


class CustomGateGRU(Layer):
    def __init__(self, 
        update_gate=None, reset_gate=None, output_gate=None,
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h


class ConvGRU(Layer):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.reset_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.output_gate = Conv2D(channels, conv_size, padding='same')

        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h


class ResGRU(ConvGRU):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        super(ConvGRU, self).__init__(**kwargs)

        self.update_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same')
        self.reset_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same')
        self.output_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='linear', padding='same')

        self.return_sequences = return_sequences
        self.time_steps = time_steps
