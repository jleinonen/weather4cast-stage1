import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Layer
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.keras.regularizers import l2
from layers import ReflectionPadding2D


def conv_block(channels, conv_size=(3,3), time_dist=False,
    norm='batch', stride=1, activation='relu', padding='reflect'):

    TD = TimeDistributed if time_dist else (lambda x: x)

    def block(x):
        if padding == 'reflect':
            pad = tuple((s-1)//2 for s in conv_size)
            x = TD(ReflectionPadding2D(padding=pad))(x)
        x = TD(Conv2D(channels, conv_size, 
            padding='valid' if padding=='reflect' else padding,
            strides=(stride,stride), 
            #kernel_regularizer=(l2(1e-4) if norm!="spectral" else None)
        ))(x)
        if activation == 'leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'relu':
            x = ReLU()(x)
        elif activation == 'elu':
            x = ELU()(x)
        if norm=="batch":
            scale = (activation not in ['leakyrelu', 'relu'])
            x = BatchNormalization(momentum=0.95, scale=scale)(x)
        return x

    return block


def res_block(channels, conv_size=(3,3), stride=1, norm='batch',
    time_dist=False, activation='leakyrelu'):

    TD = TimeDistributed if time_dist else (lambda x: x)

    def block(x):
        in_channels = int(x.shape[-1])
        x_in = x
        if (stride > 1):
            x_in = TD(AveragePooling2D(pool_size=(stride,stride)))(x_in)
        if (channels != in_channels):
            x_in = conv_block(channels, conv_size=(1,1), stride=1, 
                activation=False, time_dist=time_dist)(x_in)

        x = conv_block(channels, conv_size=conv_size, stride=stride,
            padding='reflect', norm=norm, time_dist=time_dist,
            activation=activation)(x)
        x = conv_block(channels, conv_size=conv_size, stride=1,
            padding='reflect', norm=norm, time_dist=time_dist,
            activation=activation)(x)

        x = Add()([x,x_in])

        return x

    return block



class ConvBlock(Layer):
    def __init__(self, channels, conv_size=(3,3), time_dist=False,
        norm='none', stride=1, activation='relu', padding='same',
        order=("conv", "act", "norm"), scale_norm=False):

        super().__init__()
        TD = TimeDistributed if time_dist else (lambda x: x)
        
        if padding == 'reflect':
            pad = tuple((s-1)//2 for s in conv_size)
            self.padding = TD(ReflectionPadding2D(padding=pad))
        else:
            self.padding = lambda x: x
        
        self.conv = TD(Conv2D(
            channels, conv_size, 
            padding='valid' if padding=='reflect' else padding,
            strides=(stride,stride), 
        ))

        if activation == 'leakyrelu':
            self.act = LeakyReLU(0.2)
        elif activation == 'relu':
            self.act = ReLU()
        elif activation == 'elu':
            self.act = ELU()
        else:
            self.act = Activation(activation)

        if norm == "batch":
            self.norm = BatchNormalization(momentum=0.95, scale=scale_norm)
        elif norm == "layer":
            self.norm = LayerNormalization(scale=scale_norm)
        else:
            self.norm = lambda x: x

        self.order = order

    def call(self, x):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(self.padding(x))
            elif layer == "act":
                x = self.act(x)
            elif layer == "norm":
                x = self.norm(x)
            else:
                raise ValueError("Unknown layer {}".format(layer))
        return x


class ResBlock(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels
        self.stride = kwargs.pop("stride", 1)
        time_dist = kwargs.get("time_dist", False)

        TD = TimeDistributed if time_dist else (lambda x: x)

        if self.stride > 1:
            self.pool = TD(AveragePooling2D(
                pool_size=(self.stride,self.stride)))
        else:
            self.pool = lambda x: x
        self.proj = TD(Conv2D(self.channels, kernel_size=(1,1)))
        
        self.conv_block_1 = ConvBlock(channels, stride=self.stride, **kwargs)
        self.conv_block_2 = ConvBlock(channels, activation='leakyrelu', **kwargs)
        self.add = Add()

    def call(self, x):
        x_in = self.pool(x)
        in_channels = int(x.shape[-1])        
        if in_channels != self.channels:        
            x_in = self.proj(x_in)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return self.add([x,x_in])


class GRUResBlock(ResBlock):
    def __init__(self, channels, final_activation='sigmoid', **kwargs):
        super().__init__(channels, **kwargs)
        self.final_act = Activation(final_activation)

    def call(self, x):
        x = super().call(x)
        return self.final_act(x)
