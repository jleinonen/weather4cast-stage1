import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Layer
from tensorflow.keras.layers import Activation, ReLU, LeakyReLU
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import TimeDistributed, Lambda

from blocks import conv_block, res_block, ResBlock
import datasets
from layers import ReflectionPadding2D, SpatialWarp
from rnn import CustomGateGRU, ResGRU


file_dir = os.path.dirname(os.path.abspath(__file__))


def rnn2_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = [past_in]

    def conv_gate(channels, activation='sigmoid'):
        return Conv2D(channels, kernel_size=(3,3), padding='same',
            activation=activation)

    # recurrent downsampling
    block_channels = [32, 64, 128, 256]
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = res_block(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = CustomGateGRU(
            update_gate=conv_gate(channels),
            reset_gate=conv_gate(channels),
            output_gate=conv_gate(channels, activation=None),
            return_sequences=True,
            time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(
            Conv2D(channels, kernel_size=(3,3), 
                activation='relu', padding='same')(xt[:,-1,...])
        )

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = CustomGateGRU(
            update_gate=conv_gate(channels),
            reset_gate=conv_gate(channels),
            output_gate=conv_gate(channels, activation=None),
            return_sequences=True,
            time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = res_block(block_channels[max(i-1,0)],
            time_dist=True, activation='relu')(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1),
        activation='sigmoid'))(xt)

    outputs = [
        Lambda(lambda x: x, name=name)(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def rnn3_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = [past_in]

    block_channels = [32, 64, 128, 256]

    # recurrent downsampling    
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = ResBlock(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = ResGRU(
            channels, return_sequences=True, time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(ResBlock(channels)(xt[:,-1,...]))

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ResGRU(
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ResBlock(block_channels[max(i-1,0)], time_dist=True)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1),
        activation='sigmoid'))(xt)

    outputs = [
        Lambda(lambda x: x, name=name)(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def rounded_mse(y_true, y_pred):
    return tf.reduce_mean(
        tf.square(tf.math.round(y_pred)-y_true),
        axis=-1
    )


def logit(x):
    return tf.math.log(x/(1-x))


def normlogit_loss(y_true, y_pred, M=0.997, m=0.003):
    #log (M) = -log (m) = 5.806
    y_true = tf.clip_by_value(y_true, m, M)
    y_pred = tf.clip_by_value(y_pred, m, M)
    lM = -logit(m)
    y_true = (logit(y_true) + lM) / (2*lM)
    y_pred = (logit(y_pred) + lM) / (2*lM)

    return tf.reduce_mean(tf.square(y_pred-y_true), axis=-1)


variable_weights = {
    "temperature": 1.0/0.03163512,
    "crr_intensity": 1.0/0.00024158,
    "asii_turb_trop_prob": 1.0/0.00703378,
    "cma": 1.0/0.19160305
}


def compile_model(model, output_vars, optimizer='adam'):
    var_losses = {
        "asii_turb_trop_prob": normlogit_loss,
    }
    losses = [var_losses.get(v, 'mse') for v in output_vars]
    if len(output_vars) > 1:        
        weights = [variable_weights[v]/len(output_vars) for v in output_vars]
    else:
        weights = [1.0]

    if "cma" in output_vars:
        metrics = {"cma": rounded_mse}
    else:
        metrics = None

    if compile_model:
        model.compile(loss=losses, loss_weights=weights,
            optimizer=optimizer, metrics=metrics)


def init_model(batch_gen, model_func=rnn3_model, compile=True, 
    init_strategy=True, **kwargs):

    (past_timesteps, future_timesteps) = batch_gen.sequence_length
    (num_inputs, num_outputs) = batch_gen.num_vars

    output_vars = []
    for (_, variables) in batch_gen.variables[1].items():
        output_vars.extend(variables)

    if init_strategy and tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = model_func(
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            output_names=output_vars,
            **kwargs
        )

    if compile:
        compile_model(model, output_vars)
    
    return model


def combined_model(models, output_names):
    past_in = Input(shape=models[0].input_shape[1:],
        name="past_in")
    outputs = [
        Layer(name=name)(model(past_in))
        for (model, name) in zip(models, output_names)
    ]
    comb_model = Model(inputs=[past_in], outputs=outputs)

    return comb_model


def combined_model_with_weights(
    batch_gen_valid,
    var_weights=[ # this is the best combination used in the competition
        ("CTTH", "temperature", "../models/best/resrnn-temperature.h5", rnn3_model),
        ("CRR", "crr_intensity", "../models/best/rnn-crr_intensity.h5", rnn2_model),
        ("ASII", "asii_turb_trop_prob", "../models/best/resrnn-asii_turb_trop_prob.h5", rnn3_model),
        ("CMA", "cma", "../models/best/resrnn-cma.h5", rnn3_model)
    ],
):
    models = []
    output_vars = []
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        for (dataset, variable, weights, model_func) in var_weights:
            datasets.setup_univariate_batch_gen(batch_gen_valid, dataset, variable)
            model = init_model(batch_gen_valid, model_func=model_func, 
                compile=False, init_strategy=False)
            model.load_weights(weights)
            models.append(model)
            output_vars.append(variable)

        comb_model = combined_model(models, output_vars)
        compile_model(comb_model, output_vars)

    return comb_model


def train_model(model, batch_gen_train, batch_gen_valid,
    weight_fn="model.h5", monitor="val_loss"):

    fn = os.path.join(file_dir, "..", "models", weight_fn)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn, save_weights_only=True, save_best_only=True, mode="min",
        monitor=monitor
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=3, mode="min", factor=0.2, monitor=monitor
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        patience=10, mode="min", restore_best_weights=True,
        monitor=monitor
    )
    callbacks = [checkpoint, reducelr, earlystop]

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.OFF
    model.fit(
        batch_gen_train,
        epochs=100,
        steps_per_epoch=len(batch_gen_train)//2,
        validation_data=batch_gen_valid,
        callbacks=callbacks
    )
