from __future__ import absolute_import, division, print_function

import logging


import lasagne
from lasagne.layers import Conv2DLayer as conv
from lasagne.layers import DenseLayer as dense
from lasagne.layers import MaxPool2DLayer as maxpool
from lasagne.layers import NonlinearityLayer as nonlin
from lasagne.layers import InputLayer, dropout
from theano import tensor as T
from theano import gradient

logger = logging.getLogger()
default_nonlinearity = lasagne.nonlinearities.rectify
leaky_relu = lasagne.nonlinearities.leaky_rectify


def with_end_points(net):

    def gather_end_points(inputs_var, *args, **kwargs):
        logits = lasagne.layers.get_output(net, inputs=inputs_var, **kwargs)
        predictions = gradient.disconnected_grad(T.argmax(logits, axis=1))
        prob = T.nnet.softmax(logits)
        end_points = {
            'logits': logits,
            'predictions': predictions,
            'prob': prob
        }
        return end_points

    return gather_end_points


def create_network(model_name, inputs_shape, **kwargs):
    if model_name == 'mlp':
        net = mlp(inputs_shape, **kwargs)
    elif model_name == 'lenet5':
        net = lenet5(inputs_shape)
    else:
        raise ValueError
    return net


def mlp(inputs_shape, layer_dims, nonlinearity=default_nonlinearity,
        use_dropout=False, **kwargs):
    logger.warning("Unrecognized options to the model: %s", kwargs)
    l = lasagne.layers.InputLayer(inputs_shape)
    W_init = lasagne.init.GlorotUniform()
    for i, layer_size in enumerate(layer_dims[:-1]):
        assert layer_size >= 0
        l = dense(l, layer_size, W=W_init, nonlinearity=None)
        l = nonlin(l, nonlinearity=nonlinearity)
        if use_dropout:
            l = dropout(l)
    logits = dense(l, layer_dims[-1], W=W_init, nonlinearity=None)
    return logits


def lenet5(inputs_shape, nonlinearity=default_nonlinearity,
           use_dropout=False, **kwargs):
    logger.warning("Unrecognized options to the model: %s", kwargs)

    l = InputLayer(inputs_shape)
    l = conv(l, 32, 5, nonlinearity=None)
    l = nonlin(l, nonlinearity=nonlinearity)
    l = maxpool(l, 2)

    l = conv(l, 64, 5, nonlinearity=None)
    l = nonlin(l, nonlinearity=nonlinearity)
    l = maxpool(l, 2)
    if use_dropout:
        l = dropout(l, p=0.5)
    l = dense(l, 512, nonlinearity=None)
    l = nonlin(l, nonlinearity=nonlinearity)
    if use_dropout:
        l = dropout(l, p=0.5)
    # output layers
    logits = dense(l, 10, nonlinearity=None)
    return logits
