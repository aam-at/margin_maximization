from __future__ import absolute_import, division, print_function

import numpy as np

from lasagne.objectives import categorical_crossentropy
from lasagne.utils import floatX
from theano import tensor as T
from theano import gradient


def fast_gradient_perturbation(inputs, logits, labels=None, epsilon=0.3,
                               ord=np.inf):
    epsilon = floatX(epsilon)
    if labels is None:
        raise ValueError
    nll = categorical_crossentropy(logits, labels)
    grad = T.grad(nll.sum(), inputs, consider_constant=[labels])
    if ord == np.inf:
        perturbation = T.sgn(grad)
    elif ord == 1:
        sum_ind = list(range(1, inputs.ndim))
        perturbation = grad / T.sum(T.abs_(grad), axis=sum_ind, keepdims=True)
    elif ord == 2:
        sum_ind = list(range(1, inputs.ndim))
        perturbation = grad / T.sqrt(
            T.sum(grad**2, axis=sum_ind, keepdims=True))
    perturbation *= epsilon
    return gradient.disconnected_grad(perturbation)


def adversarial_training(model, inputs, labels, epsilon):
    logits = model(inputs)
    fast_grad_perturbation = fast_gradient_perturbation(
        inputs, logits, labels, epsilon)
    logits_adversarial = model(inputs + fast_grad_perturbation)
    loss = categorical_crossentropy(logits_adversarial, labels)
    return loss
