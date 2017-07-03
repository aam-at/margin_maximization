from __future__ import absolute_import, division, print_function

from lasagne.utils import floatX
from theano import gradient
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import normalize_perturbation


def kl_with_logits(q_logits, p_logits):
    q = T.nnet.softmax(q_logits)
    q_log = T.nnet.logsoftmax(q_logits)
    p_log = T.nnet.logsoftmax(p_logits)
    loss = T.sum(q * (q_log - p_log), axis=1)
    return loss


def virtual_adversarial_perturbation(predict_fn, inputs, logits, epsilon,
                                     num_iterations=1, xi=1e-6, seed=12345):
    epsilon = floatX(epsilon)
    xi = floatX(xi)
    rng = RandomStreams(seed=seed)
    d = rng.normal(size=inputs.shape, dtype=inputs.dtype)
    for i in range(num_iterations):
        d = xi * normalize_perturbation(d)
        logits_d = predict_fn(inputs + d)
        kl = kl_with_logits(logits, logits_d)
        Hd = T.grad(kl.sum(), d)
        d = gradient.disconnected_grad(Hd)
    return epsilon * normalize_perturbation(d)


def virtual_adversarial_training(predict_fn, inputs, logits, epsilon,
                                 num_iterations=1, xi=1e-6):
    vat_perturbation = virtual_adversarial_perturbation(
        predict_fn, inputs, logits, epsilon, num_iterations, xi)
    logits_vat = predict_fn(inputs + vat_perturbation)
    loss = kl_with_logits(gradient.disconnected_grad(logits), logits_vat)
    return loss
