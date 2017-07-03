from __future__ import absolute_import, division, print_function

import numpy as np
import theano
from theano import gradient, scan_module
from theano import tensor as T
from theano.ifelse import ifelse

from utils import jacobian


def deepfool(model, inputs, labels=None, num_classes=None, norm='l2',
             max_iter=25, clip_dist=None, over_shoot=0.02):
    """Theano implementation of DeepFool https://arxiv.org/abs/1511.04599

    """
    assert norm in ['l1', 'l2']

    if num_classes is None:
        raise RuntimeError("Number of classes need to be provided for Theano")
    if labels is None:
        labels = gradient.zero_grad(T.argmax(model(inputs), axis=1))

    batch_size = inputs.shape[0]
    batch_indices = T.arange(batch_size)

    def find_perturb(perturbation):
        logits_os = model(inputs + (1 + over_shoot) * perturbation)
        y_pred = T.argmax(logits_os, axis=1)
        is_mistake = T.neq(y_pred, labels)
        current_ind = batch_indices[(1 - is_mistake).nonzero()]
        should_stop = T.all(is_mistake)

        # continue generating perturbation only for correctly classified
        inputs_subset = inputs[current_ind]
        perturbation_subset = perturbation[current_ind]
        labels_subset = labels[current_ind]
        batch_subset = T.arange(inputs_subset.shape[0])

        x_adv = inputs_subset + perturbation_subset
        logits = model(x_adv)
        corrects = logits[batch_subset, labels_subset]
        jac = jacobian(logits, x_adv, num_classes)

        # deepfool
        f = logits - T.shape_padright(corrects)
        w = jac - T.shape_padaxis(jac[batch_subset, labels_subset], axis=1)
        reduce_ind = range(2, inputs.ndim + 1)
        if norm == 'l2':
            dist = T.abs_(f) / w.norm(2, axis=reduce_ind)
        else:
            dist = T.abs_(f) / T.sum(T.abs_(w), axis=reduce_ind)
        # remove correct targets
        dist = T.set_subtensor(dist[batch_subset, labels_subset], T.constant(np.inf))
        l = T.argmin(dist, axis=1)
        dist_l = dist[batch_subset, l].dimshuffle(0, 'x', 'x', 'x')
        # avoid numerical instability and clip max value
        if clip_dist is not None:
            dist_l = T.clip(dist_l, 0, clip_dist)
        w_l = w[batch_subset, l]
        if norm == 'l2':
            reduce_ind = range(1, inputs.ndim)
            perturbation_upd = dist_l * w_l / w_l.norm(2, reduce_ind, keepdims=True)
        else:
            perturbation_upd = dist_l * T.sgn(w_l)
        perturbation = ifelse(should_stop,
                              perturbation,
                              T.inc_subtensor(
                                  perturbation[current_ind],
                                  perturbation_upd))
        return perturbation, scan_module.until(should_stop)

    initial_perturbation = T.zeros_like(inputs)
    results, _ = theano.scan(
        find_perturb, outputs_info=[initial_perturbation], n_steps=max_iter)
    perturbation = results[-1]
    return gradient.disconnected_grad(inputs + (1 + over_shoot) * perturbation)

