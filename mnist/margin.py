from __future__ import absolute_import, division, print_function

import lasagne
import numpy as np
from theano import tensor as T
from theano import gradient
from utils import jacobian

multiclass_margin = lasagne.objectives.multiclass_hinge_loss


def margin_sensitivity(inputs, logits, labels, num_outputs, ord=2):
    """Compute margin sensitivity (proposed regularization).
    """
    assert ord in [2, np.inf]

    batch_size = inputs.shape[0]
    batch_indices = T.arange(batch_size)

    # shape: labels, batch, channels, height, width
    jac = jacobian(logits, inputs, num_outputs=num_outputs, pack_dim=0)

    # basically jac_labels = jac[labels, batch_indices]
    jac_flt = jac.reshape((-1, inputs.shape[1], inputs.shape[2],
                           inputs.shape[3]))
    jac_labels_flt = jac_flt[labels * batch_size + batch_indices]
    jac_labels = jac_labels_flt.reshape(inputs.shape)

    w = jac - T.shape_padaxis(jac_labels, axis=0)
    reduce_ind = range(2, inputs.ndim + 1)
    if ord == 2:
        dist = T.sum(w**2, axis=reduce_ind)
    elif ord == np.inf:
        dist = T.sum(T.abs_(w), axis=reduce_ind)
    else:
        raise ValueError

    l = T.argmax(dist, axis=0)
    l = gradient.disconnected_grad(l)

    corrects = logits[batch_indices, labels]
    others = logits[batch_indices, l]

    corrects_grad = T.grad(corrects.sum(), inputs)
    others_grad = T.grad(others.sum(), inputs)
    reduce_ind = range(1, inputs.ndim)
    if ord == 2:
        return T.sum((corrects_grad - others_grad)**2, axis=reduce_ind)
    elif ord == np.inf:
        return T.sum(T.abs_(corrects_grad - others_grad), axis=reduce_ind)
    else:
        raise ValueError
