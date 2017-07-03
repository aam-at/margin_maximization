from __future__ import absolute_import, division, print_function

import ast
import logging
import os
import six
import sys
from contextlib import contextmanager

import lasagne
import numpy as np
from lasagne.utils import floatX
from theano import tensor as T

from flags import FLAGS


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            value = self.__getitem__(key)
        except KeyError as exc:
            return None
        if isinstance(value, dict):
            value = AttributeDict(value)
        return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


def prepare_dir(dir_path, subdir_name):
    base = os.path.join(dir_path, subdir_name)
    i = 0
    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except OSError:
            i += 1
    return name


def load_training_params(path):
    with open(os.path.join(path, 'log.txt'), 'r') as f:
        params = ast.literal_eval(f.readline())
        return AttributeDict(params)


@contextmanager
def redirect_stderr(new_target):
    old_target, sys.stderr = sys.stderr, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


@contextmanager
def silence_stderr():
    with redirect_stderr(open(os.devnull, 'w')) as redirect:
        yield redirect


def setup_train_experiment(logger, FLAGS, default_name):
    np.random.seed(FLAGS.seed)

    if FLAGS.name is None:
        FLAGS.name = default_name % FLAGS.__dict__['__flags']

    if not os.path.exists(FLAGS.data_dir) or not os.path.isdir(FLAGS.data_dir):
        raise ValueError("Could not find folder %s" % FLAGS.data_dir)
    FLAGS.train_dir = prepare_dir(FLAGS.train_dir, FLAGS.name)
    FLAGS.chks_dir = os.path.join(FLAGS.train_dir, 'chks')
    FLAGS.samples_dir = os.path.join(FLAGS.train_dir, 'samples')
    os.makedirs(FLAGS.chks_dir)
    os.makedirs(FLAGS.samples_dir)

    # configure logging
    cmd_hndl = logging.StreamHandler()
    cmd_hndl.setLevel(logging.INFO)
    logger.addHandler(cmd_hndl)
    file_hndl = logging.FileHandler(os.path.join(FLAGS.train_dir, 'log.txt'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    logger.setLevel(logging.DEBUG)

    # print config
    logger.info(FLAGS.__dict__['__flags'])


def build_result_str(header, names, values):
    assert len(names) == len(values)
    string_buffer = six.StringIO()
    string_buffer.write(header)
    for name, value in zip(names, values):
        string_buffer.write(" {}: {:.6f},".format(name, np.asscalar(value)))
    return string_buffer.getvalue()[:-1]


def save_images(images, path):
    import torch
    from torchvision.utils import save_image
    save_image(
        torch.from_numpy(images), path, int(np.sqrt(images.shape[0])))


# Theano utils
def jacobian(outputs, inputs, num_outputs, pack_dim=1):
    batch_indices = T.arange(inputs.shape[0])
    jac = []
    outputs_flt = outputs.reshape((-1, 1))
    for i in range(num_outputs):
        outputs_i = outputs_flt[batch_indices * num_outputs + i]
        jac.append(T.grad(outputs_i.sum(), inputs))
    if pack_dim is not None:
        return T.stack(jac, axis=pack_dim)
    else:
        return jac


def jacobian_penalty(inputs, outputs, num_classes):
    jac = jacobian(outputs, inputs, num_classes)
    reduce_ind = range(2, inputs.ndim + 1)
    return T.sum(jac**2, axis=reduce_ind).mean()


def l2_normalize(x, axis, epsilon=1e-12):
    epsilon = floatX(epsilon)
    x /= (epsilon + T.max(T.abs_(x), axis, keepdims=True))
    square_sum = T.sum(T.sqr(x), axis, keepdims=True)
    x /= T.sqrt(np.sqrt(epsilon) + square_sum)
    return x


def normalize_perturbation(x):
    x_shape = x.shape
    x = T.reshape(x, (x.shape[0], -1))
    x_norm = l2_normalize(x, axis=1)
    return x_norm.reshape(x_shape)


# Lasagne utils
def save_network(net, name='model', epoch=None):
    save_path = os.path.join(FLAGS.train_dir, 'chks', name)
    if epoch is not None:
        save_path = "%s-%d" % (save_path, epoch)
    save_path += ".npz"
    np.savez(save_path, *lasagne.layers.get_all_param_values(net))
    if epoch is not None:
        logging.debug("Model saved to '{}'".format(save_path))
    else:
        logging.debug("Final model saved to '{}'".format(save_path))


def load_network(net, name='model', epoch=None):
    load_path = os.path.join(FLAGS.load_dir, 'chks', name)
    logging.debug("Model loaded from '{}'".format(load_path))
    if epoch is not None:
        load_path = "%s-%d" % (load_path, epoch)
    load_path += ".npz"
    with np.load(load_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)
