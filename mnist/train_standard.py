from __future__ import absolute_import, division, print_function

import logging
import os
import time
from collections import OrderedDict

import numpy as np
import theano
from lasagne.layers import get_all_params
from lasagne.objectives import categorical_accuracy, categorical_crossentropy
from lasagne.updates import adam
from lasagne.utils import floatX
from theano import tensor as T

import flags
from data import batch_iterator, mnist_load, select_balanced_subset
from deepfool import deepfool
from models import create_network, with_end_points
from utils import (build_result_str, save_images, save_network,
                   setup_train_experiment)

# experiment parameters
flags.DEFINE_integer("seed", 1, "experiment seed")
flags.DEFINE_string("name", None, "name of the experiment")
flags.DEFINE_string("data_dir", "data", "path to data")
flags.DEFINE_string("train_dir", "runs", "path to working dir")

# gan model parameters
flags.DEFINE_string("model", "mlp", "model name (mlp or mlp_with_bn)")
flags.DEFINE_string("layer_dims", "1000-1000-1000-10", "dimensions of fully-connected layers")
flags.DEFINE_bool("use_dropout", False, "whenever to use dropout or not")

# adversary parameters
flags.DEFINE_integer("deepfool_iter", 25, "maximum number of deepfool iterations")
flags.DEFINE_float("deepfool_clip", 0.5, "perturbation clip during search")
flags.DEFINE_float("deepfool_overshoot", 0.02, "overshoot to improve speed of convergence")

# data parameters
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("test_batch_size", 100, "test batch size")
flags.DEFINE_integer("train_size", 50000, "training size")

# training parameters
flags.DEFINE_integer("num_epochs", 100, "number of epochs to run")
flags.DEFINE_float("initial_learning_rate", 0.001, "initial learning rate")
flags.DEFINE_float("learning_rate_decay_factor", 0.95, "learning rate decay factor")
flags.DEFINE_float("start_learning_rate_decay", 0, "learning rate decay factor")

# logging parameters
flags.DEFINE_integer("checkpoint_frequency", 10, "checkpoint frequency (in epochs)")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in epochs)")
flags.DEFINE_integer("summary_samples_per_class", 10, "number of samples per class to generate")

FLAGS = flags.FLAGS
logger = logging.getLogger()


def main():
    setup_train_experiment(logger, FLAGS, "%(model)s_standard" )

    logger.info("Loading data...")
    data = mnist_load(FLAGS.train_size, FLAGS.seed)
    X_train, y_train = data.X_train, data.y_train
    X_val, y_val = data.X_val, data.y_val
    X_test, y_test = data.X_test, data.y_test

    img_shape = [None, 1, 28, 28]
    train_images = T.tensor4('train_images')
    train_labels = T.lvector('train_labels')
    val_images = T.tensor4('valid_labels')
    val_labels = T.lvector('valid_labels')

    layer_dims = [int(dim) for dim in FLAGS.layer_dims.split("-")]
    num_classes = layer_dims[-1]
    net = create_network(FLAGS.model, img_shape, layer_dims=layer_dims)
    model = with_end_points(net)

    train_outputs = model(train_images)
    val_outputs = model(val_images, deterministic=True)

    # losses
    train_ce = categorical_crossentropy(train_outputs['prob'], train_labels).mean()
    train_loss = train_ce
    val_ce = categorical_crossentropy(val_outputs['prob'], val_labels).mean()
    val_deepfool_images = deepfool(
        lambda x: model(x, deterministic=True)['logits'], val_images, val_labels,
        num_classes, max_iter=FLAGS.deepfool_iter, clip_dist=FLAGS.deepfool_clip,
        over_shoot=FLAGS.deepfool_overshoot)

    # metrics
    train_acc = categorical_accuracy(train_outputs['logits'], train_labels).mean()
    train_err = 1.0 - train_acc
    val_acc = categorical_accuracy(val_outputs['logits'], val_labels).mean()
    val_err = 1.0 - val_acc
    # deepfool robustness
    reduc_ind = range(1, train_images.ndim)
    l2_deepfool = (val_deepfool_images - val_images).norm(2, axis=reduc_ind)
    l2_deepfool_norm = l2_deepfool / val_images.norm(2, axis=reduc_ind)

    train_metrics = OrderedDict([('loss', train_loss),
                                 ('nll', train_ce),
                                 ('err', train_err)])
    val_metrics = OrderedDict([('nll', val_ce),
                               ('err', val_err)])
    summary_metrics = OrderedDict([('l2', l2_deepfool.mean()),
                                   ('l2_norm', l2_deepfool_norm.mean())])

    lr = theano.shared(floatX(FLAGS.initial_learning_rate), 'learning_rate')
    train_params = get_all_params(net, trainable=True)
    train_updates = adam(train_loss, train_params, lr)

    logger.info("Compiling theano functions...")
    train_fn = theano.function([train_images, train_labels],
                               outputs=train_metrics.values(),
                               updates=train_updates)
    val_fn = theano.function([val_images, val_labels], outputs=val_metrics.values())
    summary_fn = theano.function([val_images, val_labels],
                                 outputs=summary_metrics.values() + [val_deepfool_images])

    logger.info("Starting training...")
    try:
        samples_per_class = FLAGS.summary_samples_per_class
        summary_images, summary_labels = select_balanced_subset(
            X_val, y_val, num_classes, samples_per_class)
        save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
        save_images(summary_images, save_path)

        epoch = 0
        batch_index = 0
        while epoch < FLAGS.num_epochs:
            epoch += 1

            start_time = time.time()
            train_iterator = batch_iterator(X_train, y_train, FLAGS.batch_size, shuffle=True)
            epoch_outputs = np.zeros(len(train_fn.outputs))
            for batch_index, (images, labels) in enumerate(train_iterator, batch_index + 1):
                batch_outputs = train_fn(images, labels)
                epoch_outputs += batch_outputs
            epoch_outputs /= X_train.shape[0] // FLAGS.batch_size
            logger.info(build_result_str("Train epoch [{}, {:.2f}s]:".format(
                epoch, time.time() - start_time),
                train_metrics.keys(),
                epoch_outputs))

            # update learning rate
            if epoch > FLAGS.start_learning_rate_decay:
                new_lr_value = lr.get_value() * FLAGS.learning_rate_decay_factor
                lr.set_value(floatX(new_lr_value))
                logger.debug("learning rate was changed to {:.10f}".format(new_lr_value))

            # validation
            start_time = time.time()
            val_iterator = batch_iterator(X_val, y_val, FLAGS.test_batch_size, shuffle=False)
            val_epoch_outputs = np.zeros(len(val_fn.outputs))
            for images, labels in val_iterator:
                val_epoch_outputs += val_fn(images, labels)
            val_epoch_outputs /= X_val.shape[0] // FLAGS.test_batch_size
            logger.info(build_result_str("Test epoch [{}, {:.2f}s]:".format(
                epoch, time.time() - start_time),
                val_metrics.keys(),
                val_epoch_outputs))

            if epoch % FLAGS.summary_frequency == 0:
                summary = summary_fn(summary_images, summary_labels)
                logger.info(build_result_str("Epoch [{}] adversarial statistics:".format(epoch),
                                             summary_metrics.keys(),
                                             summary[:-1]))
                save_path = os.path.join(FLAGS.samples_dir, 'epoch-%d.png' % epoch)
                df_images = summary[-1]
                save_images(df_images, save_path)

            if epoch % FLAGS.checkpoint_frequency == 0:
                save_network(net, epoch=epoch)
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt. Stopping training...")
    finally:
        save_network(net)

    # evaluate final model on test set
    test_iterator = batch_iterator(X_test, y_test, FLAGS.test_batch_size, shuffle=False)
    test_results = np.zeros(len(val_fn.outputs))
    for images, labels in test_iterator:
        test_results += val_fn(images, labels)
    test_results /= X_test.shape[0] // FLAGS.test_batch_size
    logger.info(build_result_str("Final test results:",
                                 val_metrics.keys(), test_results))


if __name__ == "__main__":
    main()
