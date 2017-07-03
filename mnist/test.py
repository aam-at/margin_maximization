from __future__ import absolute_import, division, print_function

import logging
import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T
from lasagne.objectives import categorical_accuracy

import flags
from at import fast_gradient_perturbation
from data import batch_iterator, mnist_load, select_balanced_subset
from deepfool import deepfool
from models import create_network, with_end_points
from utils import (load_network, load_training_params, build_result_str,
                   save_images)

flags.DEFINE_string("load_dir", None, "path to load checkpoint from")
flags.DEFINE_integer("load_epoch", None, "epoch for which restore model")
flags.DEFINE_string("working_dir", "test", "path to working dir")
flags.DEFINE_bool("sort_labels", True, "sort labels")
flags.DEFINE_integer("batch_size", 100, "batch_index size (default: 100)")
flags.DEFINE_float("fgsm_epsilon", 0.2, "fast gradient epsilon (default: 0.2)")
flags.DEFINE_integer("deepfool_iter", 50, "maximum number of deepfool iterations (default: 25)")
flags.DEFINE_float("deepfool_clip", 0.5, "perturbation clip during search (default: 0.1)")
flags.DEFINE_float("deepfool_overshoot", 0.02, "multiplier for final perturbation")
flags.DEFINE_integer("summary_frequency", 10, "summarize frequency")

FLAGS = flags.FLAGS
logger = logging.getLogger()


def setup_experiment():
    if not os.path.exists(FLAGS.load_dir) or not os.path.isdir(FLAGS.load_dir):
        raise ValueError("Could not find folder %s" % FLAGS.load_dir)

    train_params = load_training_params(FLAGS.load_dir)
    FLAGS.model = train_params['model']
    FLAGS.layer_dims = train_params['layer_dims']
    FLAGS.working_dir = os.path.join(FLAGS.working_dir, os.path.basename(os.path.normpath(FLAGS.load_dir)))
    FLAGS.samples_dir = os.path.join(FLAGS.working_dir, 'samples')
    if os.path.exists(FLAGS.working_dir):
        shutil.rmtree(FLAGS.working_dir)
    os.makedirs(FLAGS.working_dir)
    os.makedirs(FLAGS.samples_dir)

    # configure logging
    cmd_hndl = logging.StreamHandler()
    cmd_hndl.setLevel(logging.INFO)
    logger.addHandler(cmd_hndl)
    file_hndl = logging.FileHandler(os.path.join(FLAGS.working_dir, 'log.txt'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    logger.setLevel(logging.DEBUG)

    # print config
    logger.info(FLAGS.__dict__['__flags'])


def main():
    setup_experiment()

    data = mnist_load()
    X_test = data.X_test
    y_test = data.y_test
    if FLAGS.sort_labels:
        ys_indices = np.argsort(y_test)
        X_test = X_test[ys_indices]
        y_test = y_test[ys_indices]

    img_shape = [None, 1, 28, 28]
    test_images = T.tensor4('test_images')
    test_labels = T.lvector('test_labels')

    # loaded discriminator number of classes and dims
    layer_dims = [int(dim) for dim in FLAGS.layer_dims.split("-")]
    num_classes = layer_dims[-1]

    # create and load discriminator
    net = create_network(FLAGS.model, img_shape, layer_dims=layer_dims)
    load_network(net, epoch=FLAGS.load_epoch)
    model = with_end_points(net)

    test_outputs = model(test_images, deterministic=True)
    # deepfool images
    test_df_images = deepfool(
        lambda x: model(x, deterministic=True)['logits'],
        test_images, test_labels, num_classes, max_iter=FLAGS.deepfool_iter,
        clip_dist=FLAGS.deepfool_clip, over_shoot=FLAGS.deepfool_overshoot)
    test_df_images_all = deepfool(
        lambda x: model(x, deterministic=True)['logits'],
        test_images, num_classes=num_classes, max_iter=FLAGS.deepfool_iter,
        clip_dist=FLAGS.deepfool_clip, over_shoot=FLAGS.deepfool_overshoot)
    test_df_outputs = model(test_df_images, deterministic=True)
    # fast gradient sign images
    test_fgsm_images = test_images + fast_gradient_perturbation(
        test_images, test_outputs['logits'], test_labels, FLAGS.fgsm_epsilon)
    test_at_outputs = model(test_fgsm_images, deterministic=True)

    # test metrics
    test_acc = categorical_accuracy(test_outputs['logits'], test_labels).mean()
    test_err = 1 - test_acc
    test_fgsm_acc = categorical_accuracy(test_at_outputs['logits'], test_labels).mean()
    test_fgsm_err = 1 - test_fgsm_acc
    test_df_acc = categorical_accuracy(test_df_outputs['logits'], test_labels).mean()
    test_df_err = 1 - test_df_acc

    # adversarial noise statistics
    reduc_ind = range(1, test_images.ndim)
    test_l2_df = T.sqrt(
        T.sum((test_df_images - test_images) ** 2, axis=reduc_ind))
    test_l2_df_norm = test_l2_df / T.sqrt(
        T.sum(test_images ** 2, axis=reduc_ind))
    test_l2_df_skip = test_l2_df.sum() / T.sum(test_l2_df > 0)
    test_l2_df_skip_norm = test_l2_df_norm.sum() / T.sum(test_l2_df_norm > 0)
    test_l2_df_all = T.sqrt(
        T.sum((test_df_images_all - test_images) ** 2, axis=reduc_ind))
    test_l2_df_all_norm = test_l2_df_all / T.sqrt(
        T.sum(test_images ** 2, axis=reduc_ind))

    test_metrics = OrderedDict([('err', test_err),
                                ('err_fgsm', test_fgsm_err),
                                ('err_df', test_df_err),
                                ('l2_df', test_l2_df.mean()),
                                ('l2_df_norm', test_l2_df_norm.mean()),
                                ('l2_df_skip', test_l2_df_skip),
                                ('l2_df_skip_norm', test_l2_df_skip_norm),
                                ('l2_df_all', test_l2_df_all.mean()),
                                ('l2_df_all_norm', test_l2_df_all_norm.mean())])
    logger.info("Compiling theano functions...")
    test_fn = theano.function([test_images, test_labels],
                              outputs=test_metrics.values())
    generate_fn = theano.function([test_images, test_labels],
                                  [test_df_images, test_df_images_all],
                                  on_unused_input='ignore')

    logger.info("Generate samples...")
    samples_per_class = 10
    summary_images, summary_labels = select_balanced_subset(
        X_test, y_test, num_classes, samples_per_class)
    save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
    save_images(summary_images, save_path)
    df_images, df_images_all = generate_fn(summary_images, summary_labels)
    save_path = os.path.join(FLAGS.samples_dir, 'deepfool.png')
    save_images(df_images, save_path)
    save_path = os.path.join(FLAGS.samples_dir, 'deepfool_all.png')
    save_images(df_images_all, save_path)

    logger.info("Starting...")
    test_iterator = batch_iterator(
        X_test, y_test, FLAGS.batch_size, shuffle=False)
    test_results = np.zeros(len(test_fn.outputs))
    start_time = time.time()
    for batch_index, (images, labels) in enumerate(test_iterator, 1):
        batch_results = test_fn(images, labels)
        test_results += batch_results
        if batch_index % FLAGS.summary_frequency == 0:
            df_images, df_images_all = generate_fn(images, labels)
            save_path = os.path.join(FLAGS.samples_dir, 'b%d-df.png' % batch_index)
            save_images(df_images, save_path)
            save_path = os.path.join(FLAGS.samples_dir, 'b%d-df_all.png' % batch_index)
            save_images(df_images_all, save_path)
            logger.info(build_result_str("Batch [{}] adversarial statistics:".format(batch_index),
                                         test_metrics.keys(), batch_results))
    test_results /= batch_index
    logger.info(
        build_result_str("Test results [{:.2f}s]:".format(time.time() - start_time),
                         test_metrics.keys(), test_results))


if __name__ == "__main__":
    main()
