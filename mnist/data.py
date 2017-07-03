import gzip
import logging
import os
import sys

import numpy as np

from utils import AttributeDict

logger = logging.getLogger("data")


def mnist_load(train_size=50000, dseed=1):
    # borrowed from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    rng = np.random.RandomState(dseed)
    randix = rng.permutation(X_train.shape[0])
    X_train, X_val = X_train[randix[:train_size]], X_train[randix[train_size:]]
    y_train, y_val = y_train[randix[:train_size]], y_train[randix[train_size:]]

    logger.debug('%d examples in training dataset' % X_train.shape[0])
    logger.debug('%d examples in validation dataset' % X_val.shape[0])
    logger.debug('%d examples in testing dataset' % X_test.shape[0])

    return AttributeDict({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    })


def maybe_download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    filepath = os.path.join('./data', filename)
    if not os.path.exists(filepath):
        logger.debug("Downloading %s" % filename)
        urlretrieve(source + filename, filepath)


def load_mnist_images(filename):
    maybe_download(filename)
    # Read the inputs in Yann LeCun's binary format.
    filepath = os.path.join('./data', filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)


def load_mnist_labels(filename):
    maybe_download(filename)
    # Read the labels in Yann LeCun's binary format.
    filepath = os.path.join('./data', filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


def batch_iterator(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    n_samples = inputs.shape[0]
    if shuffle:
        # Shuffles indicies of training data, so we can draw batches
        # from random indicies instead of shuffling whole data
        indx = np.random.permutation(range(n_samples))
    else:
        indx = range(n_samples)
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = inputs[indx[sl]]
        y_batch = targets[indx[sl]]
        yield X_batch, y_batch


def example_iterator(inputs, targets, shuffle=False):
    assert len(inputs) == len(targets)
    n_samples = inputs.shape[0]
    if shuffle:
        indx = np.random.permutation(range(n_samples))
    else:
        indx = range(n_samples)
    for i in indx:
        yield inputs[i], targets[i]


# data utils
def select_balanced_subset(X, y, num_classes=10, samples_per_class=10, seed=1):
    from lasagne.utils import floatX
    total_samples = num_classes * samples_per_class
    X_subset = floatX(np.zeros([total_samples] + list(X.shape[1:])))
    y_subset = np.zeros((total_samples,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for i in range(num_classes):
        yi_indices = np.where(y == i)[0]
        rng.shuffle(yi_indices)
        X_subset[samples_per_class * i:(i + 1) * samples_per_class, ...] = X[yi_indices[:samples_per_class]]
        y_subset[samples_per_class * i:(i + 1) * samples_per_class] = i
    return X_subset, y_subset
