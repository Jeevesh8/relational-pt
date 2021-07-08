import os, shlex, glob
import random
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from functools import partial

import tensorflow as tf
from bs4 import BeautifulSoup

from .tokenize_components import get_model_inputs
from ..params import config
from ..globals import stable_config
from .configs import config as data_config
from ..arg_mining_ft.params import ft_config

cmv_modes_data = namedtuple(
    "cmv_modes_data",
    [
        "filenames", "tokenized_threads", "comp_type_labels",
        "refers_to_and_type"
    ],
)

if data_config["omit_filenames"]:
    cmv_modes_data = namedtuple(
        "cmv_modes_data",
        ["tokenized_threads", "comp_type_labels", "refers_to_and_type"],
    )


def convert_to_named_tuple(filenames, tokenized_threads, comp_type_labels,
                           refers_to_and_type, omit_filenames):
    if omit_filenames:
        return cmv_modes_data(tokenized_threads, comp_type_labels,
                              refers_to_and_type)
    return cmv_modes_data(filenames, tokenized_threads, comp_type_labels,
                          refers_to_and_type)


convert_to_named_tuple = partial(convert_to_named_tuple,
                                 omit_filenames=data_config["omit_filenames"])


def data_generator(file_list: List[str]):
    for elem in get_model_inputs(file_list):
        yield elem


def _create_min_max_boundaries(max_length: int,
                               min_boundary: int =256,
                               boundary_scale: float =1.1) -> Tuple[List[int], List[int]]:
    """Forms buckets. All samples with sequence length between the boundaries of a single bucket 
    are to be padded to the same length. 
    Args:
        max_length:     The max_length of any tokenized sequence that will be input to model.
        min_boundary:   The largest sequence length that can be accomodated in the first bucket.
        boundary_scale: The factor by which previous bucket's max length should be multiplied to get
                        the max length for next bucket.
    Returns:
        2 Lists, ``buckets_min`` and ``buckets_max`` the i-th bucket corresponds to sequence lengths 
        from buckets_min[i-1] to buckets_max[i-1].
    """
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset: tf.data.Dataset, batch_size: int, max_length: int, min_boundary: Optional[int]=256) -> tf.data.Dataset:
    """Dynamically batches the samples in dataset. Buckets of similar sample length samples are 
    made. Batches are made from elements in a bucket and each sequence is padded to the max length of
    sample in the batch.
    Args:
        dataset:      A tensorflow dataset to be dynamically batched.
        batch_size:   Number of tokens expected in a single batch. That is the expected sum of 
                      all tokens in all samples in a batch
        max_length:   Max. sequence length any sample from dataset. max_length <= batch_size.
        min_boundary: The max. length of any sequence in the 1-st bucket.
    Returns:
        A tensorflow dataset that returns dynamically batched and padded sample.
    """
    def get_sample_len(*args):
        return tf.shape(args[1])[0]
    
    if batch_size<max_length:
        raise ValueError("The expected number of tokens in a single batch(batch_size) must be \
                         >= max_length of any sequence in dataset. Got batch_size, max_length as:",
                         (batch_size, max_length))
    actual_max_len = 0
    for elem in dataset:
        sample_len = get_sample_len(*elem)
        if actual_max_len<sample_len:
            actual_max_len = sample_len
    
    if actual_max_len > max_length:
        raise AssertionError("Found sequence with longer length than specified: ", max_length,
                             "in dataset. Actual max length:", actual_max_len)
    
    buckets_min, buckets_max = _create_min_max_boundaries(max_length, min_boundary=min_boundary)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(*args) -> int:
        """Returns the index of bucket that the input specified in *args
        falls in."""
        seq_length = get_sample_len(*args)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id: int) -> int:
        """Returns the size of batch for the bucket at index bucket_id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id: int, grouped_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Batches a subset of a dataset, provided in grouped_dataset. Each element is padded 
        to the max. sequence length in the batch. Batches in same bucket may be padded to different 
        lengths.
        """
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size,
                                            padded_shapes=([],[None],[None],[None, 3]),
                                            padding_values=(None, *tuple(data_config["pad_for"].values())))
    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def get_dataset(file_list: List[str]):
    def callable_gen():
        nonlocal file_list
        for elem in data_generator(file_list):
            yield elem

    sample_wise_dataset =  tf.data.Dataset.from_generator(callable_gen,
                                                          output_signature=(
                                                            tf.TensorSpec(shape=(), dtype=tf.string, name="filenames"),
                                                            tf.TensorSpec(shape=(None),
                                                                        dtype=tf.int32,
                                                                        name="tokenized_threads"),
                                                            tf.TensorSpec(shape=(None),
                                                                        dtype=tf.int32,
                                                                        name="comp_type_labels"),
                                                            tf.TensorSpec(shape=(None, 3),
                                                                        dtype=tf.int32,
                                                                        name="refers_to_and_type"),
                                                            ),
                                                          )
    
    dataset = _batch_examples(sample_wise_dataset, 8194, 4096)
    dataset = dataset.map(convert_to_named_tuple)
    
    return dataset

def get_op_wise_split(filelist: List[str]) -> Dict[str, List[str]]:
    """Splits up filelist into groups, each of which corresponds to threads with
    the same original posts.
    """
    splits = {}
    for filepath in filelist:
        if not filepath.endswith(".xml"):
            continue
        with open(filepath) as f:
            xml_string = f.read()
        parsed_xml = BeautifulSoup(xml_string, "lxml")
        op_id = parsed_xml.thread["id"]
        if op_id not in splits:
            splits[op_id] = []
        splits[op_id].append(filepath)
    return splits


def load_dataset(
    cmv_modes_dir: str = None,
    train_sz: float = 100,
    valid_sz: float = 0,
    test_sz: float = 0,
    shuffle: bool = False,
    as_numpy_iter: bool = True,
):
    """Returns a tuple of train, valid, test datasets(the ones having non-zero size only)
    Args:
        cmv_modes_dir:  The directory to the version of cmv modes data from which the dataset is to be loaded.
                        If None, the data is downloaded into current working directory and v2.0 is used from there.
        train_sz:       The % of total threads to include in train data. By default, all the threads are included in train_data.
        valid_sz:       The % of total threads to include in validation data.
        test_sz:        The % of total threads to include in testing data.
        shuffle:        If True, the data is shuffled before splitting into train, test and valid sets.
        as_numpy_iter:  Tensorflow dataset is converted to numpy iterator, before returning.
    Returns:
        Tuple of 3 tensorflow datasets, corresponding to train, valid and test data. None is returned for the datasets
        for which size of 0 was specified.
    """
    if train_sz + valid_sz + test_sz != 100:
        raise ValueError("Train, test valid sizes must sum to 100")

    if cmv_modes_dir is None:
        os.system(
            shlex.quote(
                "git clone https://github.com/chridey/change-my-view-modes"))
        cmv_modes_dir = os.path.join(os.getcwd(), "change-my-view-modes/v2.0/")

    splits = get_op_wise_split(glob.glob(os.path.join(cmv_modes_dir, "*/*")))

    op_wise_splits_lis = list(splits.values())
    if shuffle:
        random.shuffle(op_wise_splits_lis)

    n_threads = sum([len(threads) for threads in splits.values()])
    num_threads_added = 0
    train_files, valid_files, test_files, i = [], [], [], 0

    if train_sz != 0:
        while (num_threads_added /
               n_threads) * 100 < train_sz and i < len(op_wise_splits_lis):
            train_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1
        num_threads_added = 0

    if valid_sz != 0:
        while (num_threads_added /
               n_threads) * 100 < valid_sz and i < len(op_wise_splits_lis):
            valid_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1
        num_threads_added = 0

    if test_sz != 0:
        while (num_threads_added /
               n_threads) * 100 <= test_sz and i < len(op_wise_splits_lis):
            test_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1

    train_dataset = None if len(train_files) == 0 else get_dataset(train_files)
    valid_dataset = None if len(valid_files) == 0 else get_dataset(valid_files)
    test_dataset = None if len(test_files) == 0 else get_dataset(test_files)

    if as_numpy_iter:
        train_dataset = (None if train_dataset is None else
                         train_dataset.as_numpy_iterator())
        valid_dataset = (None if valid_dataset is None else
                         valid_dataset.as_numpy_iterator())
        test_dataset = (None if test_dataset is None else
                        test_dataset.as_numpy_iterator())

    return train_dataset, valid_dataset, test_dataset
