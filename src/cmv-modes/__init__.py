import tensorflow as tf

from .tokenize_components import get_model_inputs
from ..params import config


def generator(file_list):
    for elem in get_model_inputs(file_list):
        yield elem


def get_dataset(file_list):
    def callable_gen():
        nonlocal file_list
        for elem in generator(file_list):
            yield elem

    return (tf.data.Dataset.from_generator(
        callable_gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string, name="filename"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="tokenized_thread"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="comp_type_labels"),
            tf.TensorSpec(shape=(None, 3),
                          dtype=tf.int32,
                          name="refers_to_and_type"),
            tf.TensorSpec(shape=(None), dtype=tf.int32, name="attention_mask"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="global_attention_mask"),
        ),
    ).padded_batch(
        config["batch_size"],
        padded_shapes=([], [None], [None], [None, None], [None], [None]),
        padding_values=(None, *tuple(config["pad_for"].values())),
    ).cache())
