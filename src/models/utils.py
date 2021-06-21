from transformers import BigBirdTokenizer
import jax.numpy as jnp

from ..globals import stable_config

sp_tokens = ["[URL]", "[STARTQ]", "[ENDQ]", "[UNU]"] + [
    "[USER" + str(i) + "]" for i in range(stable_config["max_users"])
]


def get_tokenizer():
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
    tokenizer.add_tokens(sp_tokens)
    return tokenizer


def add_garbage_dims(array):
    """Adds extra slice at last of every dimension, filled with zeros."""
    for i in range(len(array.shape)):
        array = jnp.concatenate(
            [
                array,
                jnp.expand_dims(
                    jnp.zeros(array.shape[:i] + array.shape[i + 1:],
                              dtype=array.dtype),
                    axis=i,
                ),
            ],
            axis=i,
        )
    return array


def remove_garbage_dims(array):
    """Removes extra slice at last of every dimension, filled with zeros,
    which were added by add_garbage_dims()"""
    for i in range(len(array.shape)):
        array = jnp.take(array, jnp.arange(array.shape[i] - 1), axis=i)
    return array
