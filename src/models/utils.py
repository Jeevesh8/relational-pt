from typing import Tuple

import jax
import jax.numpy as jnp
from transformers import BigBirdTokenizer

from ..globals import stable_config

sp_tokens = ["[URL]", "[STARTQ]", "[ENDQ]", "[UNU]"] + [
    "[USER" + str(i) + "]" for i in range(stable_config["max_users"])
]


def get_tokenizer():
    tokenizer = BigBirdTokenizer.from_pretrained(stable_config["checkpoint"])
    tokenizer.add_tokens(sp_tokens)
    return tokenizer


def add_garbage_dims(array):
    """Adds extra slice at last of every dimension, filled with zeros."""
    """FOR-LOOP equivalent
    for i, _ in enumerate(array.shape):
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
    """
    return jnp.pad(array, pad_width=tuple((0, 1) for _ in jnp.shape(array)))


def remove_garbage_dims(array):
    """Removes extra slice at last of every dimension, filled with zeros,
    which were added by add_garbage_dims()"""
    for i, _ in enumerate(array.shape):
        array = jnp.take(array, jnp.arange(array.shape[i] - 1), axis=i)
    return array


def get_samples(
    batch_size: int,
    max_len: int,
    embed_dim: int,
    max_comps: int,
    n_token_types: int = 3,
    n_rel_types: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Args:
        batch_size:     The number of samples in a batch that would be passed to the model.
        max_len:        The maximum length of any tokenized sequence in batch
        embed_dim:      The size of embedding of any single token in the sequence.
        max_comps:      The maximum number of components(posts) in any sequence.
        n_token_types:  Number of types of token labels, for e.g. ("B-P", "I-P") -> 2
        n_rel_type:     Number of types of relations between any two components.
    Returns:
        A tuple having:
            sample_logits:      of shape [batch_size, max_len, embed_dim]
            sample_length:      of shape [batch_size]; each element < max_len
            sample_comp_labels: of shape [batch_size, max_len]; each element < n_token_types
            sample_relations:   of shape [batch_size, max_comps, 3]; (i,j,k)-th element denotes link
                                from component i to component j of type k. Not guaranteed to be a tree.
    """
    key = jax.random.PRNGKey(32)

    sample_logits = jnp.zeros(
        (batch_size, max_len, embed_dim),
        dtype=jnp.float32,
    )
    sample_lengths = jnp.full((batch_size), max_len, dtype=jnp.int32)

    sample_comp_labels = jax.random.randint(key, (batch_size, max_len), 0,
                                            n_token_types)

    sample_relation_links = jax.random.randint(
        key,
        (batch_size, max_comps, 3),
        0,
        max_comps,
    )

    sample_relation_types = jnp.random.randint(key, (batch_size, max_comps, 3),
                                               0, n_rel_types)

    sample_relations = jnp.where(jnp.array([True, True, False]),
                                 sample_relation_links, sample_relation_types)

    return sample_logits, sample_lengths, sample_comp_labels, sample_relations
