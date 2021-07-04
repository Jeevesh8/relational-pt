from typing import Optional, Dict, List
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from .relational_model import relational_model
from .tree_crf import tree_crf
from .linear_crf import crf_layer
from src.globals import stable_config


def comp_prediction_loss(logits, lengths, label_tags):
    return crf_layer(n_classes=2)(hk.Linear(2)(logits), lengths, label_tags)


def relation_prediction_loss(embds, choice_mask, label_relations, max_comps,
                             embed_dim):
    model1 = relational_model(n_rels=1,
                              max_comps=max_comps,
                              embed_dim=embed_dim)
    log_energies = model1(embds, choice_mask)
    return tree_crf().disc_loss(log_energies, label_relations)


relation_prediction_loss = partial(
    relation_prediction_loss,
    max_comps=stable_config["max_comps"],
    embed_dim=stable_config["embed_dim"],
)


def predict_components(logits, lengths):
    return crf_layer(n_classes=2).batch_viterbi_decode(
        hk.Linear(2)(logits), lengths)[0]


def predict_relations(embds, choice_mask, max_comps, embed_dim):
    model1 = relational_model(n_rels=1,
                              max_comps=max_comps,
                              embed_dim=embed_dim)
    log_energies = model1(embds, choice_mask)
    return tree_crf().mst(log_energies)[1]


predict_relations = partial(
    predict_relations,
    max_comps=stable_config["max_comps"],
    embed_dim=stable_config["embed_dim"],
)

pure_cpl = hk.transform(comp_prediction_loss)
pure_rpl = hk.transform(relation_prediction_loss)

pure_pc = hk.transform(predict_components)
pure_pr = hk.transform(predict_relations)

##################################  FINETUNING  #####################################
from ..cmv_modes.configs import config as ft_config

_n_classes = len(ft_config["arg_components"])
_n_rels = len(ft_config["relations_map"])


def ft_comp_prediction_loss(logits, lengths, label_tags):
    return crf_layer(n_classes=_n_classes)(hk.Linear(_n_classes)(logits),
                                           lengths, label_tags)


def ft_relation_prediction_loss(embds, choice_mask, label_relations, max_comps,
                                embed_dim):
    model1 = relational_model(n_rels=_n_rels,
                              max_comps=max_comps,
                              embed_dim=embed_dim)
    log_energies = model1(embds, choice_mask)
    return tree_crf().disc_loss(log_energies, label_relations)


ft_relation_prediction_loss = partial(
    ft_relation_prediction_loss,
    max_comps=stable_config["max_comps"],
    embed_dim=stable_config["embed_dim"],
)


def ft_predict_components(logits, lengths):
    return crf_layer(n_classes=_n_classes).batch_viterbi_decode(
        hk.Linear(_n_classes)(logits), lengths)[0]


def ft_predict_relations(embds, choice_mask, max_comps, embed_dim):
    model1 = relational_model(n_rels=_n_rels,
                              max_comps=max_comps,
                              embed_dim=embed_dim)
    log_energies = model1(embds, choice_mask)
    return tree_crf().mst(log_energies)[1]


ft_predict_relations = partial(
    ft_predict_relations,
    max_comps=stable_config["max_comps"],
    embed_dim=stable_config["embed_dim"],
)

ft_pure_cpl = hk.transform(ft_comp_prediction_loss)
ft_pure_rpl = hk.transform(ft_relation_prediction_loss)

ft_pure_pc = hk.transform(ft_predict_components)
ft_pure_pr = hk.transform(ft_predict_relations)

def copy_weights(
    old_mat: jnp.ndarray,
    new_mat: jnp.ndarray,
    mapping: Optional[Dict[int, List[int]]] = None,
) -> jnp.ndarray:
    """
    Args:
        old_mat:   A matrix of shape [old_input_dim, old_output_dim] extracted from some pretrained model.
        new_mat:   A matrix of shape [old_input_dim, new_output_dim] extracted from randomly initialized weights of some model.
        mapping:   A dict mapping i to the list of all j such that new_mat[:, i] is to be assigned the mean{old_mat[:, j] over all j}.
                   By default, for i not specified in the mapping, mean will be taken over all the j.
    Returns:
        A matrix of same shape as new_mat, with weights copied from old_mat as specified in mapping.

    NOTE: This function combines with None indexing and jnp.squeeze to copy 1-D vectors too.
    """
    
    if mapping is None:
        mapping = {}
    
    one_dimensional = False
    if jnp.size(old_mat.shape) == jnp.size(new_mat.shape) == 1:
        one_dimensional = True
        old_mat, new_mat = old_mat[None, :], new_mat[None, :]

    old_input_dim, old_output_dim = old_mat.shape
    new_input_dim, new_output_dim = new_mat.shape

    if old_input_dim != new_input_dim:
        raise ValueError(
            "The layer's between which weights are being copied are expected to have same input dimensions. Received shapes: "
            + str(old_mat.shape) + " and " + str(new_mat.shape))

    _mapping = {i: list(range(old_output_dim)) for i in range(new_output_dim)}
    _mapping.update(mapping)

    for i, mean_over in _mapping.items():
        new_mat = jax.ops.index_update(
            new_mat,
            [list(range(new_input_dim)), i],
            jnp.mean(jnp.take_along_axis(old_mat,
                                         jnp.array([mean_over]),
                                         axis=-1),
                     axis=-1),
        )

    if one_dimensional:
        new_mat = jnp.squeeze(new_mat)

    return new_mat
