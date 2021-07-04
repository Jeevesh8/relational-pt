from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from haiku.data_structures import to_mutable_dict, to_immutable_dict

from .params import ft_config
from ..cmv_modes.configs import config as data_config
from ..training.utils import load_model_wts
from ..models import ft_pure_cpl, ft_pure_rpl, copy_weights
from ..models import get_samples
from ..globals import stable_config


def get_transition_mat(
        pt_transition_mat: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Uses a pre-trained transition matrix(if provided) of form:
           B-P  I-P
    B-P [[  x ,  y ],
    I-P [[  z ,  w ]]
    (where 'P' stands for post) to initialize a new transition matrix for
    argument mining fintuning. The transition matrix defines transitions
    amongst ['B-C', 'I-C', 'B-P', 'I-P', 'O'].
    """
    if pt_transition_mat is None:
        pt_transition_mat = np.array([[1e-4, 1.0], [0.1, 0.9]])

    rng = np.random.default_rng(12345)

    random_transition_mat = (rng.uniform(size=(5, 5), ) + 0.05)
    ac_dict = data_config["arg_components"]

    random_transition_mat[[ac_dict["I-C"], ac_dict["other"]],
                          ac_dict["I-P"]] = -np.inf
    random_transition_mat[[ac_dict["I-P"], ac_dict["other"]],
                          ac_dict["I-C"]] = -np.inf
    random_transition_mat[ac_dict["other"],
                          [ac_dict["I-P"], ac_dict["I-C"]]] = -np.inf

    random_transition_mat[ac_dict["B-C"], ac_dict["I-P"]] = -np.inf
    random_transition_mat[ac_dict["B-P"], ac_dict["I-C"]] = -np.inf

    random_transition_mat[
        [ac_dict["B-C"], ac_dict["B-P"]],
        [ac_dict["B-C"], ac_dict["B-P"]]] = pt_transition_mat[0, 0]

    random_transition_mat[ac_dict["B-C"],
                          ac_dict["I-C"]] = pt_transition_mat[0, 1]
    random_transition_mat[ac_dict["B-P"],
                          ac_dict["I-P"]] = pt_transition_mat[0, 1]

    random_transition_mat[
        [ac_dict["I-P"], ac_dict["I-C"]],
        [ac_dict["B-P"], ac_dict["B-C"]]] = pt_transition_mat[1, 0]

    random_transition_mat[ac_dict["I-C"],
                          ac_dict["I-C"]] = pt_transition_mat[1, 1]
    random_transition_mat[ac_dict["I-P"],
                          ac_dict["I-P"]] = pt_transition_mat[1, 1]

    return jnp.array(random_transition_mat)


def get_params_dict(key: jnp.ndarray,
                    base_model,
                    pt_wts_file: str = None,
                    use_pt_for_heads: bool = True) -> Dict:
    """
    Args:
        key:                For random weight.
        base_model:         A HF model instance which is same as the one used pre-training.
        pt_wts_file:        File having pre-trained weights.
        use_pt_for_heads:   A boolean indicating whether to copy weights for the relation/component prediction heads
                            from the pretrained model.
    Returns:
        params for the base model with a component detection head and a relation prediction head.
        Has same keys structure as the params returned by src.training.utils.get_params_dict()
    """
    ft_params = {}

    pt_params = load_model_wts(base_model, pt_wts_file)

    sample_logits, sample_lengths, sample_comp_labels, sample_relations = get_samples(
        ft_config["batch_size"],
        stable_config["max_len"],
        stable_config["embed_dim"],
        stable_config["max_comps"],
        len(data_config["arg_components"]),
        len(data_config["relations_map"]),
    )

    key, subkey = jax.random.split(key)
    ft_params["comp_predictor"] = ft_pure_cpl.init(subkey, sample_logits,
                                                   sample_lengths,
                                                   sample_comp_labels)
    ft_params["comp_predictor"] = to_mutable_dict(ft_params["comp_predictor"])

    key, subkey = jax.random.split(key)
    ft_params["relation_predictor"] = ft_pure_rpl.init(subkey, sample_logits,
                                                       sample_comp_labels == 0,
                                                       sample_relations)

    if use_pt_for_heads:
        ft_params["comp_predictor"]["linear"] = jax.tree_util.tree_multimap(
            copy_weights,
            (
                pt_params["comp_predictor"]["linear"],
                ft_params["comp_predictor"]["linear"],
            ),
        )
        ft_params["comp_predictor"]["transition_matrix"] = get_transition_mat(
            pt_params["comp_predictor"]["transition_matrix"])

        ft_params["relation_predictor"] = jax.tree_util.tree_multimap(
            copy_weights,
            (pt_params["relation_predictor"], ft_params["relation_predictor"]),
        )

    else:
        ft_params["comp_predictor"]["transition_matrix"] = get_transition_mat()

    ft_params["comp_predictor"] = to_immutable_dict(
        ft_params["comp_predictor"])
    ft_params["embds_params"] = pt_params["embds_params"]

    return ft_params
