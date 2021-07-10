from typing import List, Optional, Tuple, Dict
from multiprocessing import Pool
from functools import reduce

import jax
import jax.numpy as jnp
from flax import serialization
from haiku.data_structures import to_mutable_dict, to_immutable_dict

from ..globals import stable_config
from ..params import config
from ..models import pure_cpl, pure_rpl
from ..models import get_samples


def relational_metric(prediction: List[List[int]],
                      reference: List[List[int]]) -> Tuple[int, int]:
    """
    Args:
        prediction:  A list of lists of form (i,j,k) where the tuple indicates a link from i to j of type k.
        referenece:   A list of the same type. Both the lists correspond to a single subtree.
    Returns:
        The number of common elements in the predictions and reference; and the total number of unique elements in prediction
        and reference combined.
    """
    set_prediction = set()
    for elem in prediction:
        set_prediction.add(tuple(elem))

    set_reference = set()
    for elem in reference:
        set_reference.add(tuple(elem))

    print("Reference relations:", set_reference)
    print("Predicted relations:", set_prediction)

    return len(set_prediction.intersection(set_reference)), len(
        set_prediction.union(set_reference))


def batch_to_relational_lists(predictions: jnp.ndarray,
                              references: jnp.ndarray):
    """Converts padded relations batches predicted by model, and output by dataset iterator to
    lists of relations[list of tuples of form (link_from, link_to, rel_type)] in each sample.
    Args:
        predictions:    A batch of predictions of relations size [batch_size, max_comps, 3]
        references:     A batch of labels of the corresponding elements of predictions.
    """
    batchwise_num_pred_rels = jnp.sum(
        jnp.sum(predictions == config["pad_for"]["relations"], axis=-1) != 3,
        axis=-1)
    batchwise_num_ref_rels = jnp.sum(
        jnp.sum(references == config["pad_for"]["relations"], axis=-1) != 3,
        axis=-1)

    return (
        predictions.tolist(),
        references.tolist(),
        batchwise_num_pred_rels.tolist(),
        batchwise_num_ref_rels.tolist(),
    )


def calc_relation_metric(x, y, x_idx, y_idx):
    return relational_metric(x[:x_idx], y[:y_idx])


class relation_match_metric:
    def __init__(self, n_processes=None):
        super().__init__()
        self.common_relations = 0
        self.total_relations = 0
        if n_processes is None:
            self.n_processes = max(1, config["batch_size"] // 2)

    def add_batch(self, predictions, references):
        """Computes the relation match metric for the given batch of data.
        Args:
            In format specified in batch_to_relational_lists()
        Returns:
            None
        """
        (
            preds_list,
            refs_list,
            preds_idx_list,
            refs_idx_list,
        ) = batch_to_relational_lists(predictions, references)

        with Pool(self.n_processes) as p:
            samplewise_metrics = p.starmap(
                calc_relation_metric,
                zip(preds_list, refs_list, preds_idx_list, refs_idx_list),
            )

        common_rels, total_rels = reduce(
            lambda tup1, tup2: (tup1[0] + tup2[0], tup1[1] + tup2[1]),
            samplewise_metrics,
            (0, 0),
        )
        self.common_relations += common_rels
        self.total_relations += total_rels

    def compute(self):
        """Returns a dictionary containing the various values
        corresponding to the metric.
        """
        if self.total_relations == 0:
            raise AssertionError(
                "No sample with relations to compute metric over was added. Use metric.add_batch(preds, refs) to add batches first."
            )

        result = {
            "intersection": self.common_relations,
            "union": self.total_relations,
            "iou": self.common_relations / self.total_relations,
        }

        self.total_relations = 0
        self.common_relations = 0

        return result


def load_relational_metric(n_processes=None):
    return relation_match_metric(n_processes)


def convert_ids_to_tags(lis,
                        idx,
                        tags_dict: Dict[str, int] = config["post_tags"]):
    int_to_tags = {v: k for k, v in tags_dict.items()}
    return [int_to_tags[lis[i]] for i in range(0, idx)]


def batch_to_post_tags(
    references: jnp.ndarray,
    predictions: jnp.ndarray,
    tags_dict: Dict[str, int] = config["post_tags"],
    seq_lens: Optional[jnp.ndarray] = None,
    pad_id: int = config["pad_for"]["post_tags"],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Converts the post tags predicted by the model, and those output by the
    dataset iterator to actual labels like ['B-.', 'I-.'].
    Args:
        references:     Reference labels batch output by dataset iterator. Must have pad labels
                        at padded positions. Expected size: [batch_size, max_len].
        predictions:    Predictions by the model for a batch of input sequences. These don't need to have
                        pad labels at padded positions. Expected size: [batch_size, max_len].
        tags_dict:      A dictionaary from tag to the integer label it corresponds to. E.G. {"B-P" : 0, "I-P" : 1}.
        seq_lens:       The lengths of various sequences in the batch that references/predictions correspond to.
        pad_id:         The id of pad token in reference / relation labels. Only used in case seq_lens is not provided.
    Returns:
        Lists of references and predictions converted to string tags for each sample sequence in the batch.
    """
    if seq_lens is None:
        seq_lens = jnp.reshape(jnp.sum(references != pad_id, axis=-1), (-1))
    seq_lens = seq_lens.tolist()

    tags_dict = [tags_dict] * len(seq_lens)

    with Pool(sum(seq_lens) // 10000 + 1) as p:
        predictions_lis = p.starmap(
            convert_ids_to_tags, zip(predictions.tolist(), seq_lens,
                                     tags_dict))
        references_lis = p.starmap(
            convert_ids_to_tags, zip(references.tolist(), seq_lens, tags_dict))

    return references_lis, predictions_lis


def get_params_dict(key, base_model, all_dicts: bool = False) -> dict:
    """Constructs a single dictionary compoesed of parameters of the transformer_model,
    together with the component prediction and relation prediction head's parameters.
    Args:
        key:                Random key for initializing parameters of relation prediction and
                            component prediction heads
        base_model:         A HF Flax transformer model, or any other embedding model initialized with pre-trained weights.
        all_dicts:          bool to indicate whether we want all parameters to be stored in nested
                            python dicts or the haiku ones to be in FlatMap only.
    Returns:
        parmas dict with the following structure:
        {
            "embds_params" : base_model.params,
            "relation_predictor" : hk._src.data_structures.FlatMap having parameters,
            "comp_predictor" : hk._src.data_structures.FlatMap having parameters,
        }
        If all_dicts is True, then last two are converted to Python dictionaries.
    """
    params = {}

    sample_logits, sample_lengths, sample_comp_labels, sample_relations = get_samples(
        config["batch_size"],
        stable_config["max_len"],
        stable_config["embed_dim"],
        stable_config["max_comps"],
    )

    key, subkey = jax.random.split(key)
    params["comp_predictor"] = pure_cpl.init(subkey, sample_logits,
                                             sample_lengths,
                                             sample_comp_labels)

    key, subkey = jax.random.split(key)
    params["relation_predictor"] = pure_rpl.init(
        subkey,
        sample_logits,
        sample_comp_labels == 0,
        jnp.zeros((config["batch_size"], stable_config["max_len"],
                   stable_config["max_len"])),
        sample_relations,
    )

    params["embds_params"] = base_model.params

    if all_dicts:
        params["comp_predictor"] = to_mutable_dict(params["comp_predictor"])
        params["relation_predictor"] = to_mutable_dict(
            params["relation_predictor"])

    return params


def load_model_wts(base_model,
                   wts_file: Optional[str] = None,
                   to_hk_flat_map: bool = True) -> dict:
    """Loads wts from a binary file. Assumes wts are of the form output by src.training.utils.get_params_dict().
    Args:
        base_model:     The base HF model whose weights are stored in wts_file, as loaded from HF.
        wts_file:       The file havnig serialized bytes corresponding to the weights to be loaded. Random weights
                        are loaded(for heads) if this is None. Base model's weights are copied as it is, if wts_file is None.
        to_hk_flat_map: Whether to convert wts of haiku modules to hk._src.data_structures.FlatMap or not.
    Returns:
        The params dict with the same key value pairs as src.training.utils.get_params_dict()
    """
    params = get_params_dict(jax.random.PRNGKey(12),
                             base_model,
                             all_dicts=True)

    if wts_file is not None:
        with open(wts_file, "rb") as f:
            params = serialization.from_bytes(params, f.read())

    if to_hk_flat_map:
        params["comp_predictor"] = to_immutable_dict(params["comp_predictor"])
        params["relation_predictor"] = to_immutable_dict(
            params["relation_predictor"])

    return params
