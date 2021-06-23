from typing import List, Tuple
from multiprocessing import Pool
from functools import reduce

import jax.numpy as jnp

from ..params import config


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
        set_prediction = set_prediction.add(elem)

    set_reference = set()
    for elem in reference:
        set_reference = set_reference.add(elem)

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


class relation_match_metric:
    def __init__(self, n_processes=None):
        super().__init__(self)
        self.common_relations = 0
        self.total_relations = 0
        if n_processes is None:
            self.n_processes = config["batch_size"] // 2

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
                lambda x, y, x_idx, y_idx: relational_metric(
                    x[:x_idx], y[:y_idx]),
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


def batch_to_post_tags(
        references: jnp.ndarray,
        predictions: jnp.ndarray) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Converts the post tags predicted by the model, and those output by the
    dataset iterator to actual labels like ['B-.', 'I-.'].
    Args:
        references:     Reference labels batch output by dataset iterator. Must have pad labels
                        at padded positions. Expected size: [batch_size, max_len]
        predictions:    Predictions by the model for a batch of input sequences. These don't need to have
                        pad labels at padded positions. Expected size: [batch_size, max_len]
    Returns:
        Lists of references and predictions converted to string tags for each sample sequence in the batch.
    """

    seq_lens = jnp.sum(references != config["pad_for"]["post_tags"],
                       axis=-1).tolist()

    def convert_ids_to_tags(lis, idx):
        return [
            "B-P" if config["post_tags"]["B"] == lis[i] else "I-P"
            for i in range(0, idx)
        ]

    with Pool(sum(seq_lens) // 10000 + 1) as p:
        predictions_lis = p.starmap(convert_ids_to_tags,
                                    zip(predictions.tolist(), seq_lens))
        references_lis = p.starmap(convert_ids_to_tags,
                                   zip(references.tolist(), seq_lens))

    return references_lis, predictions_lis
