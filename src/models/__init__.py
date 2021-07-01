from functools import partial

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
