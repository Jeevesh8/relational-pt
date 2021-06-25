from functools import partial
from typing import Callable

from src.models.utils import get_tokenizer

import jax
import optax
import flax
import haiku as hk
from flax.training import train_state

import jax.numpy as jnp
from jax.random import PRNGKey
from transformers import FlaxBigBirdModel
from datasets import load_metric

from src.globals import stable_config
from src.params import config
from src.models import crf_layer, tree_crf, relational_model
from src.training.utils import load_relational_metric, batch_to_post_tags
from src.dataloaders.text_file_loader import get_tfds_dataset

import jax.tools.colab_tpu

jax.tools.colab_tpu.setup_tpu()

print("Devices detected: ", jax.local_devices())


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


class TrainState(train_state.TrainState):
    comp_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    relation_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    comp_predictor: Callable = flax.struct.field(pytree_node=False)
    relation_predictor: Callable = flax.struct.field(pytree_node=False)
    config: dict = flax.struct.field(pytree_node=False)


def train_step(state, batch, key):

    attention_mask = batch.input_ids != state.config["pad_for"]["input_ids"]
    lengths = jnp.sum(attention_mask, axis=-1)

    def comp_prediction_loss(params, key):
        key, subkey = jax.random.split(key)
        logits = state.apply_fn(
            batch.input_ids,
            attention_mask,
            params=params["embds_params"],
            dropout_rng=subkey,
            train=True,
        )["last_hidden_state"]
        return state.comp_prediction_loss(params["comp_predictor"], key,
                                          logits, lengths, batch.post_tags)

    grad_fn = jax.value_and_grad(comp_prediction_loss)
    key, subkey = jax.random.split(key)
    _comp_prediction_loss, grad = grad_fn(state.params, subkey)
    grad = jax.lax.pmean(grad, axis_name="device_axis")
    new_state = state.apply_gradients(grads=grad)

    def relation_prediction_loss(params, key):
        key, subkey = jax.random.split(key)
        embds = state.apply_fn(
            batch.input_ids,
            attention_mask,
            batch.post_tags,
            params=params["embds_params"],
            dropout_rng=subkey,
            train=True,
        )["last_hidden_state"]
        return state.relation_prediction_loss(
            params["relation_predictor"],
            key,
            embds,
            batch.post_tags == state.config["post_tags"]["B"],
            batch.relations,
        )

    grad_fn = jax.value_and_grad(relation_prediction_loss)
    key, subkey = jax.random.split(key)
    _relation_prediction_loss, grad = grad_fn(new_state.params, subkey)
    grad = jax.lax.pmean(grad, axis_name="device_axis")
    new_new_state = new_state.apply_gradients(grads=grad)

    losses = {
        "comp_pred_loss":
        jax.lax.pmean(_comp_prediction_loss, axis_name="device_axis"),
        "rel_pred_loss":
        jax.lax.pmean(_relation_prediction_loss, axis_name="device_axis"),
    }

    return new_new_state, losses, key


@jax.pmap
def get_preds(state, batch, key):

    attention_mask = batch.input_ids != state.config["pad_for"]["input_ids"]
    lengths = jnp.sum(attention_mask, axis=-1)

    logits = state.apply_fn(
        batch.input_ids,
        attention_mask,
        params=params["embds_params"],
        dropout_rng=key,
        train=False,
    )["last_hidden_state"]

    comp_preds = state.comp_predictor(params["comp_predictor"], key, logits,
                                      lengths)

    embds = state.apply_fn(
        batch.input_ids,
        attention_mask,
        batch.post_tags,
        params=params["embds_params"],
        dropout_rng=key,
        train=False,
    )["last_hidden_state"]

    rel_preds = state.relation_predictor(
        params["relation_predictor"],
        key,
        embds,
        batch.post_tags == state.config["post_tags"]["B"],
    )

    return comp_preds, rel_preds


comp_prediction_metric = load_metric("seqeval")
rel_prediction_metric = load_relational_metric()


def eval_step(state, batch, key):
    comp_preds, rel_preds = get_preds(state, batch, key)
    comp_preds = jnp.reshape(comp_preds, (-1, jnp.shape(comp_preds)[-1]))
    rel_preds = jnp.reshape(rel_preds, (-1, jnp.shape(rel_preds)[-2], 3))
    references, predictions = batch_to_post_tags(batch.post_tags, comp_preds)
    comp_prediction_metric.add_batch(predictions, references)
    rel_prediction_metric.add_batch(rel_preds, batch.relations)


if __name__ == "__main__":

    key = PRNGKey(42)

    transformer_model = FlaxBigBirdModel.from_pretrained(
        stable_config["checkpoint"])
    tokenizer = get_tokenizer()

    pure_cpl = hk.transform(comp_prediction_loss)
    pure_rpl = hk.transform(relation_prediction_loss)

    pure_pc = hk.transform(predict_components)
    pure_pr = hk.transform(predict_relations)

    params = {}

    sample_logits = jnp.zeros(
        (config["batch_size"], stable_config["max_len"],
         stable_config["embed_dim"]),
        dtype=jnp.float32,
    )
    sample_lengths = jnp.full((config["batch_size"]),
                              stable_config["max_len"],
                              dtype=jnp.int32)
    sample_comp_labels = jax.random.randint(
        key, (config["batch_size"], stable_config["max_len"]), 0, 2)

    sample_relations = jax.random.randint(
        key,
        (config["batch_size"], stable_config["max_comps"], 3),
        0,
        stable_config["max_comps"],
    )
    sample_relations = jnp.where(jnp.array([True, True, False]),
                                 sample_relations, 0)

    key, subkey = jax.random.split(key)
    params["comp_predictor"] = pure_cpl.init(subkey, sample_logits,
                                             sample_lengths,
                                             sample_comp_labels)

    key, subkey = jax.random.split(key)
    params["relation_predictor"] = pure_rpl.init(subkey, sample_logits,
                                                 sample_comp_labels == 0,
                                                 sample_relations)

    params["embds_params"] = transformer_model.params

    opt = optax.chain(optax.adam(learning_rate=config["lr"]))

    init_train_state = TrainState.create(
        apply_fn=transformer_model.__call__,
        params=params,
        tx=opt,
        comp_prediction_loss=pure_cpl.apply,
        relation_prediction_loss=pure_rpl.apply,
        comp_predictor=pure_pc.apply,
        relation_predictor=pure_pr.apply,
        config=config,
    )

    key = jax.random.split(key, stable_config["num_devices"])

    parallel_train_step = jax.pmap(train_step, axis_name="device_axis")

    train_dataset = get_tfds_dataset(config["train_files"], config)
    val_dataset = get_tfds_dataset(config["valid_files"], config)

    num_iters = 0
    loop_state = flax.jax_utils.replicate(init_train_state)

    losses = {
        "comp_pred_loss": jnp.array([0.0]),
        "rel_pred_loss": jnp.array([0.0])
    }

    log_loss_n_iters = 1000
    validation_n_iters = 10000

    for epoch in range(config["n_epochs"]):
        for batch in train_dataset:
            loop_state, step_losses, key = parallel_train_step(
                loop_state, batch, key)
            losses = jax.tree_multimap(lambda x, y: x + y, losses, step_losses)
            num_iters += 1

            if num_iters % log_loss_n_iters == 0:
                print(
                    "component prediction loss for iterations",
                    num_iters - log_loss_n_iters,
                    "to",
                    num_iters,
                    ":",
                    losses["comp_pred_loss"] / log_loss_n_iters,
                )
                print(
                    "relation prediction loss for iterations",
                    num_iters - log_loss_n_iters,
                    "to",
                    num_iters,
                    ":",
                    losses["rel_pred_loss"] / log_loss_n_iters,
                )
                losses["comp_pred_loss"] = 0.0
                losses["rel_pred_loss"] = 0.0

            if num_iters % validation_n_iters == 0:
                for batch in val_dataset:
                    key, subkey = jax.random.split(key)
                    eval_step(loop_state, batch, subkey)
                print(
                    "Component Prediction metrics for iterations",
                    num_iters,
                    "to",
                    num_iters - validation_n_iters,
                    ":",
                    comp_prediction_metric.compute(),
                )
                print(
                    "Relation Prediction metrics for iterations",
                    num_iters,
                    "to",
                    num_iters - validation_n_iters,
                    ":",
                    rel_prediction_metric.compute(),
                )
