from functools import partial
from typing import Callable

from flax import serialization

from src.models.utils import get_hf_model, get_tokenizer

import jax
import flax
import haiku as hk
from haiku.data_structures import to_mutable_dict
from flax.training import train_state

import jax.numpy as jnp
from jax.random import PRNGKey
from datasets import load_metric

from src.globals import stable_config
from src.params import config
from src.models import pure_cpl, pure_rpl, pure_pc, pure_pr
from src.training.utils import (
    load_relational_metric,
    batch_to_post_tags,
    get_params_dict,
)
from src.training.optimizer import get_adam_opt
from src.dataloaders.text_file_loader import get_tfds_dataset

# import jax.tools.colab_tpu

# jax.tools.colab_tpu.setup_tpu()

print("Devices detected: ", jax.local_devices())


class TrainState(train_state.TrainState):
    comp_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    relation_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    comp_predictor: Callable = flax.struct.field(pytree_node=False)
    relation_predictor: Callable = flax.struct.field(pytree_node=False)
    config: dict = flax.struct.field(pytree_node=False)


@partial(jax.pmap, axis_name="device_axis", donate_argnums=(0, 1, 2))
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
            batch.post_tags == config["post_tags"]["B"],
            batch.relations,
        )

    grad_fn = jax.value_and_grad(relation_prediction_loss)
    key, subkey = jax.random.split(key)
    _relation_prediction_loss, grad = grad_fn(new_state.params, subkey)
    grad = jax.lax.pmean(grad, axis_name="device_axis")
    new_new_state = new_state.apply_gradients(grads=grad)

    losses = {
        "comp_pred_loss": _comp_prediction_loss,
        "rel_pred_loss": _relation_prediction_loss,
    }
    return new_new_state, losses, key


@jax.pmap
def get_comp_preds(state, batch):
    attention_mask = batch.input_ids != state.config["pad_for"]["input_ids"]
    lengths = jnp.sum(attention_mask, axis=-1)

    logits = state.apply_fn(
        batch.input_ids,
        attention_mask,
        params=state.params["embds_params"],
        train=False,
    )["last_hidden_state"]

    comp_preds = state.comp_predictor(state.params["comp_predictor"],
                                      jax.random.PRNGKey(42), logits, lengths)
    return comp_preds


@jax.pmap
def get_rel_preds(state, batch):

    attention_mask = batch.input_ids != state.config["pad_for"]["input_ids"]

    embds = state.apply_fn(
        batch.input_ids,
        attention_mask,
        batch.post_tags,
        params=state.params["embds_params"],
        train=False,
    )["last_hidden_state"]

    rel_preds = state.relation_predictor(
        state.params["relation_predictor"],
        jax.random.PRNGKey(42),
        embds,
        batch.post_tags == state.config["post_tags"]["B"],
    )
    return rel_preds


def get_preds(state, batch):
    return get_comp_preds(state, batch), get_rel_preds(state, batch)


comp_prediction_metric = load_metric("seqeval")
rel_prediction_metric = load_relational_metric()


def eval_step(state, batch):
    comp_preds, rel_preds = get_preds(state, batch)
    comp_preds = jnp.reshape(comp_preds, (-1, jnp.shape(comp_preds)[-1]))
    rel_preds = jnp.reshape(rel_preds, (-1, jnp.shape(rel_preds)[-2], 3))
    post_tags = jnp.reshape(batch.post_tags, (-1, batch.post_tags.shape[-1]))
    relations = jnp.reshape(batch.relations,
                            (-1, batch.relations.shape[-2], 3))
    references, predictions = batch_to_post_tags(post_tags, comp_preds)
    comp_prediction_metric.add_batch(predictions=predictions,
                                     references=references)
    rel_prediction_metric.add_batch(rel_preds, relations)


if __name__ == "__main__":

    key = PRNGKey(42)

    tokenizer = get_tokenizer()

    transformer_model = get_hf_model(len(tokenizer))

    key, subkey = jax.random.split(key)
    params = get_params_dict(subkey, transformer_model)

    opt = get_adam_opt()

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

    train_dataset = get_tfds_dataset(config["train_files"], config)
    val_dataset = get_tfds_dataset(config["valid_files"], config)

    num_iters = 0
    loop_state = flax.jax_utils.replicate(init_train_state)
    del init_train_state

    losses = {
        "comp_pred_loss": jnp.array([0.0]),
        "rel_pred_loss": jnp.array([0.0])
    }

    log_loss_n_iters = 100
    validation_n_iters = 12000

    for epoch in range(config["n_epochs"]):

        for batch in train_dataset:
            loop_state, step_losses, key = train_step(loop_state, batch, key)
            step_losses = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0),
                                                 step_losses)
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
                for v_batch in val_dataset:
                    eval_step(loop_state, v_batch)

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

                val_dataset = get_tfds_dataset(config["valid_files"], config)

        train_dataset = get_tfds_dataset(config["train_files"], config)

        write_file = config["save_model_file"] + str(epoch)

        with open(write_file, "wb+") as f:

            to_write = {
                "embds_params":
                loop_state.params["embds_params"],
                "comp_predictor":
                to_mutable_dict(loop_state.params["comp_predictor"]),
                "relation_predictor":
                to_mutable_dict(loop_state.params["relation_predictor"]),
            }

            f.write(
                serialization.to_bytes(
                    jax.tree_util.tree_map(lambda x: jnp.take(x, [0], axis=0),
                                           to_write)))
            print("Another Train Epoch. WEIGHTS STORED AT:", write_file)
