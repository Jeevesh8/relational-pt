from functools import partial
from typing import Callable

import jax
import flax
from flax.training import train_state

import jax.numpy as jnp
from jax.random import PRNGKey
from datasets import load_metric

from src.cmv_modes import load_dataset
from src.cmv_modes.configs import config as data_config
from src.models import ft_pure_cpl, ft_pure_rpl, ft_pure_pc, ft_pure_pr
from src.utils import get_hf_model, get_tokenizer
from src.arg_mining_ft.utils import get_params_dict
from src.arg_mining_ft.params import ft_config
from src.training.optimizer import get_adam_opt
from src.training.utils import load_relational_metric, batch_to_post_tags

from src.globals import stable_config

print("Devices detected: ", jax.local_devices())


class TrainState(train_state.TrainState):
    comp_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    relation_prediction_loss: Callable = flax.struct.field(pytree_node=False)
    comp_predictor: Callable = flax.struct.field(pytree_node=False)
    relation_predictor: Callable = flax.struct.field(pytree_node=False)
    config: dict = flax.struct.field(pytree_node=False)


@partial(jax.pmap, axis_name="device_axis", donate_argnums=(0, 1, 2))
def train_step(state, batch, key):
    attention_mask = (batch.tokenized_threads !=
                      state.config["pad_for"]["tokenized_thread"])
    lengths = jnp.sum(attention_mask, axis=-1)

    def comp_prediction_loss(params, key):
        key, subkey = jax.random.split(key)
        logits = state.apply_fn(
            batch.tokenized_threads,
            attention_mask,
            params=params["embds_params"],
            dropout_rng=subkey,
            train=True,
        )["last_hidden_state"]

        logits = jax.nn.normalize(logits)

        return state.comp_prediction_loss(params["comp_predictor"], key,
                                          logits, lengths,
                                          batch.comp_type_labels)

    grad_fn = jax.value_and_grad(comp_prediction_loss)
    key, subkey = jax.random.split(key)
    _comp_prediction_loss, grad = grad_fn(state.params, subkey)
    grad = jax.lax.pmean(grad, axis_name="device_axis")
    new_state = state.apply_gradients(grads=grad)

    def relation_prediction_loss(params, key):
        key, subkey = jax.random.split(key)
        embds = state.apply_fn(
            batch.tokenized_threads,
            attention_mask,
            batch.comp_type_labels,
            params=params["embds_params"],
            dropout_rng=subkey,
            train=True,
        )["last_hidden_state"]

        embds = jax.nn.normalize(embds)

        return state.relation_prediction_loss(
            params["relation_predictor"],
            key,
            embds,
            jnp.logical_or(
                batch.comp_type_labels == state.config["arg_components"]
                ["B-C"],
                batch.comp_type_labels == state.config["arg_components"]
                ["B-P"],
            ),
            batch.refers_to_and_type,
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
    attention_mask = (batch.tokenized_threads !=
                      state.config["pad_for"]["tokenized_thread"])
    lengths = jnp.sum(attention_mask, axis=-1)

    logits = state.apply_fn(
        batch.tokenized_threads,
        attention_mask,
        params=state.params["embds_params"],
        train=False,
    )["last_hidden_state"]

    logits = jax.nn.normalize(logits)

    comp_preds = state.comp_predictor(state.params["comp_predictor"],
                                      jax.random.PRNGKey(42), logits, lengths)
    return comp_preds, lengths


@jax.pmap
def get_rel_preds(state, batch):

    attention_mask = (batch.tokenized_threads !=
                      state.config["pad_for"]["tokenized_thread"])

    embds = state.apply_fn(
        batch.tokenized_threads,
        attention_mask,
        batch.comp_type_labels,
        params=state.params["embds_params"],
        train=False,
    )["last_hidden_state"]

    embds = jax.nn.normalize(embds)

    rel_preds = state.relation_predictor(
        state.params["relation_predictor"],
        jax.random.PRNGKey(42),
        embds,
        jnp.logical_or(
            batch.comp_type_labels == state.config["arg_components"]["B-C"],
            batch.comp_type_labels == state.config["arg_components"]["B-P"],
        ),
    )
    return rel_preds


def get_preds(state, batch):
    return get_comp_preds(state, batch), get_rel_preds(state, batch)


comp_prediction_metric = load_metric("seqeval")
rel_prediction_metric = load_relational_metric()


def eval_step(state, batch):
    (comp_preds, lengths), rel_preds = get_preds(state, batch)
    comp_preds = jnp.reshape(comp_preds, (-1, jnp.shape(comp_preds)[-1]))
    lengths = jnp.reshape(lengths, (-1))
    rel_preds = jnp.reshape(rel_preds, (-1, jnp.shape(rel_preds)[-2], 3))
    post_tags = jnp.reshape(batch.comp_type_labels,
                            (-1, batch.comp_type_labels.shape[-1]))
    relations = jnp.reshape(batch.refers_to_and_type,
                            (-1, batch.refers_to_and_type.shape[-2], 3))

    references, predictions = batch_to_post_tags(
        post_tags,
        comp_preds,
        tags_dict=state.config["arg_components"],
        seq_lens=lengths,
    )

    comp_prediction_metric.add_batch(predictions=predictions,
                                     references=references)

    rel_prediction_metric.add_batch(rel_preds, relations)


if __name__ == "__main__":

    key = PRNGKey(42)

    tokenizer = get_tokenizer()

    transformer_model = get_hf_model(len(tokenizer),
                                     token_types=len(
                                         data_config["arg_components"]))

    key, subkey = jax.random.split(key)
    params = get_params_dict(subkey,
                             transformer_model,
                             ft_config["pt_wts_file"],
                             use_pt_for_heads=False)

    opt = get_adam_opt()

    init_train_state = TrainState.create(
        apply_fn=transformer_model.__call__,
        params=params,
        tx=opt,
        comp_prediction_loss=ft_pure_cpl.apply,
        relation_prediction_loss=ft_pure_rpl.apply,
        comp_predictor=ft_pure_pc.apply,
        relation_predictor=ft_pure_pr.apply,
        config=data_config,
    )

    key = jax.random.split(key, stable_config["num_devices"])

    train_dataset, _, test_dataset = load_dataset(
        ft_config["cmv_modes_dir"], **ft_config["train_test_split"])

    num_iters = 0
    loop_state = flax.jax_utils.replicate(init_train_state)
    del init_train_state

    losses = {
        "comp_pred_loss": jnp.array([0.0]),
        "rel_pred_loss": jnp.array([0.0])
    }

    log_loss_n_iters = 20

    for epoch in range(ft_config["n_epochs"]):

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

        for v_batch in test_dataset:
            eval_step(loop_state, v_batch)

        print(
            f"Component Prediction metrics after epoch {epoch} :",
            comp_prediction_metric.compute(),
        )
        print(
            f"Relation Prediction metrics after epoch {epoch} :",
            rel_prediction_metric.compute(),
        )

        train_dataset, _, test_dataset = load_dataset(
            ft_config["cmv_modes_dir"], train_sz=80, test_sz=20)
