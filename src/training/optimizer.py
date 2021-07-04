import jax
import optax
import numpy as np

from flax import traverse_util

from ..params import config

if "opt" not in config:
    from ..arg_mining_ft.params import ft_config as config

def make_lr_schedule(warmup_percentage, total_steps, restart_from=0):
    def lr_schedule(step):
        percent_complete = (step + restart_from) / total_steps

        # 0 or 1 based on whether we are before peak
        before_peak = jax.lax.convert_element_type(
            (percent_complete <= warmup_percentage), np.float32)
        # Factor for scaling learning rate
        scale = (before_peak * (percent_complete / warmup_percentage) +
                 (1 - before_peak)) * (1 - percent_complete)

        return scale

    return lr_schedule


def decay_mask_fn(params):
    """To prevent decay of weights when using adamw, for the bias parameters;
    FROM: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb"""
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale"))
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)


def get_adam_opt():
    total_steps = config["opt"]["total_steps"] * config["n_epochs"]

    lr_schedule = make_lr_schedule(
        warmup_percentage=0.1,
        total_steps=total_steps,
        restart_from=config["opt"]["restart_from"],
    )

    opt = optax.chain(
        optax.identity() if config["opt"]["max_grad_norm"] is None else
        optax.clip_by_global_norm(config["opt"]["max_grad_norm"]),
        optax.adam(learning_rate=config["opt"]["lr"])
        if config["opt"]["weight_decay"] is None else optax.adamw(
            learning_rate=config["opt"]["lr"],
            weight_decay=config["opt"]["weight_decay"],
            mask=decay_mask_fn,
        ),
        optax.scale_by_schedule(lr_schedule)
        if config["opt"]["use_schedule"] else optax.identity(),
    )

    return opt
