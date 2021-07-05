import os, glob

from ..globals import stable_config
from flax.core.frozen_dict import freeze

ft_config = {
    "batch_size": 1,
    "pt_wts_file": None,
    "cmv_modes_dir": None,
    "n_epochs": 10,
    "train_test_split": {
        "train_sz": 80,
        "test_sz": 20
    },
}

n_samples = 0
for filename in glob.glob(os.path.join(ft_config["cmv_modes_dir"], "*/*")):
    if filename.endswith(".xml"):
        n_samples += 1

ft_config["opt"] = {
    "lr":
    0.0001,
    "max_grad_norm":
    1.0,  # Gradients clipped at this norm. Use "None" for no clipping
    "total_steps":
    2 * n_samples * ft_config["train_test_split"]["train_sz"] //
    (ft_config["batch_size"] * stable_config["num_devices"]*100),
    "restart_from":
    0,
    "use_schedule":
    True,
    "weight_decay":
    None,  # Use "None" for no weight decay; adamw will be used if it is not None.
}

ft_config = freeze(ft_config)
