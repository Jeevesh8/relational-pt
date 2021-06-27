from .globals import stable_config
from .models.utils import get_tokenizer

config = {
    "pad_for": {
        "input_ids": get_tokenizer().pad_token_id,
        "post_tags": 500,
        "user_tags": 500,
        "relations": 0,
    },
  
    "batch_size":
    1,
    "post_tags": {
        "B": 0,
        "I": 1
    },
    "user_tags": [{
        "B": i,
        "I": i + 1
    } for i in range(1, stable_config["max_users"] * 2, 2)],
    "n_epochs":
    5,
    "train_files": ["../subtrees-text-4096/train.txt"],
    "valid_files": ["../subtrees-text-4096/valid.txt"],
    "save_model_file":
    "../relational_pretrained.wts",
}

n_samples = 0
for filename in config["train_files"]:
    with open(filename) as f:
        for elem in f.read().split("\n"):
            if not elem.startswith("-" * 14):
                n_samples += 1

config["opt"] = {
    "lr": 0.0001,
    "max_grad_norm": 1.0,  # Gradients clipped at this norm. Use "None" for no clipping
    "total_steps": 2
    * n_samples
    // (config["batch_size"] * stable_config["num_devices"]),
    "restart_from": 0,
    "use_schedule": True,
    "weight_decay": None,  # Use "None" for no weight decay, adamw will be used in this case.
}
