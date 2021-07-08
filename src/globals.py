from flax.core.frozen_dict import freeze

stable_config = {
    "max_len": 4096,
    "max_users": 128,
    "max_comps": 128,  # Max. in dataset=103
    "num_devices": 8,
    "checkpoint": "google/bigbird-roberta-base",
    "embed_dim": 768,
}

stable_config = freeze(stable_config)
