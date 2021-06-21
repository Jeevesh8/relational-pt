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
    2,
    "post_tags": {
        "B": 0,
        "I": 1
    },
    "user_tags": [{
        "B": i,
        "I": i + 1
    } for i in range(1, stable_config["max_users"] * 2, 2)],
}
