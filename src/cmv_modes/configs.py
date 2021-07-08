from flax.core.frozen_dict import freeze

from ..params import tokenizer

config = {
    "relations": [
        "partial_attack",
        "agreement",
        "attack",
        "rebuttal_attack",
        "understand",
        "undercutter",
        "undercutter_attack",
        "disagreement",
        "rebuttal",
        "support",
        "None",
        "partial_agreement",
        "partial_disagreement",
    ],
    "relations_map": {
        "None": ["None"],
        "support": ["agreement", "understand", "support", "partial_agreement"],
        "against": [
            "partial_attack",
            "attack",
            "rebuttal_attack",
            "undercutter",
            "undercutter_attack",
            "disagreement",
            "rebuttal",
            "partial_disagreement",
        ],
    },
    "arg_components": {
        "other": 0,
        "B-C": 1,
        "I-C": 2,
        "B-P": 3,
        "I-P": 4
    },
    "omit_filenames":
    True,

    "max_len": 4096,
    "max_comps": 128,
}

config["pad_for"] = {
    "tokenized_thread": tokenizer.pad_token_id,
    "comp_type_labels":
    config["arg_components"]["other"],  # len(config['arg_components']),
    "refers_to_and_type": 0,  # len(config['dist_to_label'])+2,
}

config = freeze(config)
