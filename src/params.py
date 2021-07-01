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
    "weight_decay": None,  # Use "None" for no weight decay; adamw will be used if it is not None.
}

if stable_config['arg-mining-finetuning']:
    tokenizer = get_tokenizer()

    config.update({
        'relations' : ['partial_attack', 'agreement',
                        'attack', 'rebuttal_attack',
                        'understand', 'undercutter',
                        'undercutter_attack', 'disagreement',
                        'rebuttal', 'support', 'None',
                        'partial_agreement', 'partial_disagreement'],

        'relations_map': {"None" : ["None"],
                        "support": ['agreement', 'understand', 'support', 'partial_agreement'],
                        "against": ["partial_attack", 'attack', 'rebuttal_attack', 'undercutter', 'undercutter_attack', 'disagreement', 'rebuttal', 'partial_disagreement']},

        'arg_components' : {'other': 0, 'B-C' : 1, 'I-C' : 2, 'B-P' : 3, 'I-P' : 4},
    })

    config['pad_for'] = {
        'tokenized_thread' : tokenizer.pad_token_id,
        'comp_type_labels' : config['arg_components']['other'], #len(config['arg_components']),
        'refers_to_and_type' : 0, #len(config['dist_to_label'])+2,
        'attention_mask' : 0,
        'global_attention_mask' : 0,
    }