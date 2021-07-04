import os, shlex
import argparse

from src.utils import get_hf_model, get_tokenizer
from src.training.utils import load_model_wts

import jax
import jax.numpy as jnp
from flax import serialization

jax.config.update("jax_platform_name", "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wts_file",
                        type=str,
                        required=True,
                        help="The serialized weights file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="google/bigbird-roberta-base",
        help=
        "The HF transformer model checkpoint used for pre-training wts stored in the wts file.",
    )
    parser.add_argument(
        "--hf_auth_token",
        type=str,
        required=True,
        help=
        "Your HF authorization token. E.G. api_DaYgaaVnGdRtznIgiNfotCHFUqmOdARmPx",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Id of model that will appear at HF hub.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer()

    transformer_model = get_hf_model(len(tokenizer))

    params = load_model_wts(transformer_model, args.wts_file, False)
    params = jax.tree_util.tree_map(jnp.squeeze, params)

    transformer_model.params = params["embds_params"]

    transformer_model.push_to_hub(args.model_id,
                                  use_auth_token=args.hf_auth_token)
    tokenizer.push_to_hub(args.model_id, use_auth_token=args.hf_auth_token)

    params.pop("embds_params")

    with open(os.path.join(args.model_id, "top_head.params"), "wb+") as f:
        f.write(serialization.to_bytes(params))

    os.chdir(args.model_id)
    os.system(shlex.quote("git add ."))
    os.system(shlex.quote('git commit -m "Added additional top head params".'))
    os.system(shlex.quote("git push"))
