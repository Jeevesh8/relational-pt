import argparse

from transformers import FlaxAutoModel

from src.models.utils import get_tokenizer
from src.training.utils import load_model_wts

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wts_file", type=str, required=True, help="The serialized weights file.")
    parser.add_argument("--checkpoint", type=str, default="google/bigbird-roberta-base", help="The HF transformer model checkpoint used for pre-training wts stored in the wts file.")
    parser.add_argument("--hf_auth_token", type=str, required=True, help="Your HF authorization token. E.G. api_DaYgaaVnGdRtznIgiNfotCHFUqmOdARmPx")
    parser.add_argument("--model_id", type=str, required=True, help="Id of model that will appear at HF hub.")
    args = parser.parse_args()
    
    transformer_model = FlaxAutoModel(args.checkpoint)
    tokenizer = get_tokenizer()
    transformer_model = transformer_model.resize_token_embeddings(len(tokenizer))

    params = load_model_wts(args.wts_file, transformer_model)
    transformer_model.params = params["embds_params"]

    transformer_model.push_to_hub(args.model_id, use_auth_token=args.hf_auth_token)
    tokenizer.push_to_hum(args.model_id, use_auth_token=args.hf_auth_token)