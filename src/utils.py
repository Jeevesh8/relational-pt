import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModel

from globals import stable_config


sp_tokens = ["[URL]", "[STARTQ]", "[ENDQ]", "[UNU]"] + [
    "[USER" + str(i) + "]" for i in range(stable_config["max_users"])
]


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(stable_config["checkpoint"])
    tokenizer.add_tokens(sp_tokens)
    return tokenizer


def get_hf_model(tokenizer_len: int,
                 token_types: int = 2,
                 key=jax.random.PRNGKey(65)):
    """Returns a HuggingFace model with extended word embeddings, for added special tokens
    added to vocabulary and any additional token types that you want to add. All the extra
    embeddings are added **after** the existing ones.
    """
    model = FlaxAutoModel.from_pretrained(stable_config["checkpoint"],
                                          vocab_size=tokenizer_len)

    original_embeds = model.params["embeddings"]["word_embeddings"][
        "embedding"]
    n_words, embed_dim = original_embeds.shape
    if tokenizer_len > n_words:
        key, subkey = jax.random.split(key)
        additional_embeds = jax.random.normal(
            subkey,
            shape=(tokenizer_len - n_words, embed_dim),
            dtype=original_embeds.dtype,
        )
        model.params["embeddings"]["word_embeddings"][
            "embedding"] = jnp.concatenate(
                [original_embeds, additional_embeds])

    original_embeds = model.params["embeddings"]["token_type_embeddings"][
        "embedding"]
    n_token_types, embed_dim = original_embeds.shape
    if token_types > n_token_types:
        key, subkey = jax.random.split(key)
        additional_embeds = jax.random.normal(
            subkey,
            shape=(token_types - n_token_types, embed_dim),
            dtype=original_embeds.dtype,
        )
        model.params["embeddings"]["token_type_embeddings"][
            "embedding"] = jnp.concatenate(
                [original_embeds, additional_embeds])

    return model
