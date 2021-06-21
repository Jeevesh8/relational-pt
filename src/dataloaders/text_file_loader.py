import re
import tensorflow as tf

from multiprocessing import Pool

from .utils import only_inside_links, tree_ids_to_nos, dict_to_inputs
from ..globals import stable_config


def get_all_trees(read_file):
    with open(read_file) as f:
        post_trees = [
            elem.strip() for elem in f.readlines()
            if not elem.startswith("-" * 14)
        ]

    comment_pattern = (
        r"<post(\d+) parent_id= (.*?)> <user(\d+)>(.+?)<\/user\d+> <\/post\d+>"
    )
    tree_with_post_parent_user_no = [
        re.findall(comment_pattern, post_tree) for post_tree in post_trees
    ]

    tree_with_post_parent_user_no = [[{
        "post_id":
        int(elem[0]),
        "parent_id":
        None if elem[1] == "None" else int(elem[1]),
        "user_no":
        int(elem[2]),
        "body": (" " if i != 0 else "") + elem[3].strip(),
    } for i, elem in enumerate(tree)]
                                     for tree in tree_with_post_parent_user_no]
    return tree_with_post_parent_user_no


def format_data(tree_with_post_parent_user_no):
    with Pool(5) as p:
        tree_with_post_parent_user_no = p.map(only_inside_links,
                                              tree_with_post_parent_user_no)
        tree_with_post_parent_user_no = p.map(tree_ids_to_nos,
                                              tree_with_post_parent_user_no)
        inputs_tags_n_rels = p.map(dict_to_inputs,
                                   tree_with_post_parent_user_no)
    return inputs_tags_n_rels


def get_generator(file_lis):
    def data_generator():
        nonlocal file_lis
        for file in file_lis:
            tree_with_post_parent_user_no = get_all_trees(file)
            inputs_tags_n_rels = format_data(tree_with_post_parent_user_no)
            for train_inp in inputs_tags_n_rels:
                yield train_inp

    return data_generator


def get_tfds_dataset(file_lis, config):
    dataset = tf.data.Dataset.from_generator(
        get_generator(file_lis),
        output_signature=(
            tf.TensorSpec(shape=(None), dtype=tf.int32, name="input_ids"),
            tf.TensorSpec(shape=(None), dtype=tf.int32, name="post_tags"),
            tf.TensorSpec(shape=(None), dtype=tf.int32, name="user_tags"),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32,
                          name="relations"),
        ),
    )

    dataset = dataset.padded_batch(
        config["batch_size"],
        padded_shapes=(
            [stable_config["max_len"]],
            [stable_config["max_len"]],
            [stable_config["max_len"]],
            [stable_config["max_comps"], 3],
        ),
        padding_values=(
            config["pad_for"]["input_ids"],
            config["pad_for"]["post_tags"],
            config["pad_for"]["user_tags"],
            config["pad_for"]["relations"],
        ),
    ).batch(stable_config["num_devices"])

    return dataset
