from typing import List, Tuple
from functools import reduce
from collections import namedtuple

from ..models.utils import get_tokenizer
from ..params import config
from ..globals import stable_config


def dict_to_inputs(
    tree: List[dict],
    tokenizer=get_tokenizer(),
    get_mat: bool = False,
    one_indexed: bool = True,
) -> Tuple[List[int], List[int], List[int], List[List[int]]]:
    """
    Takes a list of posts in a subtree. Encodes individual posts, concatenates
    the encodings, in the order they appear in tree. Assumes post_id and parent_id
    of each are equal to location(in tree) of the post being referred to. If one_indexed
    is True, posts are numbered from 1 and a relation to 0-th position means that post is root.

    Returns:
        input_ids:       The input ids of all the posts in the tree concatenated together.
        post_tags:       The post tag of each input_id. Either int[B-post] or int[I-post].
        user_tags:       The user tag of each input_id. Either int[B-user<i>] or int[I-user<i>].

        If get_mat is True:
        relations:      len(tree)Xlen(tree) sized matrix where the (i,j)-th entry is
                         1 if the parent of i is j, 0 otherwise.
        else:
        relations:      A list of tuples of form (child_post_location, parent_post_location).
    """
    encodings = [tokenizer.encode(post["body"])[1:-1] for post in tree]
    encodings[0] = [tokenizer.bos_token_id] + encodings[0]
    encodings[-1] = encodings[-1] + [tokenizer.eos_token_id]

    idxing = 1 if one_indexed else 0

    post_tags = []
    user_tags = []

    if get_mat:
        relations = [[0 for __ in len(tree) + idxing]
                     for _ in len(tree) + idxing]
    else:
        relations = []

    for post, post_encoding in zip(tree, encodings):
        post_tags += [config["post_tags"]["B"]
                      ] + [config["post_tags"]["I"]] * (len(post_encoding) - 1)
        # print(post['user_no'], len(config['user_tags']))
        user_tags += [config["user_tags"][post["user_no"]]["B"]
                      ] + [config["user_tags"][post["user_no"]]["I"]
                           ] * (len(post_encoding) - 1)

        if post["parent_id"] is not None:
            if get_mat:
                relations[post["post_id"] + idxing][post["parent_id"] +
                                                    idxing] = 1
            else:
                relations.append(
                    (post["post_id"] + idxing, post["parent_id"] + idxing))

        elif one_indexed:

            if get_mat:
                relations[post["post_id"] + 1][0] = 1
            else:
                relations.append((post["post_id"] + 1, 0))

    input_ids = reduce(lambda x, y: x + y, encodings, [])

    # It is valid to truncate like this becuase the only subtrees whose encodings are greater than config['max_len'] are ones with just single posts,
    # (see preprocess.py for how subtrees are generated). One may worry that some component present in relations may get truncated from inpu text,
    # But this is not the case as the relations only contains (1, 0, 0).
    return (
        input_ids[:stable_config["max_len"]],
        post_tags[:stable_config["max_len"]],
        user_tags[:stable_config["max_len"]],
        relations,
    )


def only_inside_links(tree: List[dict]):
    """
    Takes a list of dictionaries corresponding to posts in a subtree of the form:
    {
        'post_id' : int,
        'parent_id' : int,
        'user_no' : int,
        'body': str,
    }
    and removes the link from the root post to the remaining tree.
    """
    post_ids = [post["post_id"] for post in tree]
    outside_links = 0
    for post in tree:
        if post["parent_id"] not in post_ids:
            outside_links += 1
            if outside_links > 1:
                raise AssertionError(
                    "More than one link to posts outside the subtree exists, in the subtree: "
                    + str(tree))
            post["parent_id"] = None
    return tree


import numpy as np


def tree_ids_to_nos(tree: List[dict], convert_users: bool = True):
    """
    Converts post_id to to the location of the post in the tree list.
    Correspondingly changes parent_id of each post to match changes in post_id.
    If parent_id not found in posts, the parent is marked as None.
    user_nos are changed to be in [0, distinct users in tree-1] if convert_users is True
    """
    post_ids = [post["post_id"] for post in tree]
    users = np.unique(np.array([post["user_no"] for post in tree])).tolist()

    for i, post in enumerate(tree):
        post["post_id"] = i
        try:
            post["parent_id"] = post_ids.index(post["parent_id"])
        except ValueError:
            post["parent_id"] = None

        if convert_users:
            post["user_no"] = users.index(post["user_no"])

    return tree


modelInp = namedtuple("modelInp",
                      ["input_ids", "post_tags", "user_tags", "relations"])


def convert_to_named_tuple(input_ids, post_tags, user_tags, relations):
    return modelInp(input_ids, post_tags, user_tags, relations)
