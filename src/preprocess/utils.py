import random

from typing import Tuple

from ..models.utils import get_tokenizer


def dict_to_full_tree(tree: dict) -> Tuple[dict, str]:
    """Converts a dict of form
    {
        'id' : str,
        'title': str,
        'selftext' : str,
        'author' : str,
        'comments' : {'id' : {'id': str,
                              'parent_id': str,
                              'replies': List[str],
                              'author': str,
                              'body': str,},
                     'id1' : ...
                     ...}
    }
    to the form of:
    {
        'id': {'id' :  str,
               'parent_id':  str,
               'replies': List[str],
               'author': str,
               'body': str,},
        'id1': ...
        ...
    } by adding an entry for the OP's post in tree['comments'].

    Returns:
        A dict of the specified and the key of the root element of the tree,
        i.e. the OP's post.
    """
    tree["comments"][tree["id"]] = {}

    tree["comments"][tree["id"]]["id"] = tree["id"]
    tree["comments"][tree["id"]]["parent_id"] = ""
    tree["comments"][tree["id"]]["author"] = tree["author"]
    tree["comments"][tree["id"]]["body"] = ("<title> " + tree["title"] +
                                            " </title> " + tree["selftext"])
    tree["comments"][tree["id"]]["replies"] = [
        id for id, comment in tree["comments"].items()
        if comment["parent_id"] == tree["id"]
    ]
    return tree["comments"], tree["id"]


def size_subtrees(tree: dict,
                  root: str,
                  tokenizer=get_tokenizer(),
                  extra_tokens: int = 1):
    """
    Args:
        tree:         The tree in full_tree format of dict_to_full_tree.
        root:         The node whose sub-trees sizes are to be measured up.
        tokenizer:    Tokenizer that will be used to tokenize the sentences in the tree;
                      must implement encode() functionality.
        extra_tokens: Expected number of extra tokens that will be added to the 'body'
                      of each comment.[e.g., user tags or post tags]
    Returns:
        Modified tree, with extra 'subtree_size' attribute at each node(n) that denotes the
        length of tokenizing the combined version of all the 'body' attributes of (all the
        nodes in the subtree(n) and the node (n) itself).
    """
    def core_recursion(tree: dict, root: str):
        self_length = len(tokenizer.encode(tree[root]["body"])) + extra_tokens

        for reply_id in tree[root]["replies"]:
            self_length += (
                core_recursion(tree, reply_id) - 2
            )  # -2 for <s> </s> tokens, they are already counted in initial val of self_length

        tree[root]["subtree_size"] = self_length
        return self_length

    entire_tree_length = core_recursion(tree, root)

    return tree, root


def make_text_from_trees(tree: dict,
                         root: str,
                         preorder: bool = True,
                         shuffle: bool = False) -> str:
    """
    Makes text sequence from subtree in tree rooted at root.
    Acts on trees in format of the tree output by size_subtrees().
    Adds user tags around posts.
    """
    post_nos = {id: str(j) for j, id in enumerate(tree.keys())}

    def core_recursion(tree: dict, root: str, authors: dict) -> str:
        """
        Returns the (body of the root with user & post tags), if no replies added to the root;
        otherwise returns (body of root with user & post tags) & bodies of all posts in the subtree
        with user & post tags, in preorder if preorder is True in enclosing function, else in postorder.
        """
        if tree[root]["author"] not in authors:
            authors[tree[root]["author"]] = str(len(authors))

        author, post_id, parent_id = (
            tree[root]["author"],
            tree[root]["id"],
            tree[root]["parent_id"],
        )
        user_no = authors[author]

        node_text = ("<post" + post_nos[post_id] + " parent_id= " +
                     (str(None) if parent_id == "" else post_nos[parent_id]) +
                     "> " + "<user" + user_no + "> " +
                     tree[root]["body"].strip() + " </user" + user_no + ">" +
                     " </post" + post_nos[post_id] + ">")

        if tree[root]["replies"] == []:
            return node_text

        subtree_text = node_text + " " if preorder else ""

        for i, reply_id in enumerate(tree[root]["replies"]):
            subtree_text += (" " if i != 0 else "") + core_recursion(
                tree, reply_id, authors)

        if not preorder:
            subtree_text += " " + node_text

        return subtree_text

    def shuffle_text(text: str) -> str:
        """
        Shuffles around the post entries in text.
        """
        text = text.strip() + " "
        posts = text.split("<post")[1:]
        random.shuffle(posts)
        shuffled_text = "<post".join([""] + posts)
        return shuffled_text.strip()

    text = core_recursion(tree, root, {})

    if shuffle:
        text = shuffle_text(text)

    return text


def propagate_difference(tree: dict, root_node: str, start_node: str,
                         diff: int):
    """Subtracts diff from parent(p) of start_node, parent(pp) of p, parent of pp..
    upto the root node. Additionally deletes the subtree rooted at start_node
    from tree.Additionally, removes start_node from 'replies' of its parent.
    """
    if tree[start_node]["parent_id"] == "":
        for k in list(tree.keys()):
            tree.pop(k)
        return tree

    tree[tree[start_node]["parent_id"]]["replies"].remove(start_node)

    init_start_node = start_node

    while True:
        start_node = tree[start_node]["parent_id"]
        tree[start_node]["subtree_size"] -= diff
        if start_node == root_node:
            break

    def delete_subtree(tree: dict, start_node: str):
        for reply_id in tree[start_node]["replies"]:
            delete_subtree(tree, reply_id)
        tree.pop(start_node)

    delete_subtree(tree, init_start_node)

    return tree


def subtrees_lis(
    tree: dict,
    root_node: str,
    threshold: int = 4096 + 5,
    preorder: bool = True,
    shuffle: bool = True,
) -> List[str]:
    """
    Args:
        tree:       tree, with 'subtree_size' attribute at each node; as output by size_subtrees() function.
        root_node:  id of the root node of the tree.
        threshold:  max. permitted subtree_size for a subtree.
        preorder:   whether to arrange texts from a subtree in preorder or postorder, by default it is preorder.
        shuffle:    whether to shuffle the posts arranged in pre-order/post-order.
    Returns:
        The list of all subtrees made from the tree. A single subtree is an independent unit for
        the model. Subtrees may consist of single posts too. They are useful in allowing the model
        to learn to recognize when the entire component is a single post.
    """
    subtrees = []

    def core_recursion(tree: dict, root: str) -> int:

        if tree[root]["subtree_size"] < threshold or tree[root][
                "replies"] == []:
            subtrees.append(make_text_from_trees(tree, root, preorder,
                                                 shuffle))
            tree = propagate_difference(tree, root_node, root,
                                        tree[root]["subtree_size"])
        else:
            core_recursion(tree, tree[root]["replies"][0])

    while len(tree) != 0:
        core_recursion(tree, root_node)

    return subtrees
