import os
import argparse

from multiprocessing import Pool, set_executable

from src.preprocess.json_dataloader import load_reddit_data
from src.globals import stable_config
from src.preprocess.utils import dict_to_full_tree, size_subtrees, subtrees_lis


def process_trees(trees_lis):
    with Pool(5) as p:
        trees_with_roots = p.map(dict_to_full_tree, trees_lis)
        del trees_lis
        sized_trees_with_roots = p.starmap(size_subtrees, trees_with_roots)
        del trees_with_roots
        _subtrees = p.starmap(subtrees_lis, sized_trees_with_roots)
    return _subtrees


def write_processed_trees(write_file, _subtrees):
    with open(write_file, "w+") as f:
        for post_wise_subtrees in _subtrees:
            f.write("-" * 14 + "[NEW POST]" + "-" * 3 + "\n")
            for subtree in post_wise_subtrees:
                f.write(subtree + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folders",
        type=str,
        nargs="+",
        help="Folders having data JSON files extracted from reddit.",
    )
    parser.add_argument("--write_folder",
                        type=str,
                        help="The folder where to write the processed data.")
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0,
        help=
        "Percentage of data outside train, to keep in validation data. For 10%, specify 10. Default is 0.",
    )
    args = parser.parse_args()

    config = stable_config + {"data_folders": args.data_folders}
    train_dl = load_reddit_data(config)
    other_dl = load_reddit_data(config, mode="")

    all_post_trees = []
    for tree in train_dl.tree_generator():
        all_post_trees.append(tree)

    subtrees = process_trees(all_post_trees)
    write_processed_trees(os.path.join(args.write_folder, "train.txt"),
                          subtrees)
    del subtrees

    all_post_trees = []
    for tree in other_dl.tree_generator():
        all_post_trees.append(tree)

    subtrees = process_trees(all_post_trees)
    num_valid = int(args.valid_size * len(subtrees)) // 100

    write_processed_trees(
        os.path.join(args.write_folder, "valid.txt"),
        subtrees[:num_valid],
    )
    write_processed_trees(
        os.path.join(args.write_folder, "test.txt"),
        subtrees[num_valid:],
    )
    del subtrees
