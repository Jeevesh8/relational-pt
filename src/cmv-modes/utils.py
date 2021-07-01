import copy
from functools import wraps
from typing import List, Tuple

import tensorflow as tf
from ..params import config


def convert_outputs_to_tensors(dtype):
    def inner(func):
        @wraps(func)
        def tf_func(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return tuple(
                (tf.convert_to_tensor(elem, dtype=dtype) for elem in outputs))

        return tf_func

    return inner


def find_sub_list(sl, l):
    """
    Returns the start and end positions of sublist sl in l
    """
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll


def modified_mask_encodings(
        encoding: List[int], tok,
        markers: List[str]) -> Tuple[List[int], List[List[int]]]:
    """Modifies the text corresponding to encoding, to add spaces before discourse markers and finds the apt encodings
    to use for the matches of markers in the resulting text.

    Args:
        encoding: Encoding of original text in the data.
        tok:      The tokenizer being used. Must implement encode(), decode() methods that add <s>, </s> tokens
        markers:  List of markers to be masked. Markers will be masked in a case-insensistive way, and the case
                 of the original text will be preserved.
    Returns:
        A tuple whose first element is encoding of the modified text(with spaces before markers) and second element
        is a list of sequences corresponding to the sequences to be masked in the encoding of modified text.A sequence
        is repeated as mnay times as it occurs in encoding.

    Note:
        A marker is said to be "matched" to some part of resulting text iff both the part and the marker consist of same
        characters(may have different cases), and the immediate characters to the left and right of the part are either
        string boundaries or non-alphanumeric characters.
    """
    markers = [marker.strip() for marker in markers]

    if encoding[0] != tok.bos_token_id:
        encoding = [tok.bos_token_id] + encoding
    if encoding[-1] != tok.eos_token_id:
        encoding = encoding + [tok.eos_token_id]

    decoded_txt = tok.decode(encoding)[3:-4]
    special_tokens = tok.get_added_vocab().keys()
    #    print("Text to encode: ")
    #    print(decoded_txt)
    #    print("\n\n")
    new_txt_parts = []
    iter_txt = decoded_txt
    seqs_to_mask = []
    while True:
        found_markers_start_pos = [
            iter_txt.lower().find(marker.lower()) for marker in markers
        ]
        idx = min(filter(lambda x: x >= 0, found_markers_start_pos),
                  default=-1)
        marker = markers[found_markers_start_pos.index(idx)]
        if idx == -1:
            new_txt_parts.append(iter_txt)
            break
        #        print("Marker: ", marker, "index: ",  idx, " in ", iter_txt[max(0,idx-10):idx+10])
        if not ((idx > 0 and iter_txt[idx - 1].isalnum()) or
                (idx + len(marker) < len(iter_txt)
                 and iter_txt[idx + len(marker)].isalnum()) or
                (idx == 0 and len(new_txt_parts) > 0
                 and new_txt_parts[-1][-1].isalnum() and iter_txt[0].isalnum())
                ):  # if the match is not internal to a word

            new_txt_parts.append(iter_txt[:idx])

            if not (
                    idx == 0 or iter_txt[idx - 1] == " "
            ):  # if there isn't any pre-existing space, preceeding the marker
                new_txt_parts.append(" ")  # add a space before the markeri

            new_txt_parts.append(iter_txt[idx:idx + len(marker)])

            if not (
                    idx + len(marker) == len(iter_txt)
                    or iter_txt[idx + len(marker)] == " "
            ):  # if there isn't any pre-existing space, following the marker
                new_txt_parts.append(" ")  # add a space after the marker

            words_before = iter_txt[:idx].strip().split()
            if idx == 0 or (
                    len(words_before) > 0
                    and words_before[-1] in special_tokens
            ):  # Since, tokenization is same for "[STARTQ] On the.." and "[STARTQ]On the.." spaces following special tokens are removed while tokenization.

                to_mask = tok.encode(iter_txt[idx:idx + len(marker)])[1:-1]
            else:
                to_mask = tok.encode(" " +
                                     iter_txt[idx:idx + len(marker)])[1:-1]
            seqs_to_mask.append(to_mask)
        else:
            #            if len(new_txt_parts)>0:
            #                print("Skipping the previour marker", new_txt_parts[-1])
            new_txt_parts.append(iter_txt[:idx + len(marker)])

        iter_txt = iter_txt[idx + len(marker):]
    new_encoding = tok.encode("".join(new_txt_parts))
    return new_encoding, seqs_to_mask


def reencode_mask_tokens(encoding: List[int], tok,
                         markers: List[str]) -> Tuple[List[int], List[int]]:
    """
    Args:
        Same as modified_mask_encodings()
    Returns:
        The modified encoding for optimal matching of markers in the text corresponding to encoding,
        and the encoding with all the matches masked.
    """
    #   print("Original encoding: ", tok.decode(encoding))
    new_encoding, seqs_to_mask = modified_mask_encodings(
        encoding, tok, markers)
    #   print("New encoding: ", new_encoding)
    masked_encoding = copy.deepcopy(new_encoding)
    mask_token = tok.mask_token_id
    for seq in seqs_to_mask:
        start, end = find_sub_list(seq, masked_encoding)
        #        print("Masking", tok.decode(seq), "from", start, end)
        masked_encoding[start:end] = [mask_token] * (end - start)
    return new_encoding, masked_encoding


def get_rel_type_idx(relation: str) -> int:
    for i, v in enumerate(config["relations_map"].values()):
        if relation in v:
            return i
    return 0  # Assuming None relation is 0-th position, always.
