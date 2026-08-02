from __future__ import annotations

import os
from array import array

import regex


GPT2_TOKENIZER_REGEX = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a GPT-2-style byte-level BPE tokenizer.

    Special-token occurrences are excluded from ordinary BPE training and are
    inserted into the vocabulary as atomic tokens.  The returned vocabulary
    maps token IDs to their original byte strings.
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must include all 256 byte tokens and special tokens")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, encoding="utf-8") as file:
        text = file.read()

    # Split on special tokens first.  Keeping each surrounding chunk
    # separate prevents a merge from crossing a special-token boundary.
    chunks = [text]
    for special_token in special_tokens:
        next_chunks: list[str] = []
        for chunk in chunks:
            next_chunks.extend(regex.split(regex.escape(special_token), chunk))
        chunks = next_chunks

    # Each token is a node in a doubly linked list.  Pair locations are kept
    # incrementally, so a merge updates only its local neighbourhood.
    # array("i") c-style signed array
    token_ids = array("i")
    previous = array("i")
    following = array("i")
    active = bytearray()
    pair_locations: dict[tuple[int, int], set[int]] = {}

    def add_pair(left: int) -> None:
        right = following[left]
        if right < 0:
            return
        # this is candidate
        pair = (token_ids[left], token_ids[right])
        locations = pair_locations.setdefault(pair, set())
        locations.add(left)

    def remove_pair(left: int) -> None:
        right = following[left]
        if right < 0:
            return
        pair = (token_ids[left], token_ids[right])
        locations = pair_locations[pair]
        locations.remove(left)
        if locations:
            return
        else:
            del pair_locations[pair]

    for text_chunk in chunks:
        for segment in GPT2_TOKENIZER_REGEX.findall(text_chunk):
            encoded = segment.encode("utf-8")
            if not encoded:
                continue
            start = len(token_ids)
            for offset, byte in enumerate(encoded):
                token_ids.append(byte)
                previous.append(len(token_ids) - 2 if offset else -1)
                following.append(-1)
                active.append(1)
                if len(token_ids) > start + 1:
                    following[len(token_ids) - 2] = len(token_ids) - 1
            for left in range(start, len(token_ids) - 1):
                add_pair(left)

    vocab = {index: bytes([index]) for index in range(256)}
    next_id = 256
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        if not pair_locations:
            raise ValueError("vocab_size exceeds the number of trainable BPE merges")
        pair = max(
            pair_locations,
            key=lambda candidate: (
                len(pair_locations[candidate]),
                vocab[candidate[0]],
                vocab[candidate[1]],
            ),
        )
        left_id, right_id = pair
        locations = list(pair_locations[pair])
        if left_id == right_id:
            locations.sort()

        new_id = next_id
        next_id += 1
        merges.append((vocab[left_id], vocab[right_id]))
        vocab[new_id] = vocab[left_id] + vocab[right_id]

        for left in locations:
            if not active[left]:
                continue
            right = following[left]
            if right < 0 or not active[right]:
                continue
            if token_ids[left] != left_id or token_ids[right] != right_id:
                continue

            before = previous[left]
            after = following[right]
            if before >= 0:
                remove_pair(before)
            remove_pair(left)
            remove_pair(right)

            token_ids[left] = new_id
            following[left] = after
            if after >= 0:
                previous[after] = left
            active[right] = 0

            if before >= 0:
                add_pair(before)
            add_pair(left)

    return vocab, merges
