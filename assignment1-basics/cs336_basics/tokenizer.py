from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex


GPT2_TOKENIZER_REGEX = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.merge_outputs = {pair: pair[0] + pair[1] for pair in merges}
        self.special_tokens = tuple(sorted(special_tokens or [], key=len, reverse=True))
        self.special_token_ids = {
            special_token.encode("utf-8"): self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }
        if self.special_tokens:
            escaped = "|".join(regex.escape(token) for token in self.special_tokens)
            self.special_pattern = regex.compile(f"(?:{escaped})")
        else:
            self.special_pattern = None

    # Function of encode_piece:_encode_piece converts a pre-tokenized text piece into token IDs by encoding it
    # as UTF-8 bytes, repeatedly applying the highest-priority BPE merges, and mapping
    # the resulting byte tokens to their vocabulary IDs.
    def _encode_piece(self, piece: str) -> list[int]:
        symbols = [bytes([byte]) for byte in piece.encode("utf-8")]
        while len(symbols) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = len(self.merge_ranks)
            for left, right in zip(symbols, symbols[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            merged = self.merge_outputs[best_pair]
            result: list[bytes] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == best_pair:
                    result.append(merged)
                    index += 2
                else:
                    result.append(symbols[index])
                    index += 1
            symbols = result

        return [self.token_to_id[token] for token in symbols]

    def _encode_normal_text(self, text: str) -> Iterator[int]:
        for match in GPT2_TOKENIZER_REGEX.finditer(text):
            yield from self._encode_piece(match.group(0))

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_normal_text(text))

        token_ids: list[int] = []
        cursor = 0
        for match in self.special_pattern.finditer(text):
            token_ids.extend(self._encode_normal_text(text[cursor : match.start()]))
            token_ids.append(self.special_token_ids[match.group(0).encode("utf-8")])
            cursor = match.end()
        token_ids.extend(self._encode_normal_text(text[cursor:]))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Keep one input line as carry so tokenization remains equivalent to
        # encoding the complete stream while keeping memory bounded.
        carry = ""
        for chunk in iterable:
            carry += chunk
            lines = carry.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                carry = lines.pop()
            else:
                carry = ""
            for line in lines:
                yield from self.encode(line)
        if carry:
            yield from self.encode(carry)
