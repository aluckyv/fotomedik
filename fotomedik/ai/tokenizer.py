"""The CLIP byte-pair tokenizer: a prompt -> the 77 token ids Stable Diffusion's text
encoder takes. Vocabulary and merge list are the files of OpenAI's CLIP ViT-L/14
(vocab.json, merges.txt), downloaded like the models. Lower-case ASCII prompts are all
that is needed here, so the text clean-up is minimal.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

BOS, EOS = 49406, 49407
LENGTH = 77
_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible mapping of bytes to printable characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ClipTokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.bos = vocab.get("<|startoftext|>", BOS)
        self.eos = vocab.get("<|endoftext|>", EOS)
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def load(cls, vocab_path: Path, merges_path: Path) -> "ClipTokenizer":
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merges = []
        for line in Path(merges_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("#version") or not line.strip():
                continue
            a, b = line.split()
            merges.append((a, b))
        return cls(vocab, merges)

    def _bpe(self, token: str) -> list[str]:
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        return word

    def _ids(self, text: str) -> list[int]:
        text = " ".join(text.lower().split())
        if text in self._cache:
            return self._cache[text]
        b2u = _bytes_to_unicode()
        ids: list[int] = []
        for tok in _PATTERN.findall(text):
            tok = "".join(b2u[b] for b in tok.encode("utf-8"))
            for piece in self._bpe(tok):
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
        self._cache[text] = ids
        return ids

    def encode(self, text: str, length: int = LENGTH) -> list[int]:
        """[BOS] + the prompt's tokens + [EOS], padded with EOS to `length` (cut if longer)."""
        ids = self._ids(text)[: length - 2]
        out = [self.bos] + ids + [self.eos]
        return out + [self.eos] * (length - len(out))
