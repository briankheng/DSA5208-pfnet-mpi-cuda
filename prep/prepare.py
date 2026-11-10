import os
import sys
from collections import Counter

import numpy as np
from nltk.corpus import stopwords

digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}


def clean(words, stop):
    drop = stop | digit_words
    return [w for w in words if w not in drop]


def build(words, cap):
    counts = Counter(words)
    # the ids are defined by (count descending, word), not by where a word first appears in the text
    vocab = sorted(counts, key=lambda w: (-counts[w], w))[:cap]
    index = {w: i for i, w in enumerate(vocab)}
    ids = np.array([index.get(w, -1) for w in words], dtype=np.int32)
    return vocab, ids


def main():
    with open(sys.argv[1]) as f:
        words = f.read().split()
    words = clean(words, set(stopwords.words("english")))
    vocab, ids = build(words, cap=32768)

    with open(os.path.join(sys.argv[2], "vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    ids.tofile(os.path.join(sys.argv[2], "tokens.bin"))

    counts = np.bincount(ids[ids >= 0])
    print(f"tokens kept: {len(ids)}")
    print(f"vocabulary size: {len(vocab)}")
    # rarest word kept for m = 2000 and for n = 4096, 8192, 16384
    for i in (1999, 4095, 8191, 16383):
        print(f"count at id {i}: {counts[i]}")


if __name__ == "__main__":
    main()
