import pandas as pd
import pickle
import time
import numpy as np
from bpe_utils import BPEUtils

class Tokenizer:
    def __init__(self, rules):
        self.rules = rules
        self.end_of_text = "<|endoftext|>"
        self.end_of_text_id = 256 + len(self.rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in self.rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        self.id_to_bytes[self.end_of_text_id] = self.end_of_text.encode('utf-8')
        self.cache = {}

    def encode(self, data):
        result = []
        for token in BPEUtils.pretokenize(data):
            if token in self.cache:
                ids = self.cache[token]
            else:
                ids = list(token.encode('utf-8'))
                while len(ids) >= 2:
                    pairs = set(zip(ids, ids[1:]))
                    pair = min(pairs, key=lambda p: self.rules.get(p, float('inf')))
                    if pair not in self.rules:
                        break

                    ids = BPEUtils.merge(ids, pair, self.rules[pair])

                self.cache[token] = ids

            result.extend(ids)

        return result


    def decode(self, ids):
        decoded = b"".join(self.id_to_bytes[i] for i in ids)
        return decoded.decode('utf-8', errors='replace')


if __name__ == "__main__":
    rules = None
    with open("output/bpe_rules_16k.pkl", "rb") as f:
        rules = pickle.load(f)

    tokenizer = Tokenizer(rules)

    paths = [
        './dataset/poems.parquet',
    ]

    f = open('./output/poem_train.bin', "wb")

    start = time.perf_counter()
    for path in paths:
        step = 0
        df = pd.read_parquet(path, columns=['text'])
        df = df.sample(frac=1, random_state=0)

        size = len(df["text"])
        for text in df["text"]:
            if step % 100 == 0:
                print(f"step: {step}, total: {size}")

            encoded = tokenizer.encode(text) + [tokenizer.end_of_text_id]
            f.write(np.array(encoded, dtype=np.uint16).tobytes())
            step += 1

    f.close()
    print(f"encode: {time.perf_counter() - start:.2f}s")

    # ids = np.fromfile('./output/train.bin', dtype=np.uint16)
    # print(tokenizer.decode(ids[:100000]))