import pandas as pd
import pickle
import time
from collections import defaultdict
from bpe_utils import BPEUtils

class BPETrainer:
    def __init__(self, filepath):
        self.vocab_size = 16000
        self.df = pd.read_parquet(filepath, columns=["text"])

    def train_bpe(self):
        token_count = defaultdict(int)
        for text in self.df["text"][:10000]:
            for token in BPEUtils.pretokenize(text):
                token_count[token] += 1

        words = [list(token.encode('utf-8')) for token in token_count.keys()]
        counts = list(token_count.values())

        count_map = {}
        pair_words = defaultdict(set)
        for i, ids in enumerate(words):
            BPEUtils.count_pairs(ids, count_map, counts[i])
            for pair in zip(ids, ids[1:]):
                pair_words[pair].add(i)

        pair_size = self.vocab_size - 256 - 1
        rules = {}

        for step in range(pair_size):
            if step % 100 == 0:
                print(f"step: {step}, total: {pair_size}")

            if not count_map:
                break

            max_pair = None
            max_count = -1
            for key, value in count_map.items():
                if max_count < value or (value == max_count and max_pair < key):
                    max_pair = key
                    max_count = value

            new_id = 256 + step
            rules[max_pair] = new_id

            for i in pair_words.pop(max_pair):
                old_ids = words[i]
                new_ids = BPEUtils.merge(old_ids, max_pair, new_id)

                BPEUtils.count_pairs(old_ids, count_map, -counts[i])
                BPEUtils.count_pairs(new_ids, count_map, counts[i])
                for pair in zip(new_ids, new_ids[1:]):
                    pair_words[pair].add(i)

                for pair in zip(old_ids, old_ids[1:]):
                    if count_map.get(pair) == 0:
                        del count_map[pair]

                words[i] = new_ids

        return rules


if __name__ == "__main__":
    start = time.perf_counter()
    bpe_trainer = BPETrainer("./dataset/train-00000-of-00003.parquet")
    print(f"load: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    rules = bpe_trainer.train_bpe()
    print(f"train: {time.perf_counter() - start:.2f}s")

    with open("./output/bpe_rules.pkl", "wb") as f:
        pickle.dump(rules, f)
