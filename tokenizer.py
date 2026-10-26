import pandas as pd
import pickle
from bpe_utils import BPEUtils

class Tokenizer:
    def __init__(self, rules):
        self.rules = rules

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in self.rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

    def encode(self, data):
        result = []
        for token in BPEUtils.pretokenize(data):
            ids = list(token.encode('utf-8'))

            while len(ids) >= 2:
                pairs = set(zip(ids, ids[1:]))
                pair = min(pairs, key=lambda p: self.rules.get(p, float('inf')))
                if pair not in self.rules:
                    break

                ids = BPEUtils.merge(ids, pair, self.rules[pair])

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
    encoded = tokenizer.encode('대한민국의 수도는 서울이다')
    print(encoded)
    print([tokenizer.decode([i]) for i in encoded])

    # df = pd.read_parquet("./dataset/train-00001-of-00003.parquet", columns=["text"])
    # texts = df["text"].sample(n=300, random_state=0)

    # n_chars = n_bytes = n_tokens = 0
    # for text in texts:
    #     n_chars += len(text)
    #     n_bytes += len(text.encode('utf-8'))
    #     n_tokens += len(tokenizer.encode(text))

    # print(f"bytes/token: {n_bytes / n_tokens:.2f}")
    # print(f"chars/token: {n_chars / n_tokens:.2f}")