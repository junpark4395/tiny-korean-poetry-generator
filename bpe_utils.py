import regex as re

class BPEUtils:

    @staticmethod
    def pretokenize(text):
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.findall(pattern, text)

    @staticmethod
    def count_pairs(ids, count_map, weight):
        for pair in zip(ids, ids[1:]):
            count_map[pair] = count_map.get(pair, 0) + weight

    @staticmethod
    def merge(ids, pair, new_id):
        merged_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                merged_ids.append(new_id)
                i += 2
            else:
                merged_ids.append(ids[i])
                i += 1

        return merged_ids
