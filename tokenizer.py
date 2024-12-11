import regex as re


class BPETokenizer:
    """
    BPE Tokenizer
    referenced by https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """

    def __init__(self, num_merges):

        self.num_merges = num_merges
        self.vocab_size = 256 + num_merges

        self.merges = {}  # merged (pair: idx) for encoding
        self.vocabs_cache = {idx: bytes([idx]) for idx in range(255)}  # vocabs_cache for decoding

        self.gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train(self, text):
        """
        During Tokenizer training, input text will have a huge amount!
        Save merged words in the self.vocabs_cache dictionary
        """
        ids = self._get_ids(text)

        for k in range(self.num_merges):
            stats = self._get_stats(ids)  # {pair: count}
            top_pair = max(stats, key=stats.get)
            idx = k + 256
            ids = self._merge(ids, top_pair, idx)

            self.merges[top_pair] = idx
            print(f"merging {top_pair} into a new token {idx}")

        # add merged text into vocabs_cache for decoding
        for (p0, p1), idx in self.merges.items():
            self.vocabs_cache[idx] = self.vocabs_cache[p0] + self.vocabs_cache[p1]  # concat bytes

        return

    def encode(self, text):
        ids = self._get_ids(text)

        while True:
            stats = self._get_stats(ids)
            if len(stats) == 0:
                break
            pair = min(stats, key=lambda pair: self.merges.get(pair, float("inf")))  # min idx in the merges
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        text = b"".join([self.vocabs_cache[idx] for idx in ids])
        text = text.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def _get_ids(text):
        tokens = text.encode("utf-8")
        return list(map(int, tokens))

    @staticmethod
    def _get_stats(ids):
        counts = {}
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids, top_pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == top_pair[0] and ids[i + 1] == top_pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
