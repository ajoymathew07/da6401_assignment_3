import spacy
from datasets import load_dataset
from collections import Counter
import torch
from typing import cast, Dict
class Multi30kDataset:
    def __init__(self, split='train', min_freq=2, max_len = 100):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.min_freq = min_freq

        self.dataset = load_dataset('bentrevett/multi30k', split=self.split)
        self.max_len = max_len
        
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.src_vocab = None
        self.tgt_vocab = None

    def tokenize_de(self, text):
        return [token.text.lower() for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent

        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.dataset:

            example = cast(Dict[str, str], example)
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        self.src_vocab = self._build_vocab_from_counter(src_counter)
        self.tgt_vocab = self._build_vocab_from_counter(tgt_counter)
    
    def _build_vocab_from_counter(self, counter):
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        idx = len(self.special_tokens)
        for token, freq in counter.most_common():  # most_common for reproducibility
            if freq >= self.min_freq and token not in vocab:
                vocab[token] = idx
                idx += 1
        return vocab
    
    def numericalize(self, tokens, vocab):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() before process_data()")
        data = []
        for example in self.dataset:
            example = cast(Dict[str, str], example)
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])
            src_tokens = ["<sos>"] + src_tokens + ["<eos>"]
            tgt_tokens = ["<sos>"] + tgt_tokens + ["<eos>"]

  # Skip overly long sequences
            if len(src_tokens) > self.max_len or len(tgt_tokens) > self.max_len:
                continue

            src_indices = self.numericalize(src_tokens, self.src_vocab)
            tgt_indices = self.numericalize(tgt_tokens, self.tgt_vocab)

            data.append((src_indices, tgt_indices))
        return data