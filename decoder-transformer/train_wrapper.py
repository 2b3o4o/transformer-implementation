import torch
import spacy
from .utils import slice_text, build_dataset
from .train import Trainer
from .model import TransformerNetwork

class TrainWrapper:
    def __init__(self, model: TransformerNetwork, train_files: [str]=None, test_files: [str]=None, tokenizer=None, device=None):
        context_len = model.context_len

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer if tokenizer else spacy.load("en_core_web_sm")
        if not train_files:
            train_files = [
            "../data/_part1.txt",
            # "../data/_part2.txt",
            # "../data/_part3.txt",
            # "../data/_part4.txt",
            # "../data/_part5.txt",
            # "../data/_part6.txt",
            # "../data/_part7.txt"
        ]
        if not test_files:
            test_files = ["../data/much_ado_about_nothing_gut.txt"]

        self.train_texts = []
        self.test_texts = []
        for file_name in train_files:
            with open(file_name, 'r', encoding='utf-8') as file:
                self.train_texts.append(file.read())
        for file_name in test_files:
            with open(file_name, 'r', encoding='utf-8') as file:
                self.test_texts.append(file.read())

        all_tokens = []
        all_tokens.extend(['<PAD>', '<UNK>']) # special tokens

        for text in self.train_texts + self.test_texts:
            doc = tokenizer(text)
            tokens = [token.text for token in doc]
            all_tokens.extend(tokens)

        unique_tokens = set(all_tokens)
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.reverse_vocab = {i: token for i, token in enumerate(unique_tokens)}

        slice_length = context_len + 1
        slice_offset = slice_length # TODO: Implement richer, cleaner configuration of slice offsets
        # note: slice_text returns an n by slice_length tensor of ints. (from vocab)
        train_slices = [] # list of tensors
        for text in self.train_texts:
            # train_slices.append(slice_by_line(text, context_len))
            train_slices.append(slice_text(text, slice_length, slice_offset, context_len))
            train_slices.append(slice_text(text, slice_length - 2, 1, context_len))
            train_slices.append(slice_text(text, 5, 1, context_len))
        self.train_dataset = build_dataset(torch.cat(train_slices, dim=0))
        test_slices = [] # list of tensors
        for text in self.test_texts:
            # test_slices.append(slice_by_line(text, context_len))
            test_slices.append(slice_text(text, slice_length - 3, 1, context_len))
        self.test_dataset = build_dataset(torch.cat(test_slices, dim=0))

        self.trainer = Trainer
    
    def train(self):
        train
