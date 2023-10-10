from relative_logger import get_logger
import torch
import spacy
from utils import slice_text, build_dataset
from train import Trainer
from model import TransformerNetwork

logger = get_logger(__name__)

class TrainWrapper:
    def __init__(self, train_files, test_files, context_len: int=16, tokenizer=None, device=None):
        logger.debug("Initializing TrainWrapper object...")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer if tokenizer else spacy.load("en_core_web_sm")
        self.trainer: Trainer = None

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
            doc = self.tokenizer(text)
            tokens = [token.text for token in doc]
            all_tokens.extend(tokens)

        unique_tokens = set(all_tokens)
        logger.info(f"Finished processing pre-training data. Creating dictionary of size {len(unique_tokens)}...")
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.reverse_vocab = {i: token for i, token in enumerate(unique_tokens)}

        slice_length = context_len + 1
        slice_offset = slice_length # TODO: Implement richer, cleaner configuration of slice offsets
        # note: slice_text returns an n by slice_length tensor of ints. (from vocab)
        train_slices = [] # list of tensors
        for text in self.train_texts:
            train_slices.append(slice_text(text, slice_length, slice_offset, context_len, self.tokenizer, self.device, self.vocab))
            train_slices.append(slice_text(text, slice_length - 2, 1, context_len, self.tokenizer, self.device, self.vocab))
            train_slices.append(slice_text(text, 5, 1, context_len, self.tokenizer, self.device, self.vocab))
        self.train_dataset = build_dataset(torch.cat(train_slices, dim=0))
        test_slices = [] # list of tensors
        for text in self.test_texts:
            test_slices.append(slice_text(text, slice_length - 3, 1, context_len, self.tokenizer, self.device, self.vocab))
        self.test_dataset = build_dataset(torch.cat(test_slices, dim=0))   
    
    def setup(self, model: TransformerNetwork):
        self.trainer = Trainer(model, self.train_dataset, self.test_dataset, self.device)

    def train(self, epochs, do_val=True):
        assert self.trainer
        self.trainer.train(epochs, do_val)

    @property
    def test_loader(self):
        return self.trainer.test_loader
