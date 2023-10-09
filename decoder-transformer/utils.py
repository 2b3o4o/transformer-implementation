import torch
from torch import tensor
from torch.nn.functional import softmax
from train import CompletionDataset
from relative_logger import get_logger

logger = get_logger(__name__)

def prep_input_string(string, context_len, vocab, tokenizer) -> tensor:
    """Takes an input string with up to context_len tokens and returns a tensor full of integers, which can be passed into the model"""
    tokens = tokenizer(string)

    output = torch.full([context_len], vocab['<PAD>'])
    for in_pos in range(len(tokens)):
        out_pos = context_len - len(tokens) + in_pos
        output[out_pos] = vocab[tokens[in_pos].text]

    return output

def prep_tokens(tokens, length, vocab) -> tensor:
    output = torch.full([length], vocab['<PAD>'])
    for in_pos in range(len(tokens)):
        out_pos = length - len(tokens) + in_pos
        output[out_pos] = vocab[tokens[in_pos].text]

    return output

# slice_offset is the number of tokens separating the start of one slice from the start of the previous.
# slice_offset == slice_length means no overlap, slice_offset == 1 means maximum overlap.
def slice_text(text: str, slice_length, slice_offset, context_len, tokenizer, output_device, vocab) -> tensor:
    slices = []
    tokens = tokenizer(text)

    for i in range(0, len(tokens), slice_offset):
        slices.append(tokens[i:i+slice_length])

    output = torch.zeros([len(slices), context_len + 1]) # use context_len + 1 because we need to include the label
    for i, slice in enumerate(slices):
        output[i] = prep_tokens(slice, context_len + 1, vocab)

    assert output.shape[1] == context_len + 1
    return output.to(output_device)

def slice_by_line(text: str, context_len, tokenizer, output_device) -> tensor:
    slices = text.split("\n")
    tokens = [tokenizer(slice) for slice in slices]

    output = torch.zeros([len(tokens), context_len + 1])
    for i, token_line in enumerate(tokens):
        output[i] = prep_tokens(token_line, context_len + 1)

    return output.to(output_device)

def build_dataset(slices: tensor) -> CompletionDataset:
    features = slices[:, :-1]
    labels = slices[:, -1]
    
    dataset = CompletionDataset(features, labels)
    return dataset

def check_input_data(input, reverse_vocab):
    """
    Sanity check helper for CompletionDataset objects.
    Usage:
    check_input_data(train_dataset[0])
    This should print something like:
    Features:
    ['The', 'Complete', 'Works', 'of', 'William', 'Shakespeare', '\n\n', 'by', 'William', 'Shakespeare', '\n\n\n\n\n                    ', 'Contents', '\n\n    ', 'THE', 'SONNETS', '\n    ']
    Label:
    ALL
    """
    features = input[0].int().tolist()
    label = input[1].int().item()
    features_str = [reverse_vocab[f] for f in features]
    label_str = reverse_vocab[label]
    logger.debug(f"Features:\n{features_str}")
    logger.debug(f"Label:\n{label_str}")

def infer_completion(model, device, vocab, reverse_vocab, input_text: str, context_len, tokenizer):
    encoded_input = prep_input_string(input_text, context_len, vocab, tokenizer).unsqueeze(0).float().to(device)
    
    model.train(False)
    pred = model(encoded_input)
    return reverse_vocab[torch.argmax(softmax(pred, dim=1), dim=1).item()]

def check_test_accuracy(model, test_loader):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Deactivates autograd, reduces memory usage and speeds up computations
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Accuracy on test set: {accuracy}%")