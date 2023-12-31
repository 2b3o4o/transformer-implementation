import argparse
from relative_logger import get_logger
from train_wrapper import TrainWrapper
from model import TransformerNetwork
from utils import infer_completion, check_test_accuracy

logger = get_logger(__name__)

def micro_model() -> (TrainWrapper, TransformerNetwork, int):
    logger.info("Creating model from micro preset...")
    train_files = ["../data/_part1.txt"]
    test_files = ["../data/much_ado_about_nothing_gut.txt"]
    context_len = 8
    trainer = TrainWrapper(context_len=context_len, train_files=train_files, test_files=test_files)
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=1, model_dim=32, att_heads=4, ff_hidden_dim=64, name="micro")
    trainer.setup(model)
    return (trainer, model, context_len)

def tiny_model() -> (TrainWrapper, TransformerNetwork, int):
    logger.info("Creating model from tiny preset...")
    train_files = ["../data/_part1.txt"]
    test_files = ["../data/much_ado_about_nothing_gut.txt"]
    context_len = 16
    trainer = TrainWrapper(context_len=context_len, train_files=train_files, test_files=test_files)
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=2, model_dim=128, att_heads=4, ff_hidden_dim=256, name="tiny")
    trainer.setup(model)
    return (trainer, model, context_len)

def small_model() -> (TrainWrapper, TransformerNetwork, int):
    logger.info("Creating model from small preset...")
    train_files = [
        "../data/_part1.txt",
        "../data/_part2.txt",
        "../data/_part3.txt",
        "../data/_part4.txt",
        "../data/_part5.txt",
        "../data/_part6.txt",
    ]
    test_files = ["../data/_part7.txt"]
    context_len = 32
    trainer = TrainWrapper(context_len=context_len, train_files=train_files, test_files=test_files)
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=6, model_dim=256, att_heads=8, ff_hidden_dim=1024, name="small")
    trainer.setup(model)
    return (trainer, model, context_len)

def main(preset: str, epochs: int):
    if preset == "micro":
        trainer, model, context_len = micro_model()
    elif preset == "tiny":
        trainer, model, context_len = tiny_model()
    elif preset == "small":
        trainer, model, context_len = small_model()
    else:
        raise ValueError(f"Invalid preset: {preset}")

    trainer.train(epochs)

    # Sanity checks:
    check_test_accuracy(model, trainer.test_loader)
    string = "From fairest creatures we desire"
    logger.info(f"Completion test: {string}{infer_completion(model, model.device, trainer.vocab, trainer.reverse_vocab, string, context_len, trainer.tokenizer)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument("preset", type=str, choices=["micro", "tiny", "small"], help="Model size preset (micro, tiny, small).")
    parser.add_argument("epochs", type=int, help="Number of training epochs.")
    args = parser.parse_args()

    main(args.preset, args.epochs)
