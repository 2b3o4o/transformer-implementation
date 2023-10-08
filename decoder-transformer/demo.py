from train_wrapper import TrainWrapper
from model import TransformerNetwork
from utils import infer_completion, check_test_accuracy

def micro_model() -> (TrainWrapper, TransformerNetwork, int):
    train_files = ["../data/_part1.txt"]
    test_files = ["../data/much_ado_about_nothing_gut.txt"]
    context_len = 8
    trainer = TrainWrapper(context_len=context_len, train_files=train_files, test_files=test_files)
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=1, model_dim=32, att_heads=4, ff_hidden_dim=64)
    trainer.setup(model)
    return (trainer, model, context_len)

def tiny_model() -> (TrainWrapper, TransformerNetwork, int):
    train_files = ["../data/_part1.txt"]
    test_files = ["../data/much_ado_about_nothing_gut.txt"]
    context_len = 16
    trainer = TrainWrapper(context_len=context_len, train_files=train_files, test_files=test_files)
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=2, model_dim=128, att_heads=4, ff_hidden_dim=256)
    trainer.setup(model)
    return (trainer, model, context_len)

def small_model() -> (TrainWrapper, TransformerNetwork, int):
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
    model = TransformerNetwork(output_dict_size=len(trainer.vocab), context_len=context_len, num_layers=6, model_dim=256, att_heads=8, ff_hidden_dim=1024)
    trainer.setup(model)
    return (trainer, model, context_len)

def main():
    trainer, model, context_len = micro_model() # Try uncommenting larger presets
    # trainer, model, context_len = tiny_model()
    # trainer, model, context_len = small_model()
    trainer.train(1) # Try increasing to 100
    
    check_test_accuracy(model, trainer.test_loader)
    string = "From fairest creatures we desire"
    print(f"Completion test: {string}{infer_completion(model, model.device, trainer.vocab, trainer.reverse_vocab, string, context_len, trainer.tokenizer)}")

if __name__ == "__main__":
    main()