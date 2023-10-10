from relative_logger import get_logger
import torch
from torch.utils.data import DataLoader
from model import TransformerNetwork

logger = get_logger(__name__)

class CompletionDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

class Trainer:
    def __init__(self, model: TransformerNetwork, train_dataset: CompletionDataset, test_dataset: CompletionDataset, device: torch.device=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    def train_one_epoch(self, do_validation: bool=True):
        self.model.train(True)
        torch.set_printoptions(profile="short")
        batches = 0
        avg_loss = 0
        for step, (features, labels) in enumerate(self.train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(features)

            loss = self.loss_func(preds, labels)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss
            batches = step + 1
            
            del features
            del labels

        avg_loss = avg_loss / batches
        logger.info(f"Average loss for training batches in this epoch: {avg_loss}")

        if do_validation:
            self.model.train(False)
            batches = 0
            avg_loss = 0
            for step, (features, labels) in enumerate(self.test_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                preds = self.model(features)
                loss = self.loss_func(preds, labels)

                avg_loss += loss
                batches = step + 1

                del features
                del labels

            avg_loss = avg_loss / batches
            logger.info(f"Average loss for validation batches in this epoch: {avg_loss}")

    def train(self, epochs, do_val=True):
        for i in range(epochs):
            logger.info(f"Epoch {i}:")
            self.train_one_epoch(do_val)
            torch.cuda.empty_cache()
