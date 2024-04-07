from typing import Tuple, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scheduled_optim import ScheduledOptim
from config import Config
from t5_model import T5Model

class T5Trainer:
    """
    A trainer class for T5Model, designed to encapsulate the training and testing routines. It manages the training epochs,
    data loading, model optimization, and evaluation, providing a streamlined workflow for experimenting with the T5Model.
    The trainer supports both training and testing phases, handling the forward pass, loss computation, gradient backpropagation,
    and parameter updates.

    Attributes:
        model (T5Model): The T5Model instance to be trained and evaluated.
        log_freq (int): Frequency of logging training progress (number of iterations).
        batch_size (int): The size of input data batches for training and testing.
        save_path (str): Path where the trained model checkpoints will be saved.
        device (torch.device): The device (CPU/GPU) on which the model and data should be loaded.
        train_data (DataLoader): DataLoader instance providing access to the training data.
        test_data (DataLoader, optional): DataLoader instance providing access to the test data. Default is None.
        optim (ScheduledOptim): The optimizer with a learning rate scheduling mechanism used for training.
        criterion (nn.NLLLoss): The loss function used for model training.

    Args:
        config (Config): Configuration object containing model, training, and optimization settings.
        t5 (T5Model): The T5Model instance to be trained and evaluated.
        optim (ScheduledOptim): The optimizer with a learning rate scheduling mechanism used for training.
        device (torch.device): The device (CPU/GPU) on which the model and data should be loaded.
        train_dataloader (DataLoader): DataLoader instance providing access to the training data.
        test_dataloader (DataLoader, optional): DataLoader instance providing access to the test data. Default is None.
    """

    def __init__(self, config: Config, t5: T5Model, optim: ScheduledOptim, device: torch.device,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None):
        self.model = t5
        self.log_freq: int = config.log_freq
        self.batch_size: int = config.batch_size
        self.save_path: str = config.save_path
        self.device = device
        # Setting the train and test data loader
        self.train_data: DataLoader = train_dataloader
        self.test_data: DataLoader = test_dataloader
        self.optim: ScheduledOptim = optim

        # Using Negative Log Likelihood Loss function
        self.criterion: nn.NLLLoss = nn.NLLLoss(ignore_index=-100)


        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def train(self, epoch: int) -> None:
        """
        Train the T5Model for one epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch: int) -> None:
        """
        Evaluates the T5Model on the test dataset, if provided.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch: int, data_iter: DataLoader, train: bool = True) -> None:
        """
        Perform an iteration of training or testing.

        Args:
            epoch (int): Current epoch number.
            data_iter (DataLoader): DataLoader for the data.
            train (bool): Whether to train the model (True) or test (False).
        """
        str_code: str = "train" if train else "test"
        avg_loss: float = 0.0
        i: int = 0

        print("Number of batches in DataLoader:", len(data_iter))

        for batch in data_iter:

            batch = {key: value.to(self.device) for key, value in batch.items()}

            encoder_ids = batch['encoder_ids']
            decoder_ids = batch['decoder_ids']
            labels = batch['labels']
            mask = batch['attention_mask']

            lm_output = self.model.forward(encoder_ids, decoder_ids, mask)

            loss = self.criterion(lm_output.transpose(1, 2), labels)

            if train:
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step_and_update_lr()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                output_str = "Epoch: {}, Iteration: {}, Avg Loss: {:.4f}, Current Loss: {:.4f}".format(
                    post_fix['epoch'], post_fix['iter'], post_fix['avg_loss'], post_fix['loss']
                )
                print(output_str)

            i += 1

        print("Epoch %d, %s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))


    def save(self, epoch: int) -> str:
        """
        """
        output_path: str = self.save_path + ".ep%d" % epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        self.model.to(self.device)

        print("Epoch %d Model and Optimizer State Saved on:" % epoch, output_path)
        return output_path
