import torch
import argparse
import numpy as np
import random as rd
from trainer import T5Trainer
from adafactor import AdaFactor 
from dataset import CustomTextDataset
from torch.utils.data import DataLoader
from scheduled_optim import ScheduledOptim


def set_seeds(config):
    """
    Sets the seed for random number generators in Python's `random` module, NumPy, and PyTorch to ensure reproducible results.
    If CUDA is available and specified in the configuration, it also sets the seed for CUDA's random number generator and
    makes CUDA's operations deterministic.

    This function is crucial for experiments where reproducibility is important, as it ensures that the model initialization,
    data shuffling, and other operations that rely on random number generation can be replicated exactly.

    Args:
        config (Config): A configuration object containing at least a `seed` attribute and a `with_cuda` boolean indicating
                         whether CUDA-specific seeds need to be set for reproducibility.
    """
    rd.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available() and config.with_cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run(config):
    """
    The main function that sets up the environment, loads the dataset, initializes the T5 model along with its optimizer and
    scheduler, and then runs the training and testing loops according to the provided configuration.

    This function serves as the entry point for training the T5 model, orchestrating the process from dataset preparation
    to model training and evaluation. It leverages the `T5Trainer` class to abstract away the complexities of the training
    and testing loops.

    Args:
        config (Config): A configuration object containing all necessary parameters to initialize datasets, the model,
                         optimizer, scheduler, and other components of the training process. This includes dataset paths,
                         model hyperparameters, training options, device configuration, etc.

    Note:
        This function is designed to be called directly from the command line or as part of a script. It reads the configuration,
        prepares the datasets, sets the computational device, initializes the model and its components, and finally starts the
        training process followed by testing, if a test dataset is provided.
    """
    # Set random seeds
    set_seeds(config)

    print("Loading Train Dataset...")

    # Load training dataset
    train_dataset = CustomTextDataset(config.data_dir, tokenizer= config.tokenizer_path, max_token_len = config.max_token_len)

    # Load test dataset if provided
    test_dataset = CustomTextDataset(config.data_dir, tokenizer= config.tokenizer_path, max_token_len = config.max_token_len) if config.test_dataset is not None else None

    # Setup cuda device for T5 training
    cuda_condition: bool = torch.cuda.is_available() and config.with_cuda
    device: torch.device = torch.device("cuda:0" if cuda_condition else "cpu")

    # Initialize T5Model
    t5 = T5Model(config).to(device)

    # Distributed GPU training if CUDA can detect more than 1 GPU
    if config.with_cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUs for T5Model" % torch.cuda.device_count())
        t5: nn.DataParallel = nn.DataParallel(t5, device_ids=config.cuda_devices)

    # Initialize optimizer and scheduler
    """To use a manual (external) learning rate schedule you should set scale_parameter=False and relative_step=False.
      In T5 case, additional optimizer operations like gradient clipping should not be used alongside Adafactor. We also set warmup_init to False.
      # https://discuss.huggingface.co/t/t5-finetuning-tips/684
    """
    optim = Adafactor(t5.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=None)  
    optim_schedule = ScheduledOptim(optim, config.hidden_size, config.n_warmup_steps)

    # Create data loaders
    batch_size = config.batch_size
    train_data_loader = DataLoader(train_dataset, batch_size = batch_size, worker_init_fn=np.random.seed(config.seed), shuffle = True)
    test_data_loader = DataLoader(test_dataset, batch_size= batch_size, worker_init_fn=np.random.seed(config.seed)) if test_dataset is not None else None

    # Initialize t5 trainer
    trainer = T5Trainer(config, t5, optim_schedule, device, train_data_loader, test_data_loader)

    # Training loop
    for epoch in range(config.epochs):
        # Train the model
        trainer.train(epoch)

        # Save the model
        trainer.save(epoch)

        # Test the model if test data is available
        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    # Load configuration
    config = Config(
        prop=0.15,
        tokenizer_path='t5-base',
        max_token_len= 768,
        bidirectional= True,
        num_buckets= 32,
        max_distance= 128,
        data_dir= 'dataset/train.txt',
        hidden_size= 768,
        vocab_size= 32000,
        hidden_dropout_prob= 0.1,
        num_heads= 12,
        num_blocks= 8,
        n_warmup_steps= 10000,
        lr= 0.01,
        cuda_devices=None,
        with_cuda= True,
        log_freq= 10,
        batch_size= 16,
        save_path= 'tmp',
        seed= 2024,
        test_dataset= None,
        epochs= 2
    )

    run(config)
