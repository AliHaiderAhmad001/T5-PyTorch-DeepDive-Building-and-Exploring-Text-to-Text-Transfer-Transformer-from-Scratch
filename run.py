import torch
import argparse
import numpy as np
import random as rd
from trainer import T5Trainer
from adafactor import AdaFactor 
from dataset import CustomTextDataset
from torch.utils.data import DataLoader
from scheduled_optim import ScheduledOptim
from config import Config

def set_seeds(seed, with_cuda):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and with_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a T5 model.')

    parser.add_argument('--prop', type=float, default=0.15, help='Proportion of tokens to mask.')
    parser.add_argument('--tokenizer_path', type=str, default='t5-base', help='Path to tokenizer.')
    parser.add_argument('--max_token_len', type=int, default=768, help='Maximum token length.')
    parser.add_argument('--data_dir', type=str, default='dataset/train.txt', help='Training data directory.')
    parser.add_argument('--hidden_size', type=int, default=768, help='Size of hidden layers.')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Size of vocabulary.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout probability for hidden layers.')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads.')
    parser.add_argument('--num_blocks', type=int, default=8, help='Number of blocks in the model.')
    parser.add_argument('--n_warmup_steps', type=int, default=10000, help='Number of warmup steps for the optimizer.')
    parser.add_argument('--with_cuda', action='store_true', help='Use CUDA for training.')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--save_path', type=str, default='tmp/checkpoints', help='Path to save model checkpoints.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--test_dataset', type=str, default=None, help='Path to the test dataset.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')

    return parser.parse_args()

def main():
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
    args = parse_arguments()

    # Set random seeds
    set_seeds(args.seed, args.with_cuda)

    # Load configuration from args
    config = Config(
        prop=args.prop,
        tokenizer_path=args.tokenizer_path,
        max_token_len=args.max_token_len,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        n_warmup_steps=args.n_warmup_steps,
        with_cuda=args.with_cuda,
        log_freq=args.log_freq,
        batch_size=args.batch_size,
        save_path=args.save_path,
        seed=args.seed,
        test_dataset=args.test_dataset,
        epochs=args.epochs
    )

    print("Configuration loaded from command line arguments.")

    print("Loading Train Dataset...")

    # Load training dataset
    train_dataset = CustomTextDataset(config.data_dir, tokenizer= config.tokenizer_path, max_token_len = config.max_token_len)

    if config.test_dataset is not None:
        print("Loading Test Dataset...")
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
    main()

