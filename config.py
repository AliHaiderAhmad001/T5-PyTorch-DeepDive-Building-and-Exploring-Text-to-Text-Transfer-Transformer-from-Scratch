class Config:
    """
    Configuration class for your model and training.

    Args:
        prop (float): Proportion of tokens to mask in each sentence. Default is 0.15.
        tokenizer_path (str): Path to the tokenizer. Default is "bert-base-uncased".
        max_token_len (int): Maximum sequence length. Default is 128.
        data_dir (str): Directory containing the data file.

        # Embeddings params
        hidden_size (int): Size of the hidden layers. Default is 768.
        vocab_size (int): Size of the vocabulary. Default is 30522.
        hidden_dropout_prob (float): Dropout probability for hidden layers. Default is 0.1.

        # Attention params
        num_heads (int): Number of attention heads. Default is 8.

        # model params
        num_blocks (int): Number of blocks in the BERT model. Default is 12.

        # Optimizer params
        n_warmup_steps (int): Number of warmup steps for the optimizer. Default is 10000.

        # Trainer params
        cuda_devices (list): List of CUDA devices. Default is None.
        with_cuda (bool): Flag to use CUDA. Default is True.
        log_freq (int): Logging frequency. Default is 10.
        batch_size (int): Batch size for training. Default is 64.
        save_path (str): Path to save model checkpoints. Default is 'tmp/checkpoints'.

        # Run the model params
        seed (int): Random seed for reproducibility. Default is 0.
        test_dataset (str): Path to the test dataset or None. Default is None.
        epochs (int): Number of training epochs. Default is 1.
    """

    def __init__(self, prop=0.15, tokenizer_path="t5-base", max_token_len=768, data_dir="dataset/train.txt",
                 hidden_size=768, bidirectional = True, num_buckets = 32, max_distance = 128,
                 vocab_size=32000, hidden_dropout_prob=0.1, num_heads=8, num_blocks=12,
                 n_warmup_steps=10000, cuda_devices=None, with_cuda=True, log_freq=10, batch_size=64, save_path='tmp/checkpoints',
                 seed=0, test_dataset=None, epochs=1):

        # Dataset params
        self.prop = prop
        self.tokenizer_path = tokenizer_path
        self.max_token_len = max_token_len
        self.data_dir = data_dir

        # Relative pos params
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Embeddings params
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 100 # We add 100 for the number of sentinals we use
        self.hidden_dropout_prob = hidden_dropout_prob

        # Attention params
        self.num_heads = num_heads

        # BERT model params
        self.num_blocks = num_blocks

        # Optimizer params
        self.n_warmup_steps = n_warmup_steps

        # Trainer params
        self.cuda_devices = cuda_devices
        self.with_cuda = with_cuda
        self.log_freq = log_freq
        self.batch_size = batch_size
        self.save_path = save_path

        # Run the model params
        self.seed = seed
        self.test_dataset = test_dataset
        self.epochs = epochs
