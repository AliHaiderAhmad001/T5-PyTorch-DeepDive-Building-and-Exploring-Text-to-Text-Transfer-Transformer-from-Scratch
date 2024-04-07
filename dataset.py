from torch.utils.data import Dataset
from transformers import T5TokenizerFast

class CustomTextDataset(Dataset):
    """
    A PyTorch Dataset class for handling custom text data.

    Args:
        filename (str): Path to the file containing the text data.
        tokenizer (str or transformers.PreTrainedTokenizerFast, optional): Pre-trained tokenizer or tokenizer name to use for tokenization. Defaults to 't5-base'.
        max_token_len (int, optional): Maximum length of tokens for encoding. Defaults to 512.
    """
    def __init__(self, filename, tokenizer='t5-base', max_token_len=512):
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer)
        self.items = []
        self.max_token_len = max_token_len

        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                encoder_in = file.readline().strip()
                if not encoder_in:
                    break
                decoder_in = file.readline().strip()
                decoder_out = file.readline().strip()
                self.items.append((encoder_in, decoder_in, decoder_out))

                _ = file.readline()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the encoder and decoder inputs, attention mask, and labels.
        """
        encoder_in, decoder_in, decoder_out = self.items[idx]

        encoder_ids = self.tokenizer(
            encoder_in,
            max_length=self.max_token_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_ids = self.tokenizer(
            decoder_in,
            max_length=self.max_token_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids
        labels = self.tokenizer(
            decoder_out,
            max_length=self.max_token_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids

        return {
            "encoder_ids": encoder_ids["input_ids"].squeeze(),
            "attention_mask": encoder_ids["attention_mask"].squeeze(),
            "decoder_ids": target_ids.squeeze(),
            "labels": labels.squeeze()
        }
