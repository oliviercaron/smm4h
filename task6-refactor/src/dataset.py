# src/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class VaccineDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        tokenizer: AutoTokenizer, 
        max_length: int = 384, 
        truncation: bool = True
    ):
        """
        Custom PyTorch Dataset for vaccine adverse event detection
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame
            tokenizer (AutoTokenizer): Tokenizer for text encoding
            max_length (int): Maximum sequence length
            truncation (bool): Whether to truncate sequences
        """
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Tokenize and prepare individual item
        
        Returns:
            Dict with tokenized inputs and label
        """
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=self.truncation,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item