# src/model.py
from transformers import (
    AutoModelForSequenceClassification, 
    AutoConfig, 
    AutoTokenizer
)
import torch.nn as nn

class VAEMClassificationModel:
    """
    Vaccine Adverse Event Mention (VAEM) Classification Model
    
    This class handles model configuration, initialization, 
    and provides methods for model customization.
    """
    @staticmethod
    def create_model(
        model_name: str = "allenai/biomed_roberta_base", 
        num_labels: int = 2, 
        dropout_rate: float = 0.3,
        gradient_checkpointing: bool = True
    ):
        """
        Create a configured sequence classification model
        
        Args:
            model_name (str): Pretrained model name/path
            num_labels (int): Number of output classes
            dropout_rate (float): Dropout probability for model layers
            gradient_checkpointing (bool): Enable memory-efficient training
        
        Returns:
            Configured model with custom dropout and optional gradient checkpointing
        """
        # Configure model with specific dropout and labels
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        # Load pretrained model with custom configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=config
        )
        
        # Optional: Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def load_tokenizer(
        model_name: str = "allenai/biomed_roberta_base", 
        max_length: int = 384
    ):
        """
        Load and configure tokenizer
        
        Args:
            model_name (str): Tokenizer model name
            max_length (int): Maximum sequence length
        
        Returns:
            Configured tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optional: Set default tokenization parameters
        tokenizer.truncation_side = 'right'
        tokenizer.model_max_length = max_length
        
        return tokenizer
    
    @staticmethod
    def add_custom_classifier_head(
        model, 
        hidden_size: int = 768, 
        num_labels: int = 2, 
        dropout_rate: float = 0.3
    ):
        """
        Add a custom classification head to the base model
        
        Args:
            model: Base transformer model
            hidden_size (int): Size of the hidden layer
            num_labels (int): Number of output classes
            dropout_rate (float): Dropout probability
        
        Returns:
            Model with custom classification head
        """
        # Add dropout and classification layers
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        return model