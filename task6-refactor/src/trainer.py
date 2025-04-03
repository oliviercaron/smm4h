# src/trainer.py
from transformers import Trainer
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    fbeta_score
)

class CustomVAEMTrainer(Trainer):
    """
    Custom Trainer for Vaccine Adverse Event Mention Detection
    
    Extends the Hugging Face Trainer with custom loss handling 
    and enhanced evaluation metrics
    """
    def __init__(
        self, 
        *args, 
        alpha: float = 0.33, 
        beta: float = 0.33, 
        gamma: float = 0.33, 
        **kwargs
    ):
        """
        Initialize custom trainer with configurable loss weights
        
        Args:
            alpha (float): Cross-entropy loss weight
            beta (float): Focal loss weight
            gamma (float): Dice loss weight
        """
        super().__init__(*args, **kwargs)
        self.alpha_weight = alpha
        self.beta_weight = beta
        self.gamma_weight = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined custom loss
        
        Args:
            model: Transformer model
            inputs: Batch inputs
            return_outputs (bool): Whether to return model outputs
        
        Returns:
            Combined loss and optional model outputs
        """
        from src.loss import combined_loss
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = combined_loss(
            logits, 
            labels, 
            alpha=self.alpha_weight,
            beta=self.beta_weight,
            gamma=self.gamma_weight
        )
        
        return (loss, outputs) if return_outputs else loss
    
    @staticmethod
    def compute_metrics(eval_pred):
        """
        Enhanced metrics computation
        
        Args:
            eval_pred: Evaluation predictions
        
        Returns:
            Dictionary of performance metrics
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        # Compute multiple metrics with Fβ=1.3 bias towards recall
        return {
            'precision': precision_score(labels, preds, pos_label=1),
            'recall': recall_score(labels, preds, pos_label=1),
            'f1_binary': f1_score(labels, preds, pos_label=1),
            'f1_beta': fbeta_score(labels, preds, beta=1.3, pos_label=1)
        }
    
    def adaptive_threshold_search(
        self, 
        eval_dataset, 
        start_threshold: float = 0.4, 
        end_threshold: float = 0.6, 
        step: float = 0.01
    ):
        """
        Search for optimal classification threshold
        
        Args:
            eval_dataset: Validation dataset
            start_threshold (float): Starting threshold
            end_threshold (float): Ending threshold
            step (float): Threshold increment
        
        Returns:
            Best threshold and corresponding F1β score
        """
        predictions = self.predict(eval_dataset)
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        labels = predictions.label_ids
        
        best_threshold = 0.5
        best_fbeta = 0
        
        for thresh in np.arange(start_threshold, end_threshold, step):
            preds = (probs >= thresh).astype(int)
            fbeta = fbeta_score(labels, preds, beta=1.3, pos_label=1)
            
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = thresh
        
        return best_threshold, best_fbeta