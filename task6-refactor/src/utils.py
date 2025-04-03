# src/utils.py
import os
import logging
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def set_global_seed(seed: int = 42):
    """
    Set global random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        Dictionary of configuration parameters
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_file: str = 'experiment.log', level=logging.INFO):
    """
    Configure logging for the experiment
    
    Args:
        log_file (str): Path to log file
        level (int): Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_learning_curves(
    log_history: list,
    eval_history: list,
    output_dir: str = 'plots'
):
    """
    Plot learning curves from Trainer logs

    Args:
        log_history (list): List of trainer logs (each is a dict)
        eval_history (list): Same as log_history, used for eval points
        output_dir (str): Where to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract training loss per epoch
    train_epochs = [log["epoch"] for log in log_history if "loss" in log and "epoch" in log]
    train_loss = [log["loss"] for log in log_history if "loss" in log]

    # Extract eval loss and f1 per epoch
    eval_epochs = [log["epoch"] for log in eval_history if "eval_loss" in log and "epoch" in log]
    eval_loss = [log["eval_loss"] for log in eval_history if "eval_loss" in log]
    eval_f1 = [log.get("eval_f1_beta", None) for log in eval_history if "eval_loss" in log]

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss')
    plt.plot(eval_epochs, eval_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1β
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_f1, label='F1β Score')
    plt.title('Validation F1β Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1β')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()


def plot_confusion_matrix(
    confusion_matrix, 
    class_names=['Negative', 'Positive'], 
    output_dir: str = 'plots'
):
    """
    Plot confusion matrix heatmap
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (List[str]): Names of classes
        output_dir (str): Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()