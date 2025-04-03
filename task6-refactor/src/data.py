# src/data.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, List
import nltk
import re

class DataProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Perform basic text cleaning for medical social media text
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits 
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
        
        Returns:
            Tuple of training and test DataFrames
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Rename columns for consistency
        train_df = train_df.rename(columns={"labels": "label"})
        test_df = test_df.rename(columns={"labels": "label"})
        
        # Optional: Clean text
        train_df['text'] = train_df['text'].apply(DataProcessor.clean_text)
        test_df['text'] = test_df['text'].apply(DataProcessor.clean_text)
        
        return train_df, test_df
    
    @staticmethod
    def stratified_kfold_split(
        df: pd.DataFrame, 
        n_splits: int = 5, 
        random_state: int = 42
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Perform stratified K-Fold cross-validation splits
        
        Args:
            df (pd.DataFrame): Input DataFrame
            n_splits (int): Number of splits
            random_state (int): Random seed for reproducibility
        
        Returns:
            List of (train, validation) DataFrame pairs
        """
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        
        splits = []
        for train_idx, valid_idx in skf.split(df['text'], df['label']):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]
            splits.append((train_df, valid_df))
        
        return splits
    
    @staticmethod
    def data_augmentation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data augmentation via paraphrasing
        
        Note: This is a placeholder. Actual implementation 
        would require advanced NLP techniques like back-translation
        or using paraphrasing models.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Augmented DataFrame
        """
        # TODO: Implement sophisticated data augmentation
        # Placeholder simple augmentation
        augmented_df = df.copy()
        minority_class = df[df['label'] == 1]
        
        # Simple duplication of minority class
        augmented_df = pd.concat([
            augmented_df, 
            minority_class.sample(n=len(minority_class)//2, replace=True)
        ])
        
        return augmented_df