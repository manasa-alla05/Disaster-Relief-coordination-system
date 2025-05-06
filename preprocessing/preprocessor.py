import re
import pandas as pd
from transformers import BertTokenizer
from config import Config
import numpy as np

class DisasterPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        
    def preprocess_text(self, text, preprocessing_type="PRE-2"):
        """
        Preprocess text according to the paper's specifications
        PRE-1: Traditional preprocessing
        PRE-2: BERT-specific preprocessing (lighter)
        """
        if preprocessing_type == "PRE-1":
            # Handle emoticons
            text = self._replace_emoticons(text)
            # Remove stock market tickers
            text = re.sub(r'\$\w*\b', '', text)
            # Remove old-style retweet text
            text = re.sub(r'^RT[\s]+', '', text)
            # Remove hyperlinks
            text = re.sub(r'https?:\/\/\S+', '', text)
            # Remove words with numbers
            text = re.sub(r'\w*\d\w*', '', text)
            # Change "n't" to "not"
            text = re.sub(r"n't\b", " not", text)
            # Remove @name
            text = re.sub(r'@\w+\b', '', text)
            # Replace &amp; with &
            text = re.sub(r'&amp;', '&', text)
            # Remove trailing whitespace
            text = text.strip()
            
        elif preprocessing_type == "PRE-2":
            # BERT-specific preprocessing (lighter)
            # Handle emoticons
            text = self._replace_emoticons(text)
            # Remove @name
            text = re.sub(r'@\w+\b', '', text)
            # Remove stock market tickers
            text = re.sub(r'\$\w*\b', '', text)
            # Remove old-style retweet text
            text = re.sub(r'^RT[\s]+', '', text)
            # Remove hyperlinks
            text = re.sub(r'https?:\/\/\S+', '', text)
            # Remove words with numbers
            text = re.sub(r'\w*\d\w*', '', text)
            # Replace &amp; with &
            text = re.sub(r'&amp;', '&', text)
            # Remove trailing whitespace
            text = text.strip()
            
        return text
    
    def _replace_emoticons(self, text):
        # Basic emoticon replacement (can be expanded)
        emoticon_map = {
            ":)": "smiling face",
            ":(": "sad face",
            "<3": "heart",
            ":D": "big smile",
            ";)": "winking face"
        }
        for emoticon, replacement in emoticon_map.items():
            text = text.replace(emoticon, replacement)
        return text
    
    def tokenize_texts(self, texts):
        """Tokenize texts for BERT input"""
        return self.tokenizer(
            texts.tolist(),
            max_length=Config.MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
    
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess raw data"""
        df = pd.read_csv(filepath)
        
        # Basic cleaning
        df = df.drop_duplicates()
        df = df.dropna(subset=['text'])
        
        # Apply preprocessing
        df['cleaned_text'] = df['text'].apply(
            lambda x: self.preprocess_text(x, preprocessing_type="PRE-2")
        )
        
        return df