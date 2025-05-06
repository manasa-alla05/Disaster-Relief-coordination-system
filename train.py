import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing.preprocessor import DisasterPreprocessor
from models.bert_classifier import DisasterBERTClassifier
from config import Config

def main():
    # Initialize preprocessor
    preprocessor = DisasterPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_and_preprocess_data(Config.DATA_PATH)
    
    # Split data (Stage I - binary classification)
    train_df, test_df = train_test_split(
        df, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_df, 
        test_size=Config.VAL_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    # Tokenize data
    train_encodings = preprocessor.tokenize_texts(train_df['cleaned_text'])
    val_encodings = preprocessor.tokenize_texts(val_df['cleaned_text'])
    test_encodings = preprocessor.tokenize_texts(test_df['cleaned_text'])
    
    # Convert labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        y_train
    )).batch(Config.BATCH_SIZE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask']
        },
        y_val
    )).batch(Config.BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        y_test
    )).batch(Config.BATCH_SIZE)
    
    # Initialize and train model
    model = DisasterBERTClassifier(
        num_classes=1,  # Binary classification for Stage I
        stage="stage_1"
    )
    
    # Train model
    history = model.train(train_dataset, val_dataset)
    
    # Evaluate model
    results = model.evaluate(test_dataset)
    print(f"Test results: {results}")
    
    # Save model
    model.model.save_pretrained(os.path.join("models", "stage_1_model"))
    
    # For Stage II, you would filter disaster-related tweets and repeat
    # the process with the multi-class classifier

if __name__ == "__main__":
    main()