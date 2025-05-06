import tensorflow as tf
from transformers import BertTokenizer
from models.bert_classifier import DisasterBERTClassifier
from preprocessing.preprocessor import DisasterPreprocessor
from config import Config

class DisasterPredictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.preprocessor = DisasterPreprocessor()
        self.model = tf.keras.models.load_model(model_path)
    
    def predict(self, text):
        """Predict class for input text"""
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess_text(text)
        
        # Tokenize
        encodings = self.tokenizer(
            [cleaned_text],
            max_length=Config.MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
        
        # Make prediction
        preds = self.model.predict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        })
        
        return preds[0]

# Example usage
if __name__ == "__main__":
    predictor = DisasterPredictor(os.path.join("models", "stage_1_model"))
    sample_text = "Earthquake in our area! Need medical help urgently #disaster"
    prediction = predictor.predict(sample_text)
    print(f"Prediction: {prediction}")