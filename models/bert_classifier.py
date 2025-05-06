import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from config import Config

class DisasterBERTClassifier:
    def __init__(self, num_classes, stage="stage_1"):
        self.num_classes = num_classes
        self.stage = stage
        self.bert = TFBertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self._build_model()
        
    def _build_model(self):
        # Input layers
        input_ids = tf.keras.layers.Input(
            shape=(Config.MAX_SEQ_LENGTH,), 
            dtype=tf.int32, 
            name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(Config.MAX_SEQ_LENGTH,), 
            dtype=tf.int32, 
            name="attention_mask"
        )
        
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        
        # Classification head - based on paper's best architecture
        if self.stage == "stage_1":
            # Binary classification
            x = BatchNormalization()(pooled_output)
            x = Dropout(0.1)(x)
            x = Dense(8, activation='relu')(x)
            x = Dropout(0.1)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            loss = BinaryCrossentropy()
        else:
            # Multi-class classification
            x = BatchNormalization()(pooled_output)
            x = Dropout(0.1)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            loss = CategoricalCrossentropy()
        
        # Create model
        self.model = Model(
            inputs=[input_ids, attention_mask], 
            outputs=outputs
        )
        
        # Compile model
        optimizer = Adam(learning_rate=Config.LEARNING_RATE)
        metrics = [
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def train(self, train_data, val_data=None, epochs=Config.EPOCHS):
        """Train the model"""
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=Config.BATCH_SIZE
        )
        return history
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        return self.model.evaluate(test_data)
    
    def predict(self, input_data):
        """Make predictions"""
        return self.model.predict(input_data)