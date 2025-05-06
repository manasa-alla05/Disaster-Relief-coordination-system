import os

class Config:
    # Data paths
    DATA_PATH = os.path.join("data", "data.csv")
    PROCESSED_PATH = os.path.join("data", "processed")
    SPLITS_PATH = os.path.join("data", "splits")
    
    # BERT model parameters
    BERT_MODEL_NAME = "bert-base-uncased"
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    
    # Training parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Stage I (Binary classification)
    STAGE_I_CLASSES = ["non_disaster", "disaster"]
    
    # Stage II (Multi-class classification)
    STAGE_II_CLASSES = [
        "injured_or_dead",
        "missing_trapped_found",
        "infrastructure_damage",
        "shelter_needs_supplies",
        "volunteering",
        "other_relevant_info"
    ]