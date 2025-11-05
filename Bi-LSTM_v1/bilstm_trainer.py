import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# TensorFlow and Keras Modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Configuration ---

# Directory containing .pkl data 
DATA_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\data_v1' 
# Directory where the trained model will be saved (Bi-LSTM_v1 folder)
MODEL_SAVE_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\Bi-LSTM_v1' 

# Language configurations
language_configs = {
    # Feature size, for LSTM ,is used  as 'sequence length'
    'EN': {'feature_size': 42}, 
    'AR': {'feature_size': 42}, 
    'TR': {'feature_size': 84}, 
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50           
LSTM_UNITS = 128      # Number of units in Bi-LSTM layer
BATCH_SIZE = 64
DROPOUT_RATE = 0.3
VERBOSE = 1

# --- FUNCTIONS ---

def load_data(lang, feature_size):
    """
    Loads feature vectors and labels for the specified language,
    prepares data for LSTM, and splits into training/test sets.
    """
    print(f"\n--- Loading Data for {lang} Language ---")
    
    X_path = os.path.join(DATA_DIR, f'X_{lang}.pkl')
    y_path = os.path.join(DATA_DIR, f'y_{lang}.pkl')
    label_map_path = os.path.join(DATA_DIR, f'label_map_{lang}.pkl')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path) or not os.path.exists(label_map_path):
        print(f"ERROR: {lang} data or label map not found. Please check path: {DATA_DIR}")
        print("Please run 'extract_and_save_data.py' first to create .pkl files.")
        return None, None, None, None, None, None

    try:
        # Data Loading
        with open(X_path, 'rb') as f:
            X = pickle.load(f)
        with open(y_path, 'rb') as f:
            y = pickle.load(f)
        with open(label_map_path, 'rb') as f:
            labels_map = pickle.load(f)
            
        print(f"Loaded: X size {X.shape}, y size {y.shape}")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Split into Training and Test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Reshape Data for LSTM
        # LSTM input should be (samples, timesteps, features)
        # Here we treat each coordinate as a "timestep"
        # Input shape: (samples, feature_size, 1) -> (42 or 84 timesteps, 1 feature)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # One-Hot Encode Labels
        num_classes = len(np.unique(y))
        y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
        
        print(f"Ready: Training data {X_train.shape}, Test data {X_test.shape}")
        print(f"Total number of classes: {num_classes}")
        
        return X_train, X_test, y_train_one_hot, y_test_one_hot, num_classes, labels_map

    except Exception as e:
        print(f"Error while loading or preparing data: {e}")
        return None, None, None, None, None, None

def create_bilstm_model(input_shape, num_classes):
    """
    Defines the Bidirectional LSTM (Bi-LSTM) model.
    """
    model = Sequential()
    
    # Bi-LSTM Layer: Processes data in both forward and backward directions
    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=False), input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    
    # Dense Layers
    model.add(Dense(LSTM_UNITS, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    
    # Output Layer: Number of neurons equal to class count, softmax activation
    model.add(Dense(num_classes, activation='softmax'))
    
    # Model compilation
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def save_model_and_labels(lang, model, labels_map):
    """Saves the trained model and label map."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Save file name as 'bilstm'
    model_path = os.path.join(MODEL_SAVE_DIR, f'bilstm_{lang}.h5')
    label_map_path = os.path.join(MODEL_SAVE_DIR, f'label_map_{lang}.pkl')
    
    # Save model in H5 format
    model.save(model_path)
    
    # Save label map
    with open(label_map_path, 'wb') as f:
        pickle.dump(labels_map, f)
        
    print(f"\nModel and Labels successfully saved: {model_path}")

def train_and_evaluate_bilstm():
    """Trains and evaluates Bi-LSTM models for all languages."""
    
    # GPU/CPU settings
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU.")
    else:
        print("Using CPU.")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    for lang, config in language_configs.items():
        feature_size = config['feature_size']
        
        # 1. Load and Prepare Data
        X_train, X_test, y_train_one_hot, y_test_one_hot, num_classes, labels_map = load_data(lang, feature_size)
        
        if X_train is None:
            continue
            
        # 2. Create Model
        input_shape = (X_train.shape[1], 1)
        model = create_bilstm_model(input_shape, num_classes)
        
        print(f"\n--- {lang} Model Summary ---")
        model.summary()

        # 3. Train Model
        print(f"\n--- Training {lang} Model (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}) ---")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train_one_hot,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test_one_hot),
            verbose=VERBOSE
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\nTraining Completed. Duration: {training_duration:.2f} seconds")

        # 4. Save Model
        save_model_and_labels(lang, model, labels_map)

        # 5. Evaluation and Reporting
        print(f"\n--- {lang} Model Evaluation and Reporting ---")
        
        y_pred_one_hot = model.predict(X_test)
        y_test_labels = np.argmax(y_test_one_hot, axis=1)
        y_pred_labels = np.argmax(y_pred_one_hot, axis=1)
        
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        id_to_label = {v: k for k, v in labels_map.items()}
        target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
        
        report = classification_report(y_test_labels, y_pred_labels, target_names=target_names, zero_division=0)
        print("\nClassification Report:\n", report)
        
        # Save report
        with open(os.path.join(MODEL_SAVE_DIR, f'bilstm_report_{lang}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Model: Bi-LSTM\n")
            f.write(f"Language: {lang}\n")
            f.write(f"Training Duration: {training_duration:.2f} seconds\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
            
        print(f"{lang} report saved.")
        print("-" * 50)


if __name__ == '__main__':
    train_and_evaluate_bilstm()