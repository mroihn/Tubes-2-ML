import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os


class LSTMClassifier:
    
    def __init__(self, vocab_size, embedding_dim=128, max_sequence_length=100, num_classes=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, num_lstm_layers=1, lstm_units=64, bidirectional=False, dropout_rate=0.2):
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            name='embedding'
        ))
        
        # LSTM layers
        for i in range(num_lstm_layers):
            return_sequences = (i < num_lstm_layers - 1) 
            
            if bidirectional:
                model.add(Bidirectional(LSTM(
                    lstm_units,
                    return_sequences=return_sequences,
                    name=f'bidirectional_lstm_{i+1}'
                )))
            else:
                model.add(LSTM(
                    lstm_units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}'
                ))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Dense output layer
        model.add(Dense(self.num_classes, activation='softmax', name='dense_output'))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=1):
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate macro F1 score
        f1_macro = f1_score(y_test, y_pred_classes, average='macro')
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        return f1_macro, y_pred_classes
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def save_weights(self, filepath):
        if self.model is None:
            raise ValueError("No model to save weights from")
        
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model must be built before loading weights")
        
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")
    
    def plot_training_history(self, save_path=None):
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model must be built first")
        
        return self.model.summary()


class HyperparameterExperiment:
    
    def __init__(self, vocab_size, embedding_dim=128, max_sequence_length=100, num_classes=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.results = {}
    
    def experiment_lstm_layers(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                              layer_configs=[1, 2, 3], epochs=10):
        print("=== Experimenting with LSTM Layer Numbers ===")
        
        for num_layers in layer_configs:
            print(f"\nTraining model with {num_layers} LSTM layer(s)...")
            
            # Build and train model
            classifier = LSTMClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                max_sequence_length=self.max_sequence_length,
                num_classes=self.num_classes
            )
            
            classifier.build_model(num_lstm_layers=num_layers, lstm_units=64)
            classifier.compile_model()
            history = classifier.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
            
            # Evaluate
            f1_score, _ = classifier.evaluate(X_test, y_test)
            
            # Store results
            self.results[f'layers_{num_layers}'] = {
                'model': classifier,
                'history': history,
                'f1_score': f1_score,
                'config': {'num_layers': num_layers, 'lstm_units': 64, 'bidirectional': False}
            }
            
            print(f"Macro F1 Score: {f1_score:.4f}")
        
        return self.results
    
    def experiment_lstm_units(self, X_train, y_train, X_val, y_val, X_test, y_test,
                             unit_configs=[32, 64, 128], epochs=10):
        print("=== Experimenting with LSTM Unit Numbers ===")
        
        for units in unit_configs:
            print(f"\nTraining model with {units} LSTM units...")
            
            # Build and train model
            classifier = LSTMClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                max_sequence_length=self.max_sequence_length,
                num_classes=self.num_classes
            )
            
            classifier.build_model(num_lstm_layers=1, lstm_units=units)
            classifier.compile_model()
            history = classifier.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
            
            # Evaluate
            f1_score, _ = classifier.evaluate(X_test, y_test)
            
            # Store results
            self.results[f'units_{units}'] = {
                'model': classifier,
                'history': history,
                'f1_score': f1_score,
                'config': {'num_layers': 1, 'lstm_units': units, 'bidirectional': False}
            }
            
            print(f"Macro F1 Score: {f1_score:.4f}")
        
        return self.results
    
    def experiment_bidirectional(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10):
        print("=== Experimenting with LSTM Direction ===")
        
        for bidirectional in [False, True]:
            direction = "Bidirectional" if bidirectional else "Unidirectional"
            print(f"\nTraining {direction} LSTM model...")
            
            # Build and train model
            classifier = LSTMClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                max_sequence_length=self.max_sequence_length,
                num_classes=self.num_classes
            )
            
            classifier.build_model(num_lstm_layers=1, lstm_units=64, bidirectional=bidirectional)
            classifier.compile_model()
            history = classifier.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
            
            # Evaluate
            f1_score, _ = classifier.evaluate(X_test, y_test)
            
            # Store results
            self.results[f'direction_{direction.lower()}'] = {
                'model': classifier,
                'history': history,
                'f1_score': f1_score,
                'config': {'num_layers': 1, 'lstm_units': 64, 'bidirectional': bidirectional}
            }
            
            print(f"Macro F1 Score: {f1_score:.4f}")
        
        return self.results
    
    def plot_comparison(self, experiment_type, save_path=None):
        if experiment_type == 'layers':
            configs = [k for k in self.results.keys() if k.startswith('layers_')]
            title = "LSTM Layers Comparison"
        elif experiment_type == 'units':
            configs = [k for k in self.results.keys() if k.startswith('units_')]
            title = "LSTM Units Comparison"
        elif experiment_type == 'direction':
            configs = [k for k in self.results.keys() if k.startswith('direction_')]
            title = "LSTM Direction Comparison"
        else:
            raise ValueError("Invalid experiment type")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for config in configs:
            history = self.results[config]['history']
            label = config.replace('_', ' ').title()
            
            ax1.plot(history.history['loss'], label=f'{label} - Train')
            ax1.plot(history.history['val_loss'], '--', label=f'{label} - Val')
        
        ax1.set_title(f'{title} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        for config in configs:
            history = self.results[config]['history']
            label = config.replace('_', ' ').title()
            
            ax2.plot(history.history['accuracy'], label=f'{label} - Train')
            ax2.plot(history.history['val_accuracy'], '--', label=f'{label} - Val')
        
        ax2.set_title(f'{title} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"{'Configuration':<20} {'F1 Score':<10}")
        print("-" * 30)
        
        for config_name, result in self.results.items():
            print(f"{config_name:<20} {result['f1_score']:<10.4f}")


