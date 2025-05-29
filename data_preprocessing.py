import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


class DataPreprocessor:
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.text_vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def load_nusax_data(self):
        try:
            dataset = load_dataset("indonlp/NusaX-senti", "ind")
            
            train_data = dataset['train']
            val_data = dataset['validation']
            test_data = dataset['test']
        
            train_df = pd.DataFrame({
                'text': train_data['text'],
                'label': train_data['label']
            })
            
            val_df = pd.DataFrame({
                'text': val_data['text'],
                'label': val_data['label']
            })
            
            test_df = pd.DataFrame({
                'text': test_data['text'],
                'label': test_data['label']
            })
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"Error loading NusaX dataset: {e}")
            print("Using dummy data for testing...")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        texts = [
            "Saya sangat senang dengan produk ini",
            "Produk ini buruk sekali",
            "Biasa saja, tidak istimewa",
            "Luar biasa bagus!",
            "Sangat mengecewakan",
            "Cukup baik untuk harga segini",
            "Tidak memuaskan sama sekali",
            "Sangat puas dengan pelayanan",
            "Kualitas produk standar",
            "Sempurna!"
        ] * 100  # Repeat to have enough data
        
        labels = [1, 0, 2, 1, 0, 2, 0, 1, 2, 1] * 100  # 0: negative, 1: positive, 2: neutral
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Split into train, val, test
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        return train_df, val_df, test_df
    
    def preprocess_with_keras_tokenizer(self, train_texts, val_texts, test_texts):
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(train_texts)
        
        # Convert texts to sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        val_sequences = self.tokenizer.texts_to_sequences(val_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        
        # Pad sequences
        train_padded = pad_sequences(train_sequences, maxlen=self.max_sequence_length, padding='post')
        val_padded = pad_sequences(val_sequences, maxlen=self.max_sequence_length, padding='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_sequence_length, padding='post')
        
        return train_padded, val_padded, test_padded
    
    def preprocess_with_text_vectorization(self, train_texts, val_texts, test_texts):
        # Create TextVectorization layer
        self.text_vectorizer = TextVectorization(
            max_tokens=self.max_vocab_size,
            output_sequence_length=self.max_sequence_length,
            output_mode='int'
        )
        
        # Adapt on training data
        self.text_vectorizer.adapt(train_texts)
        
        # Transform texts
        train_vectorized = self.text_vectorizer(train_texts)
        val_vectorized = self.text_vectorizer(val_texts)
        test_vectorized = self.text_vectorizer(test_texts)
        
        return train_vectorized, val_vectorized, test_vectorized
    
    def encode_labels(self, train_labels, val_labels, test_labels):
        print(train_labels)
        self.label_encoder.fit(train_labels)
        
        # Transform all labels
        train_encoded = self.label_encoder.transform(train_labels)
        val_encoded = self.label_encoder.transform(val_labels)
        test_encoded = self.label_encoder.transform(test_labels)
        
        return train_encoded, val_encoded, test_encoded
    
    def get_vocab_size(self):
        if self.tokenizer:
            return len(self.tokenizer.word_index) + 1
        elif self.text_vectorizer:
            return self.text_vectorizer.vocabulary_size()
        else:
            return self.max_vocab_size
    
    def get_num_classes(self):
        return len(self.label_encoder.classes_)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.load_nusax_data()
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Preprocess with Keras tokenizer
    train_padded, val_padded, test_padded = preprocessor.preprocess_with_keras_tokenizer(
        train_df['text'].values, val_df['text'].values, test_df['text'].values
    )
    
    # Encode labels
    train_labels, val_labels, test_labels = preprocessor.encode_labels(
        train_df['label'].values, val_df['label'].values, test_df['label'].values
    )
    
    print(f"Processed train data shape: {train_padded.shape}")
    print(f"Vocabulary size: {preprocessor.get_vocab_size()}")
    print(f"Number of classes: {preprocessor.get_num_classes()}")
