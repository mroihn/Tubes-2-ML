import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import pickle


class LSTMForwardPropagation:
    
    def __init__(self):
        self.weights = {}
        self.embedding_weights = None
        self.vocab_size = None
        self.embedding_dim = None
        self.lstm_units = None
        self.num_classes = None
        
    def load_keras_weights(self, model_path):
        model = load_model(model_path)
        
        # Extract embedding weights
        embedding_layer = model.get_layer('embedding')
        self.embedding_weights = embedding_layer.get_weights()[0]
        self.vocab_size, self.embedding_dim = self.embedding_weights.shape
        
        # Extract LSTM weights
        lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name.lower()]
        self.weights['lstm_layers'] = []
        
        for lstm_layer in lstm_layers:
            lstm_weights = lstm_layer.get_weights()
            if len(lstm_weights) == 3:  # kernel, recurrent_kernel, bias
                kernel, recurrent_kernel, bias = lstm_weights
                self.lstm_units = kernel.shape[1] // 4  # 4 gates in LSTM
                
                # Split weights for different gates (input, forget, cell, output)
                W_i, W_f, W_c, W_o = np.split(kernel, 4, axis=1)
                U_i, U_f, U_c, U_o = np.split(recurrent_kernel, 4, axis=1)
                b_i, b_f, b_c, b_o = np.split(bias, 4, axis=0)
                
                layer_weights = {
                    'W_i': W_i, 'W_f': W_f, 'W_c': W_c, 'W_o': W_o,
                    'U_i': U_i, 'U_f': U_f, 'U_c': U_c, 'U_o': U_o,
                    'b_i': b_i, 'b_f': b_f, 'b_c': b_c, 'b_o': b_o
                }
                self.weights['lstm_layers'].append(layer_weights)
        
        # Extract dense layer weights
        dense_layer = model.get_layer('dense_output')
        dense_weights = dense_layer.get_weights()
        self.weights['dense'] = {
            'W': dense_weights[0],
            'b': dense_weights[1]
        }
        self.num_classes = dense_weights[0].shape[1]
        
        print(f"Loaded weights from {model_path}")
        print(f"Vocab size: {self.vocab_size}, Embedding dim: {self.embedding_dim}")
        print(f"LSTM units: {self.lstm_units}, Num classes: {self.num_classes}")
        print(f"Number of LSTM layers: {len(self.weights['lstm_layers'])}")
        
        return model
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -250, 250)) 
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def embedding_forward(self, input_ids):
        # input_ids shape: (batch_size, sequence_length)
        # output shape: (batch_size, sequence_length, embedding_dim)
        return self.embedding_weights[input_ids]
    
    def lstm_cell_forward(self, x_t, h_prev, c_prev, weights):
        # x_t shape: (batch_size, input_dim)
        # h_prev shape: (batch_size, hidden_dim)
        # c_prev shape: (batch_size, hidden_dim)
        
        # Input gate
        i_t = self.sigmoid(np.dot(x_t, weights['W_i']) + np.dot(h_prev, weights['U_i']) + weights['b_i'])
        
        # Forget gate
        f_t = self.sigmoid(np.dot(x_t, weights['W_f']) + np.dot(h_prev, weights['U_f']) + weights['b_f'])
        
        # Candidate values
        c_tilde = self.tanh(np.dot(x_t, weights['W_c']) + np.dot(h_prev, weights['U_c']) + weights['b_c'])
        
        # Output gate
        o_t = self.sigmoid(np.dot(x_t, weights['W_o']) + np.dot(h_prev, weights['U_o']) + weights['b_o'])
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t
    
    def lstm_layer_forward(self, inputs, layer_weights):
        # inputs shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_length, input_dim = inputs.shape
        
        # Initialize hidden and cell states
        h_t = np.zeros((batch_size, self.lstm_units))
        c_t = np.zeros((batch_size, self.lstm_units))
        
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            x_t = inputs[:, t, :]  # (batch_size, input_dim)
            h_t, c_t = self.lstm_cell_forward(x_t, h_t, c_t, layer_weights)
            outputs.append(h_t)
        
        # Stack outputs: (batch_size, sequence_length, hidden_dim)
        outputs = np.stack(outputs, axis=1)
        
        return outputs, h_t, c_t
    
    def dense_forward(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # Apply linear transformation
        logits = np.dot(inputs, self.weights['dense']['W']) + self.weights['dense']['b']
        
        # Apply softmax
        probabilities = self.softmax(logits)
        
        return probabilities
    
    def forward_propagation(self, input_ids):
        # Step 1: Embedding
        embedded = self.embedding_forward(input_ids)
        
        # Step 2: LSTM layers
        current_input = embedded
        for i, layer_weights in enumerate(self.weights['lstm_layers']):
            outputs, final_h, final_c = self.lstm_layer_forward(current_input, layer_weights)
            
            # For stacked LSTMs, use all outputs as input to next layer
            # For final layer, use only the last output
            if i < len(self.weights['lstm_layers']) - 1:
                current_input = outputs
            else:
                # Use final hidden state for classification
                lstm_output = final_h
        
        # Step 3: Dense layer
        predictions = self.dense_forward(lstm_output)
        
        return predictions
    
    def predict(self, X):
        predictions = self.forward_propagation(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes, predictions
    
    def evaluate(self, X_test, y_test):
        predicted_classes, probabilities = self.predict(X_test)
        f1_macro = f1_score(y_test, predicted_classes, average='macro')
        return f1_macro, predicted_classes


class BidirectionalLSTMForwardPropagation(LSTMForwardPropagation):
    
    def load_keras_weights(self, model_path):
        model = load_model(model_path)
        
        # Extract embedding weights
        embedding_layer = model.get_layer('embedding')
        self.embedding_weights = embedding_layer.get_weights()[0]
        self.vocab_size, self.embedding_dim = self.embedding_weights.shape
        
        # Extract bidirectional LSTM weights
        bidirectional_layers = [layer for layer in model.layers if 'bidirectional' in layer.name.lower()]
        self.weights['lstm_layers'] = []
        
        for bid_layer in bidirectional_layers:
            # Get forward and backward LSTM weights
            forward_lstm = bid_layer.forward_layer
            backward_lstm = bid_layer.backward_layer
            
            forward_weights = forward_lstm.get_weights()
            backward_weights = backward_lstm.get_weights()
            
            # Process forward weights
            if len(forward_weights) == 3:
                kernel, recurrent_kernel, bias = forward_weights
                self.lstm_units = kernel.shape[1] // 4
                
                W_i_f, W_f_f, W_c_f, W_o_f = np.split(kernel, 4, axis=1)
                U_i_f, U_f_f, U_c_f, U_o_f = np.split(recurrent_kernel, 4, axis=1)
                b_i_f, b_f_f, b_c_f, b_o_f = np.split(bias, 4, axis=0)
                
                forward_weights_dict = {
                    'W_i': W_i_f, 'W_f': W_f_f, 'W_c': W_c_f, 'W_o': W_o_f,
                    'U_i': U_i_f, 'U_f': U_f_f, 'U_c': U_c_f, 'U_o': U_o_f,
                    'b_i': b_i_f, 'b_f': b_f_f, 'b_c': b_c_f, 'b_o': b_o_f
                }
            
            # Process backward weights
            if len(backward_weights) == 3:
                kernel, recurrent_kernel, bias = backward_weights
                
                W_i_b, W_f_b, W_c_b, W_o_b = np.split(kernel, 4, axis=1)
                U_i_b, U_f_b, U_c_b, U_o_b = np.split(recurrent_kernel, 4, axis=1)
                b_i_b, b_f_b, b_c_b, b_o_b = np.split(bias, 4, axis=0)
                
                backward_weights_dict = {
                    'W_i': W_i_b, 'W_f': W_f_b, 'W_c': W_c_b, 'W_o': W_o_b,
                    'U_i': U_i_b, 'U_f': U_f_b, 'U_c': U_c_b, 'U_o': U_o_b,
                    'b_i': b_i_b, 'b_f': b_f_b, 'b_c': b_c_b, 'b_o': b_o_b
                }
            
            layer_weights = {
                'forward': forward_weights_dict,
                'backward': backward_weights_dict
            }
            self.weights['lstm_layers'].append(layer_weights)
        
        # Extract dense layer weights
        dense_layer = model.get_layer('dense_output')
        dense_weights = dense_layer.get_weights()
        self.weights['dense'] = {
            'W': dense_weights[0],
            'b': dense_weights[1]
        }
        self.num_classes = dense_weights[0].shape[1]
        
        print(f"Loaded bidirectional weights from {model_path}")
        return model
    
    def bidirectional_lstm_layer_forward(self, inputs, layer_weights):
        # Forward direction
        forward_outputs, forward_h, forward_c = self.lstm_layer_forward(inputs, layer_weights['forward'])
        
        # Backward direction (reverse the sequence)
        backward_inputs = inputs[:, ::-1, :]  # Reverse time dimension
        backward_outputs, backward_h, backward_c = self.lstm_layer_forward(backward_inputs, layer_weights['backward'])
        backward_outputs = backward_outputs[:, ::-1, :]  # Reverse back
        
        # Concatenate forward and backward outputs
        combined_outputs = np.concatenate([forward_outputs, backward_outputs], axis=-1)
        combined_final_h = np.concatenate([forward_h, backward_h], axis=-1)
        
        return combined_outputs, combined_final_h
    
    def forward_propagation(self, input_ids):
        # Step 1: Embedding
        embedded = self.embedding_forward(input_ids)
        
        # Step 2: Bidirectional LSTM layers
        current_input = embedded
        for i, layer_weights in enumerate(self.weights['lstm_layers']):
            outputs, final_h = self.bidirectional_lstm_layer_forward(current_input, layer_weights)
            
            if i < len(self.weights['lstm_layers']) - 1:
                current_input = outputs
            else:
                lstm_output = final_h
        
        # Step 3: Dense layer
        predictions = self.dense_forward(lstm_output)
        
        return predictions


def compare_implementations(keras_model_path, X_test, y_test, is_bidirectional=False):
    
    keras_model = load_model(keras_model_path)
    keras_predictions = keras_model.predict(X_test)
    keras_classes = np.argmax(keras_predictions, axis=1)
    keras_f1 = f1_score(y_test, keras_classes, average='macro')
    
    # Manual implementation
    if is_bidirectional:
        manual_model = BidirectionalLSTMForwardPropagation()
    else:
        manual_model = LSTMForwardPropagation()
    
    manual_model.load_keras_weights(keras_model_path)
    manual_f1, manual_classes = manual_model.evaluate(X_test, y_test)
    
    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    print(f"Keras Model F1 Score: {keras_f1:.4f}")
    print(f"Manual Implementation F1 Score: {manual_f1:.4f}")
    print(f"Difference: {abs(keras_f1 - manual_f1):.4f}")
    
    # Check prediction agreement
    agreement = np.mean(keras_classes == manual_classes)
    print(f"Prediction Agreement: {agreement:.4f}")
    
    return keras_f1, manual_f1, agreement


