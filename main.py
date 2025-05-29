import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from lstm_classifier import LSTMClassifier, HyperparameterExperiment
from forward_propagation import LSTMForwardPropagation, BidirectionalLSTMForwardPropagation, compare_implementations


def main():
    print("=== LSTM Forward Propagation Implementation ===")
    print("Starting comprehensive LSTM text classification experiment...\n")
    
    # Create directories for saving models and plots
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ============= 1. DATA PREPROCESSING =============
    print("1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor(max_vocab_size=5000, max_sequence_length=50)
    
    # Load dataset
    train_df, val_df, test_df = preprocessor.load_nusax_data()
    print(f"Dataset loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Preprocess texts
    X_train, X_val, X_test = preprocessor.preprocess_with_keras_tokenizer(
        train_df['text'].values, val_df['text'].values, test_df['text'].values
    )
    
    # Encode labels
    y_train, y_val, y_test = preprocessor.encode_labels(
        train_df['label'].values, val_df['label'].values, test_df['label'].values
    )
    
    vocab_size = preprocessor.get_vocab_size()
    num_classes = preprocessor.get_num_classes()
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Sequence length: {preprocessor.max_sequence_length}")
    
    # ============= 2. BASELINE MODEL TRAINING =============
    print("\n2. Training baseline LSTM model...")
    
    baseline_classifier = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=64,
        max_sequence_length=preprocessor.max_sequence_length,
        num_classes=num_classes
    )
    
    baseline_classifier.build_model(num_lstm_layers=1, lstm_units=32, bidirectional=False)
    baseline_classifier.compile_model()
    baseline_classifier.get_model_summary()
    
    # Train baseline model
    baseline_history = baseline_classifier.train(
        X_train, y_train, X_val, y_val, epochs=15, batch_size=32
    )
    
    # Evaluate baseline
    baseline_f1, _ = baseline_classifier.evaluate(X_test, y_test)
    print(f"Baseline F1 Score: {baseline_f1:.4f}")
    
    # Save baseline model
    baseline_classifier.save_model('models/baseline_model.h5')
    baseline_classifier.save_weights('models/baseline_weights.h5')
    
    # Plot training history
    baseline_classifier.plot_training_history('plots/baseline_training.png')
    
    # ============= 3. HYPERPARAMETER EXPERIMENTS =============
    print("\n3. Starting hyperparameter experiments...")
    
    experiment = HyperparameterExperiment(
        vocab_size=vocab_size,
        embedding_dim=64,
        max_sequence_length=preprocessor.max_sequence_length,
        num_classes=num_classes
    )
    
    # A. Experiment with number of LSTM layers
    print("\n3A. Experimenting with LSTM layer numbers...")
    experiment.experiment_lstm_layers(
        X_train, y_train, X_val, y_val, X_test, y_test,
        layer_configs=[1, 2, 3], epochs=10
    )
    experiment.plot_comparison('layers', 'plots/lstm_layers_comparison.png')
    
    # B. Experiment with number of LSTM units
    print("\n3B. Experimenting with LSTM unit numbers...")
    experiment.experiment_lstm_units(
        X_train, y_train, X_val, y_val, X_test, y_test,
        unit_configs=[32, 64, 128], epochs=10
    )
    experiment.plot_comparison('units', 'plots/lstm_units_comparison.png')
    
    # C. Experiment with bidirectional vs unidirectional
    print("\n3C. Experimenting with LSTM direction...")
    experiment.experiment_bidirectional(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=10
    )
    experiment.plot_comparison('direction', 'plots/lstm_direction_comparison.png')
    
    # Print experiment summary
    experiment.print_summary()
    
    # Save best models from experiments
    best_layers_model = experiment.results['layers_2']['model']  # Assuming 2 layers is good
    best_layers_model.save_model('models/best_layers_model.h5')
    
    best_units_model = experiment.results['units_64']['model']  # Assuming 64 units is good
    best_units_model.save_model('models/best_units_model.h5')
    
    bidirectional_model = experiment.results['direction_bidirectional']['model']
    bidirectional_model.save_model('models/bidirectional_model.h5')
    
    # ============= 4. MANUAL FORWARD PROPAGATION =============
    print("\n4. Implementing manual forward propagation...")
    
    # Test manual implementation with baseline model
    print("\n4A. Testing unidirectional LSTM manual implementation...")
    keras_f1_uni, manual_f1_uni, agreement_uni = compare_implementations(
        'models/baseline_model.h5', X_test, y_test, is_bidirectional=False
    )
    
    # Test manual implementation with bidirectional model
    print("\n4B. Testing bidirectional LSTM manual implementation...")
    keras_f1_bi, manual_f1_bi, agreement_bi = compare_implementations(
        'models/bidirectional_model.h5', X_test, y_test, is_bidirectional=True
    )
    
    # ============= 5. FINAL RESULTS SUMMARY =============
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    print(f"Baseline Model F1 Score: {baseline_f1:.4f}")
    print(f"Best Configuration Results:")
    experiment.print_summary()
    
    print(f"\nForward Propagation Comparison:")
    print(f"Unidirectional - Keras: {keras_f1_uni:.4f}, Manual: {manual_f1_uni:.4f}, Agreement: {agreement_uni:.4f}")
    print(f"Bidirectional - Keras: {keras_f1_bi:.4f}, Manual: {manual_f1_bi:.4f}, Agreement: {agreement_bi:.4f}")
    
    # ============= 6. ANALYSIS AND CONCLUSIONS =============
    print("\n" + "="*50)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*50)
    
    # Find best performing configurations
    layer_results = {k: v['f1_score'] for k, v in experiment.results.items() if k.startswith('layers_')}
    unit_results = {k: v['f1_score'] for k, v in experiment.results.items() if k.startswith('units_')}
    direction_results = {k: v['f1_score'] for k, v in experiment.results.items() if k.startswith('direction_')}
    
    best_layers = max(layer_results, key=layer_results.get)
    best_units = max(unit_results, key=unit_results.get)
    best_direction = max(direction_results, key=direction_results.get)
    
    print(f"\nBest Configurations:")
    print(f"- Number of LSTM Layers: {best_layers} (F1: {layer_results[best_layers]:.4f})")
    print(f"- Number of LSTM Units: {best_units} (F1: {unit_results[best_units]:.4f})")
    print(f"- LSTM Direction: {best_direction} (F1: {direction_results[best_direction]:.4f})")
    
    print(f"\nConclusions:")
    print("1. LSTM Layer Analysis:")
    if layer_results['layers_2'] > layer_results['layers_1']:
        print("   - Adding more LSTM layers generally improves performance")
    else:
        print("   - Single LSTM layer performs well, additional layers may cause overfitting")
    
    print("2. LSTM Units Analysis:")
    if unit_results['units_128'] > unit_results['units_32']:
        print("   - More LSTM units generally improve model capacity and performance")
    else:
        print("   - Smaller models may be sufficient for this dataset")
    
    print("3. Direction Analysis:")
    if direction_results['direction_bidirectional'] > direction_results['direction_unidirectional']:
        print("   - Bidirectional LSTMs capture more context and perform better")
    else:
        print("   - Unidirectional LSTMs are sufficient for this task")
    
    print("4. Manual Implementation:")
    if agreement_uni > 0.95 and agreement_bi > 0.95:
        print("   - Manual forward propagation implementation is highly accurate")
        print("   - Successfully replicated Keras functionality")
    else:
        print("   - Manual implementation needs refinement")
    
    print(f"\nAll models and plots saved in respective directories.")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
