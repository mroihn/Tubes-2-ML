import matplotlib.pyplot as plt


def plot_training_history_local(history, model_name_title, save_path_prefix):
    plt.figure(figsize=(12, 5))
    plt.suptitle(model_name_title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_prefix:
        plt.savefig(os.path.join(results_images_dir,
                    f"{save_path_prefix}_history.png"))
    plt.show()


def evaluate_and_report_local(model, x_test, y_test, model_name="Model", class_names=None, save_path_prefix=None):
    print(f"\n--- Evaluation Report for: {model_name} ---")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_proba = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    macro_f1 = f1_score(y_test, y_pred_classes, average='macro')
    print(f"Test Macro F1-Score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes,
          target_names=class_names if class_names else [f'Class {i}' for i in range(num_classes)]))

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else np.arange(
                    num_classes),
                yticklabels=class_names if class_names else np.arange(num_classes))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path_prefix:
        plt.savefig(os.path.join(results_images_dir,
                    f"{save_path_prefix}_cm.png"))
    plt.show()
    return loss, accuracy, macro_f1
