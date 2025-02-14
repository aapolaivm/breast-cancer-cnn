def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct_predictions / len(y_true)

def calculate_precision(y_true, y_pred):
    true_positives = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    predicted_positives = sum(y_p == 1 for y_p in y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def calculate_recall(y_true, y_pred):
    true_positives = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    actual_positives = sum(y_t == 1 for y_t in y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_model(model, val_images, val_labels):
    """
    Evaluate the model on validation data and print metrics.
    
    Args:
        model: Trained tensorflow model
        val_images: Validation image array
        val_labels: Validation labels array
    """
    # Get model predictions
    predictions = model.predict(val_images)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    # Calculate metrics
    accuracy = calculate_accuracy(val_labels, predictions)
    precision = calculate_precision(val_labels, predictions)
    recall = calculate_recall(val_labels, predictions)
    f1 = calculate_f1_score(val_labels, predictions)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }