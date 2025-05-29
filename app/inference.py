import numpy as np
from app.model import load_model

def predict_signal(data: np.ndarray, model_name: str = "tcn") -> dict:
    """
    Run inference using the selected model.

    Args:
        data (np.ndarray): Input signal, shape (187, 1)
        model_name (str): Name of the model to use

    Returns:
        dict: label and confidence
    """
    model, model_type = load_model(model_name)

    if model_type == "keras":
        input_data = np.expand_dims(data, axis=0)  # Shape: (1, 187, 1)
        prediction = model.predict(input_data)[0]
    elif model_type == "svm":
        flat_input = data.flatten().reshape(1, -1)  # SVM expects 1D feature vector
        prediction = model.predict_proba(flat_input)[0]
    else:
        raise ValueError("Unknown model type")

    label_idx = int(np.argmax(prediction))
    label = "ECG" if label_idx == 0 else "EEG"

    return {
        "label": label,
        "confidence": float(np.max(prediction)),
        "raw_output": prediction.tolist(),
        "model_used": model_name
    }
