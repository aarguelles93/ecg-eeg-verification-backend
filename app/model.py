import tensorflow as tf
import joblib
import os
from app.model_config import MODEL_CONFIG

MODEL_DIR = "app/model"

def load_model(model_name: str):
    """
    Load the specified model by name.

    Args:
        model_name (str): One of ['tcn', 'cnn', 'cnn_lstm', 'mlp', 'svm']

    Returns:
        model (loaded object), model_type ('keras' or 'svm')
    """
    model_map = {
        "tcn": "tcn_best.keras",
        "cnn": "simple_cnn_best.keras",
        "cnn_lstm": "cnn_lstm_best.keras",
        "mlp": "mlp_salvaged_best.keras",
        "svm": "svm_model.joblib"
    }

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model: {model_name}")

    model_info = MODEL_CONFIG[model_name]
    model_path = os.path.join(MODEL_DIR, model_info["file"])

    if model_info["type"] == "svm":
        model = joblib.load(model_path)
        return model, "svm"
    else:
        model = tf.keras.models.load_model(model_path)
        return model, "keras"
