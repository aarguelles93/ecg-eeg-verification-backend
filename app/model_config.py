# app/model_config.py

MODEL_CONFIG = {
    "tcn": {
        "file": "tcn_best.keras",
        "type": "keras"
    },
    "cnn": {
        "file": "simple_cnn_best.keras",
        "type": "keras"
    },
    "cnn_lstm": {
        "file": "cnn_lstm_best.keras",
        "type": "keras"
    },
    "mlp": {
        "file": "mlp_salvaged_best.keras",
        "type": "keras"
    },
    "svm": {
        "file": "svm_model.joblib",
        "type": "svm"
    }
}
