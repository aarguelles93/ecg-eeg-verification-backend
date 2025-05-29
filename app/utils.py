import pandas as pd
import numpy as np

MODEL_INPUT_CONFIG = {
    "ecg": {
        "length": 187
    },
    "eeg": {
        "length": 187
    }
}

def validate_and_preprocess(df: pd.DataFrame) -> np.ndarray:
    # Drop any non-numeric columns (just in case)
    df = df.select_dtypes(include=[np.number])

    # Flatten all values
    data = df.values.flatten()

    # Trim/pad to 187
    data = data[:187]
    if data.shape[0] < 187:
        data = np.pad(data, (0, 187 - data.shape[0]), mode='constant')

    # Normalize
    data = (data - np.mean(data)) / np.std(data)

    return data.reshape((187, 1))
