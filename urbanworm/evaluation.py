__version__ = '0.0.1'

import pandas as pd
import numpy as np

def evaluate(y_true, y_pred):
    """
    Evaluate the model performance using RMSE and MAE
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae