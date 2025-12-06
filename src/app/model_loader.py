import joblib 
import os
from pathlib import Path

def load_model():
    root = os.path.abspath(Path(__file__).resolve().parent.parent.parent)
    model = joblib.load(os.path.join(root, "models", "rain_clf.pkl"))
    print('Load Model berhasil')
    return model

