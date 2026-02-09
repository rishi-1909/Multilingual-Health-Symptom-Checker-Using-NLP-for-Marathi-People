# ================================
# FILE 2: data.py  
# ================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from utils import MARATHI_SYMPTOM_MAP

def create_sample_dataset() -> pd.DataFrame:
    """
    Create synthetic dataset compatible with Kaggle disease-symptom format [web:43][web:47]
    Expected format for real Kaggle dataset: columns = ['disease', 'symptom_1', 'symptom_2', ...]
    """
    data = {
        'disease': [
            'Common Cold', 'Common Cold', 'Viral Fever', 'Viral Fever', 'Gastroenteritis',
            'Gastroenteritis', 'Hypertension', 'Hypertension', 'Diabetes', 'Diabetes'
        ] * 20,  # 200 samples
        
        # Binary symptom presence (1/0) - 10 common symptoms
        'fever': [1, 0, 1, 1, 0, 0, 0, 0, 0, 1] * 20,
        'headache': [1, 1, 1, 0, 0, 1, 1, 0, 0, 0] * 20,
        'cough': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 20,
        'stomach_pain': [0, 0, 0, 0, 1, 1, 0, 0, 1, 0] * 20,
        'fatigue': [1, 0, 1, 1, 1, 0, 1, 1, 1, 1] * 20,
        'chest_pain': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0] * 20,
        'dizziness': [0, 1, 1, 0, 0, 1, 1, 1, 1, 0] * 20,
        'vomiting': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1] * 20,
        'runny_nose': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 20,
        'sore_throat': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 20
    }
    
    df = pd.DataFrame(data)
    return df

def load_dataset(file_path: str = None) -> pd.DataFrame:
    """Load dataset - synthetic or real Kaggle file"""
    if file_path and pd.io.common.file_exists(file_path):
        print(f"Loading real dataset from: {file_path}")
        df = pd.read_csv(file_path)
        # Expect Kaggle format: first col disease, rest symptoms (0/1)
        symptom_cols = [col for col in df.columns if col != df.columns[0]]
        print(f"Dataset loaded: {df.shape[0]} samples, {len(symptom_cols)} symptoms")
        return df
    else:
        print("Using synthetic dataset for demo (replace with real Kaggle CSV)")
        return create_sample_dataset()

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert to features (symptoms) and labels (diseases)"""
    symptom_cols = [col for col in df.columns if col != 'disease']
    X = df[symptom_cols].values
    y = df['disease'].values
    
    # For multi-label, but here single-label per sample
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform([label.split() for label in y])  # Simple split for demo
    
    diseases = mlb.classes_.tolist()
    return X, y_bin.argmax(axis=1), diseases, symptom_cols
