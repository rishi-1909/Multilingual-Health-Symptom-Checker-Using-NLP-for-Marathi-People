# ================================
# model.py  (clean, fixed version)
# ================================
import os
from typing import List, Dict, Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier

# if you use these in training scripts, keep them; otherwise you can remove
from data import load_dataset, prepare_data
from utils import extract_symptoms_marathi, DISEASE_INFO


# --------- CONSTANTS ----------
MODEL_NAME = "l3cube-pune/marathi-bert-v2"   # public Marathi BERT
MODEL_PATH = "model.pth"
RF_PATH = "rf_model.pkl"
DISEASES_PATH = "diseases.pkl"


# --------- DATASET FOR BERT ----------
class SymptomDataset(Dataset):
    """
    Simple dataset that converts a text (space-separated symptom names)
    and a numeric label into tokenized tensors for BERT.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# --------- BERT CLASSIFIER ----------
class IndicBERTClassifier(nn.Module):
    """
    Simple classification head on top of Marathi BERT.
    """

    def __init__(self, num_classes: int, model_name: str = MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use pooled output if available else [CLS] token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# --------- MAIN SERVICE CLASS ----------
class SymptomChecker:
    """
    Loads trained RF + BERT models and exposes a .predict(marathi_text) method.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        rf_path: str = RF_PATH,
        diseases_path: str = DISEASES_PATH
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.diseases: List[str] = []
        self.symptom_cols: List[str] = []
        self.model: Optional[IndicBERTClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None

        self._load_models(model_path, rf_path, diseases_path)

    # ----- internal load -----
    def _load_models(self, model_path: str, rf_path: str, diseases_path: str) -> None:
        """
        Load RandomForest + BERT weights and metadata if present.
        """

        # Load RF + metadata
        if os.path.exists(rf_path) and os.path.exists(diseases_path):
            self.rf_model = joblib.load(rf_path)
            self.diseases, self.symptom_cols = joblib.load(diseases_path)
            print("✅ RandomForest model loaded.")
        else:
            print("⚠️ RF model or diseases metadata not found. Run training first.")

        # Load BERT classifier
        if os.path.exists(model_path) and self.diseases:
            self.model = IndicBERTClassifier(num_classes=len(self.diseases), model_name=self.model_name)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ BERT classifier loaded.")
        else:
            print("⚠️ BERT model weights not found. Only RF (if available) will be used.")

    # ----- helper: symptoms -> vector -----
    def symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert a list of symptom names to a binary vector matching symptom_cols.
        """
        if not self.symptom_cols:
            # if models not trained yet – 10 zeros just as a safe default shape
            return np.zeros(10, dtype=np.float32)

        vec = np.zeros(len(self.symptom_cols), dtype=np.float32)
        for s in symptoms:
            if s in self.symptom_cols:
                idx = self.symptom_cols.index(s)
                vec[idx] = 1.0
        return vec

    # ----- rule-based fallback diagnosis -----
    def _rule_based_diagnosis(self, symptoms: List[str]):
        """
        वापरकर्त्याच्या लक्षणांवरून DISEASE_INFO मधून सर्वोत्तम जुळणारा आजार निवडा.
        """
        if not symptoms:
            return "Unknown", {}

        best_disease = "Unknown"
        best_score = 0
        best_info = {}

        symptom_set = set(symptoms)

        for disease, info in DISEASE_INFO.items():
            disease_symptoms = set(info.get("symptoms", []))
            if not disease_symptoms:
                continue
            overlap = len(symptom_set & disease_symptoms)
            if overlap > best_score:
                best_score = overlap
                best_disease = disease
                best_info = info

        return best_disease, best_info

    # ----- main prediction -----
    def predict(self, marathi_text: str) -> Dict:
        """
        Main inference pipeline. Returns a dict:
        {
            'symptoms': [...],
            'disease': '...',
            'explanation': '...',
            'advice': '...',
            'confidence': float,
            'severity': 'low/medium/high'
        }
        """

        # 1. Extract symptoms from Marathi text (rule-based)
        symptoms = extract_symptoms_marathi(marathi_text)
        symptom_vec = self.symptoms_to_vector(symptoms)

        predictions: Dict[str, Dict] = {}

        # 2. RandomForest prediction (fast, uses structured vector)
        if self.rf_model is not None and len(symptom_vec) == len(self.symptom_cols):
            rf_pred_idx = int(self.rf_model.predict([symptom_vec])[0])
            if self.diseases and 0 <= rf_pred_idx < len(self.diseases):
                rf_disease = self.diseases[rf_pred_idx]
            else:
                rf_disease = "Unknown"

            if hasattr(self.rf_model, "predict_proba"):
                rf_proba = float(self.rf_model.predict_proba([symptom_vec]).max())
            else:
                rf_proba = 0.0

            predictions["rf"] = {
                "disease": rf_disease,
                "confidence": rf_proba,
                "symptoms_detected": symptoms
            }

        # 3. BERT prediction (text-based)
        if self.model is not None and self.diseases:
            symptoms_text = " ".join(symptoms) if symptoms else marathi_text
            enc = self.tokenizer(
                symptoms_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"]
                )
                probs = torch.softmax(logits, dim=-1)
                bert_idx = int(torch.argmax(probs, dim=-1).cpu().numpy()[0])
                bert_conf = float(probs.max().cpu().numpy())

            bert_disease = self.diseases[bert_idx] if 0 <= bert_idx < len(self.diseases) else "Unknown"

            predictions["bert"] = {
                "disease": bert_disease,
                "confidence": bert_conf,
                "symptoms_detected": symptoms
            }

        # 4. Choose primary prediction or fallback to rule-based
        if "rf" in predictions and "bert" in predictions:
            primary_pred = (
                predictions["rf"]
                if predictions["rf"]["confidence"] >= predictions["bert"]["confidence"]
                else predictions["bert"]
            )
        elif "rf" in predictions:
            primary_pred = predictions["rf"]
        elif "bert" in predictions:
            primary_pred = predictions["bert"]
        else:
            # 👉 No ML model prediction available – use rule-based fallback
            disease, info = self._rule_based_diagnosis(symptoms)
            if disease != "Unknown":
                confidence = 0.6   # pseudo-confidence for rule-based
            else:
                confidence = 0.0

            explanation = info.get(
                "explanation",
                "अज्ञात आजार / Unknown condition."
            )
            advice = info.get(
                "advice",
                "कृपया ताबडतोब जवळच्या डॉक्टरांशी संपर्क करा. / Please consult a doctor immediately."
            )
            severity = info.get("severity", "medium")

            return {
                "symptoms": symptoms,
                "disease": disease,
                "explanation": explanation,
                "advice": advice,
                "confidence": confidence,
                "severity": severity,
            }

        # 5. Build final structured response using DISEASE_INFO (for RF/BERT)
        disease = primary_pred["disease"]
        confidence = primary_pred["confidence"]
        symptoms_detected = primary_pred.get("symptoms_detected", symptoms)

        info = DISEASE_INFO.get(disease, {})
        explanation = info.get(
            "explanation",
            "ही माहिती अंदाजावर आधारित आहे. कृपया याला वैद्यकीय सल्ला समजू नका. / This is an AI-based estimate, not medical advice."
        )
        advice = info.get(
            "advice",
            "तुमच्या लक्षणांबाबत तज्ञ डॉक्टरांचा सल्ला घ्या. / Please talk to a qualified doctor about your symptoms."
        )
        severity = info.get("severity", "medium")

        return {
            "symptoms": symptoms_detected,
            "disease": disease,
            "explanation": explanation,
            "advice": advice,
            "confidence": confidence,
            "severity": severity,
        }
