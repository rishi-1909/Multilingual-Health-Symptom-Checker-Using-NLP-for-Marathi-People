# utils.py
from typing import List, Dict, Set

# ----------------- DISCLAIMER -----------------
DISCLAIMER = """
> ⚠️ **महत्वाची सूचना / Important Notice**  
> हे साधन केवळ शैक्षणिक उद्देशासाठी तयार केलेले आहे.  
> हा AI मॉडेल वैद्यकीय निदानाचा पर्याय नाही.  
> कृपया कोणतीही गंभीर लक्षणे असल्यास ताबडतोब तज्ञ डॉक्टरांचा सल्ला घ्या.
"""

# ----------------- SYMPTOM KEYWORDS -----------------
# internal symptom codes → list of Marathi keywords
SYMPTOM_KEYWORDS: Dict[str, List[str]] = {
    "fever": ["ताप", "ताप आहे", "उष्णता"],
    "stomach_pain": ["पोटात दुखत", "पोटदुखी", "पोट दुखत"],
    "headache": ["डोकं दुखत", "डोकेदुखी", "डोक्यात दुखत"],
    "cough": ["खोकला", "खोकत"],
    "cold": ["सर्दी", "नाक वाहत", "नाक चोंदत"],
    "vomiting": ["उलटी", "ओकाऱ्या", "ओकारी"],
    "diarrhea": ["जुलाब", "सैल शौच"],
    "body_pain": ["शरीरदुखी", "संपूर्ण अंग दुखत"],
    "fatigue": ["थकवा", "दम लागणे", "दमणे"],
}

# ✅ This is what data.py is trying to import
# Marathi phrase → internal symptom code
MARATHI_SYMPTOM_MAP: Dict[str, str] = {
    kw: code
    for code, keywords in SYMPTOM_KEYWORDS.items()
    for kw in keywords
}


def extract_symptoms_marathi(text: str) -> List[str]:
    """
    Marathi free-text → list of internal symptom codes.
    """
    text = (text or "").lower()
    found: Set[str] = set()

    for code, keywords in SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.add(code)
                break

    return list(found)


# ----------------- DISEASE INFO -----------------
# Map disease name → metadata, including which symptoms are typical
DISEASE_INFO: Dict[str, Dict] = {
    "Viral Fever": {
        "symptoms": {"fever", "headache", "body_pain", "fatigue", "cold"},
        "explanation": (
            "तुम्हाला व्हायरल ताप असण्याची शक्यता आहे. यात ताप, डोकेदुखी, अंगदुखी "
            "आणि थकवा अशी लक्षणे दिसू शकतात."
        ),
        "advice": (
            "पुरेशी विश्रांती घ्या, पाणी आणि द्रव पदार्थ जास्त प्रमाणात घ्या. "
            "ताप जास्त राहिला तर डॉक्टरांचा सल्ला घ्या."
        ),
        "severity": "medium",
    },
    "Gastritis / Stomach Infection": {
        "symptoms": {"stomach_pain", "vomiting", "diarrhea", "fever"},
        "explanation": (
            "तुमच्या लक्षणांवरून पोटाचा त्रास (गॅस्ट्रायटिस / संसर्ग) असू शकतो. "
            "यात पोटदुखी, उलटी, जुलाब आणि काही वेळा तापही येऊ शकतो."
        ),
        "advice": (
            "तिखट, तेलकट आणि जड अन्न टाळा. पुरेशी पाणी आणि ओआरएस घ्या. "
            "लक्षणे वाढल्यास किंवा रक्तस्त्राव / तीव्र वेदना असल्यास ताबडतोब डॉक्टरांकडे जा."
        ),
        "severity": "high",
    },
    "Common Cold": {
        "symptoms": {"cold", "cough", "headache"},
        "explanation": (
            "सर्दीचा हलका संसर्ग दिसत आहे. नाक वाहणे, खोकला आणि हलकी डोकेदुखी "
            "ही सामान्य लक्षणे आहेत."
        ),
        "advice": (
            "गरम पाण्याची वाफ घ्या, कोमट पाणी प्या. साधारणतः काही दिवसांत बरे होते, "
            "परंतु श्वास घेण्यास त्रास होत असल्यास डॉक्टरांचा सल्ला घ्या."
        ),
        "severity": "low",
    },
}
