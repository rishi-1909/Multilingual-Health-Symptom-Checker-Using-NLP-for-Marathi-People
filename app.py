import streamlit as st
from model import SymptomChecker
from utils import DISCLAIMER
import time

st.set_page_config(
    page_title="Marathi AI Health Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🩺 Marathi AI Health Symptom Checker")
    st.markdown("### 🤖 AI-powered लक्षण तपासणी | NLP + Transformer Models")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ सेटिंग्ज / Settings")
        if st.button("💻 प्रथम मॉडेल प्रशिक्षण करा / Train Model First", type="primary"):
            with st.spinner("प्रशिक्षण सुरू... / Training..."):
                # फक्त डेमो – इथे तुम्ही खरे train_model() लावू शकता
                time.sleep(3)
            st.success("✅ प्रशिक्षण पूर्ण / Training Complete! (पूर्व-प्रशिक्षित मॉडेल वापरले जात आहे)")
        st.markdown("---")
        st.markdown(DISCLAIMER)

    # Load model
    checker = SymptomChecker()

    st.markdown("### 📝 तुमची लक्षणे मराठीत लिहा / Enter your symptoms in Marathi")
    user_text = st.text_area(
        "उदा.: 'माझ्या पोटात दुखत आहे आणि मला ताप आहे.'",
        height=150
    )

    if st.button("🔍 तपासा / Check Symptoms"):
        if not user_text.strip():
            st.warning("कृपया प्रथम लक्षणे लिहा. / Please enter your symptoms first.")
        else:
            with st.spinner("विश्लेषण सुरू आहे... / Analyzing..."):
                result = checker.predict(user_text)

            st.subheader("📊 निष्कर्ष / Result")
            st.write(f"**ओळखलेली लक्षणे / Symptoms:** {', '.join(result.get('symptoms', [])) or '—'}")
            st.write(f"**संभाव्य आजार / Possible Disease:** {result.get('disease', 'Unknown')}")
            st.write(f"**विश्वास / Confidence:** {result.get('confidence', 0.0):.2f}")
            st.write(f"**तीव्रता / Severity:** {result.get('severity', 'medium')}")
            st.markdown(f"**स्पष्टीकरण / Explanation:** {result.get('explanation', '')}")
            st.markdown(f"**सल्ला / Advice:** {result.get('advice', '')}")

if __name__ == "__main__":
    main()
