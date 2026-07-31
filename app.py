import streamlit as st
import cv2
import numpy as np
import pickle
import os
from feature_extraction import extract_features

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Padi",
    layout="wide"
)

# ===============================
# Load Model (Cache to improve performance)
# ===============================
@st.cache_resource
def load_model():
    model_path = "model/svm_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

def main():
    model = load_model()
    
    # ===============================
    # CSS (Menyesuaikan UI Anda)
    # ===============================
    st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .header {
        background-color: #e7f5ea;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .badge {
        background-color: #4CAF50;
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===============================
    # Header
    # ===============================
    st.markdown("""
    <div class="header">
        <h2>Deteksi Penyakit Daun Padi</h2>
        <p>Identifikasi penyakit daun padi menggunakan Support Vector Machine</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Model SVM tidak ditemukan! Silakan jalankan script `train_model.py` terlebih dahulu untuk melatih dan menyimpan model.")
        return

    # ===============================
    # Layout Upload & Preview
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Unggah Citra Daun Padi")
        uploaded_file = st.file_uploader(
            "Upload Gambar",
            type=["jpg", "jpeg", "png"]
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Pratinjau Citra")
        preview_area = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # Hasil Prediksi
    # ===============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

    if uploaded_file is not None:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            preview_area.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                width=450
            )

            features = extract_features(image).reshape(1, -1)
            prediction = model.predict(features)[0]
            confidence = np.max(model.predict_proba(features)) * 100

            # Jika tingkat kepercayaan model sangat rendah, kemungkinan besar bukan daun padi
            if confidence < 75.0:
                st.warning("⚠️ Maaf, gambar yang Anda unggah tidak dikenali. Kemungkinan ini BUKAN gambar daun padi yang valid sesuai dataset.")
                st.info(f"*(Tingkat keyakinan model terlalu rendah: {confidence:.2f}%)*")
            else:
                st.markdown(f"""
                <p><strong>Penyakit :</strong>
                <span class="badge">{prediction.upper()}</span></p>
                <p><strong>Akurasi :</strong> {confidence:.2f}%</p>
                """, unsafe_allow_html=True)
        else:
            st.error("File gambar tidak valid atau rusak.")

    else:
        st.write("Silakan unggah citra daun padi untuk melihat hasil prediksi.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
