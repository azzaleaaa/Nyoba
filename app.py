import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import json

st.title("Perbandingan Model Deteksi Kanker Kulit")

uploaded_file = st.file_uploader("Unggah gambar lesi kulit", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Ganti model_id sesuai model yang ada di Hugging Face hub
    models = {
        "Vision Transformer": "Anwarkh1/Skin_Cancer-Image_Classification",
        "ConvNext": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch"
    }

    for model_name, model_id in models.items():
        st.subheader(f"Model: {model_name}")
        with st.spinner(f"Memproses dengan {model_name}..."):
            # Memuat konfigurasi dari file JSON
            with open(f"configs/{model_name.lower().replace(' ', '_')}_config.json", "r") as f:
                custom_config = json.load(f)
            with open(f"configs/{model_name.lower().replace(' ', '_')}_preprocessor_config.json", "r") as f:
                custom_preprocessor_config = json.load(f)

            # Memuat preprocessor dan model dengan konfigurasi yang dimodifikasi
            processor = AutoImageProcessor.from_pretrained(model_id, **custom_preprocessor_config)
            model = AutoModelForImageClassification.from_pretrained(model_id, config=custom_config)

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred_idx = torch.argmax(probs).item()
            pred_class = model.config.id2label[pred_idx]
            confidence = probs[0][pred_idx].item()

            if confidence >= 0.5:
                st.write(f"Prediksi: **{pred_class}**")
                st.write(f"Akurasi Prediksi: **{confidence:.2%}**")
            else:
                st.write("⚠️ Model tidak cukup yakin untuk melakukan prediksi (akurasi < 50%).")

st.markdown("""---
### 🧠 Credit
**📦 Model:**  
- [Vision Transformer by Anwarkh1](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)  
- [ConvNext by Pranavkpba2000](https://huggingface.co/Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch)
""")
