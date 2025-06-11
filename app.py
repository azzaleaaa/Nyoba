import streamlit as st
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
)
import torch
import json
import os

st.title("Perbandingan Model Deteksi Kanker Kulit")

uploaded_file = st.file_uploader("Unggah gambar lesi kulit", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    models = {
        "Vision Transformer": "Anwarkh1/Skin_Cancer-Image_Classification",
        "ConvNext": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch"
    }

    for model_name, model_id in models.items():
        st.subheader(f"Model: {model_name}")
        with st.spinner(f"Memproses dengan {model_name}..."):
            config_path = f"configs/{model_name.lower().replace(' ', '_')}_config.json"
            preprocessor_config_path = f"configs/{model_name.lower().replace(' ', '_')}_preprocessor_config.json"

            if not os.path.exists(config_path) or not os.path.exists(preprocessor_config_path):
                st.error(f"File konfigurasi untuk {model_name} tidak ditemukan.")
                continue

            with open(config_path, "r") as f:
                config_dict = json.load(f)
            with open(preprocessor_config_path, "r") as f:
                preprocessor_dict = json.load(f)

            config = AutoConfig.from_pretrained(model_id, **config_dict)
            processor = AutoImageProcessor.from_pretrained(model_id, **preprocessor_dict)

            inputs = processor(images=image, return_tensors="pt")
            model = AutoModelForImageClassification.from_pretrained(model_id, config=config)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item()

            # Tangani kasus KeyError di id2label
            id2label = model.config.id2label
            pred_class = id2label.get(str(pred_idx)) or id2label.get(pred_idx) or f"Label tidak ditemukan ({pred_idx})"

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
