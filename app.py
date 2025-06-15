import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# URL до API
API_URL = "https://img-classification-7zj5.onrender.com/predict/"

st.title("🐔🐄🐎🐑 Классификация изображений животных")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Входное изображение", use_container_width=True)

    if st.button("Классифицировать"):
        # Предобработка изображения
        img_resized = image.resize((64, 64))
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Отправка на API
        files = {"file": ("image.png", img_bytes, "image/png")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            # Предсказанный класс (строка)
            class_name = result.get("predicted_class", "unknown")

            st.subheader("✅ Предсказанный класс:")
            st.write(class_name)

            # Вероятности
            probs = result.get("probabilities", {})

            st.subheader("📊 Распределение вероятностей:")
            fig, ax = plt.subplots()
            ax.bar(probs.keys(), probs.values(), color="skyblue")
            ax.set_ylabel("Вероятность")
            ax.set_ylim([0, 1])
            st.pyplot(fig)
        else:
            st.error("Ошибка при обращении к API:")
            st.text(response.text)
