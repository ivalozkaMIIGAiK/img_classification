import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

# Заголовок
st.title("🐾 Animal Classifier")
st.subheader("Загрузите изображение животного для классификации")

# API endpoint (замени на своё при развертывании)
API_URL = "https://your-api-url.onrender.com/predict/"

# Форма для загрузки изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Отображение изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Кнопка классификации
    if st.button("Классифицировать"):
        with st.spinner("Анализ изображения..."):
            # Отправка POST-запроса
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(API_URL, files=files)
                result = response.json()

                # Отображение результата
                st.success(f"✅ Предсказанный класс: **{result['predicted_class']}**")

                # Визуализация вероятностей
                st.subheader("Распределение вероятностей:")
                labels = list(result["probabilities"].keys())
                probs = list(result["probabilities"].values())

                fig, ax = plt.subplots()
                ax.barh(labels, probs, color="skyblue")
                ax.set_xlim([0, 1])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка при обращении к API: {e}")
