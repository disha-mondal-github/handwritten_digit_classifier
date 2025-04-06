import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the trained Keras ANN model
model = load_model('digit_recognizer_model.h5')

# Load and preprocess MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values

st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Select an image from the MNIST test set (0–9999) and let the model predict the digit.")

# Select image index from test set
index = st.slider("Choose Image Index", 0, 9999, 0)

# Display the selected image
st.image(x_test[index], width=150, caption=f"Actual Label: {y_test[index]}")

# Prepare image for prediction
input_image = x_test[index].reshape(1, 28, 28, 1)

# Make prediction
prediction = model.predict(input_image)
predicted_digit = np.argmax(prediction)

# Display result
st.subheader(f"🔢 Predicted Digit: {predicted_digit}")
st.write(f"📊 Confidence Scores: {np.round(prediction[0], 2)}")

# Optional: Plot confidence scores
fig, ax = plt.subplots()
ax.bar(range(10), prediction[0])
ax.set_xticks(range(10))
ax.set_xlabel("Digit")
ax.set_ylabel("Confidence")
st.pyplot(fig)
