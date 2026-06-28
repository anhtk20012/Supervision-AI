import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

st.title("Face Recognition - Detect Multiple Faces")

uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    face_locations = face_recognition.face_locations(
        image_np,
        number_of_times_to_upsample=1,
        model="hog"
    )

    result = image_np.copy()

    for i, (top, right, bottom, left) in enumerate(face_locations, start=1):
        cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            result,
            f"Face {i}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    st.image(result, caption=f"Detected {len(face_locations)} face(s)", use_container_width=True)