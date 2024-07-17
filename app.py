import streamlit as st
from PIL import Image
import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity

def main():
    st.title("Tampering Card Detection")

    uploaded_files = st.file_uploader("Upload Original and Tampered Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        original_image = None
        tampered_image = None
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            if original_image is None:
                original_image = image
            else:
                tampered_image = image

        st.image(original_image, caption='Original Image', use_column_width=True)
        st.image(tampered_image, caption='Tampered Image', use_column_width=True)
        original_image=original_image.resize((250,160))
        tampered_image=original_image.resize((250,160))
        # Convert images to grayscale
        original_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        tampered_gray = cv2.cvtColor(np.array(tampered_image), cv2.COLOR_RGB2GRAY)

        # Calculate structural similarity index
        (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw bounding rectangles
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(tampered_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

        st.image(diff, caption='Difference Image', use_column_width=True)
        st.image(thresh, caption='Thresholded Difference Image', use_column_width=True)
        st.write(f'Similarity is about {score} perc')
if __name__ == "__main__":
    main()
