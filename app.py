import streamlit as st
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv  # Import the Conv class
import torch
from torch.nn import Sequential
import cv2
import tempfile
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Keyboard and Mouse Detection",
    page_icon="🖱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads and caches the YOLOv8 model, adding required classes to PyTorch's trusted list."""
    try:
        # This is the fix for the PyTorch 2.6+ security update.
        # It tells PyTorch to trust architecture classes from ultralytics and torch itself.
        torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# --- UI Sidebar ---
st.sidebar.title("Configuration")
MODEL_PATH = os.path.join('weights', 'best.pt')

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

confidence = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

st.sidebar.header("Detection Mode")
option = st.sidebar.selectbox(
    'Choose an option',
    ('Image', 'Video', 'Webcam'),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n"
    "This app uses a YOLOv8 model to detect keyboards and mice in real-time "
    "from images, videos, or a webcam feed."
)

# --- Main Application Logic ---
st.title("Keyboard and Mouse Detection")

# Initialize session state for webcam
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

if option == 'Image':
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption='Uploaded Image', use_container_width=True)
            
            with st.spinner("Detecting objects..."):
                results = model(img_array, conf=confidence)
                annotated_image = results[0].plot()
            
            with col2:
                st.image(annotated_image, caption='Detected Image', use_container_width=True, channels="BGR")

        except Exception as e:
            st.error(f"Error processing image: {e}")

elif option == 'Video':
    st.header("Video Detection")
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            st.info("Processing video... Please wait.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.success("Video processing complete.")
                    break

                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            if cap: cap.release()
            tfile.close()
            os.unlink(video_path)

elif option == 'Webcam':
    st.header("Webcam Detection")
    
    if st.button("Start Webcam"):
        st.session_state.webcam_active = True

    if st.button("Stop Webcam"):
        st.session_state.webcam_active = False

    if st.session_state.webcam_active:
        st.info("Webcam is active. Click 'Stop Webcam' to end.")
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Check if it is connected and not in use.")
            else:
                st_frame = st.empty()
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    if not ret: break

                    results = model(frame, conf=confidence)
                    annotated_frame = results[0].plot()
                    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred with the webcam feed: {e}")
        finally:
            if cap: cap.release()
            if not st.session_state.webcam_active:
                st.success("Webcam stopped.")