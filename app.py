import streamlit as st
import numpy as np
from streamlit_helper import get_video_frames, process_video_detection
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Bahasa Isyarat BISINDO",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Deteksi Bahasa Isyarat BISINDO 🤟')
st.write('Unggah video atau gambar, atau gunakan webcam untuk deteksi bahasa isyarat secara real-time.')

# Path ke model
model_path = 'best.pt'

# Pilihan mode
selected_mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Unggah Video", "Unggah Gambar", "Deteksi Webcam"]
)

# Tombol bantuan
if st.sidebar.button('Bantuan'):
    st.sidebar.info("""
    **Mode Unggah Video**: Unggah file video (format .mp4).
    **Mode Unggah Gambar**: Unggah file gambar (format .jpg, .jpeg, .png).
    **Mode Deteksi Webcam**: Izinkan akses ke webcam untuk deteksi real-time.
    """)

# Memuat model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

if selected_mode == "Unggah Gambar":
    st.header('Unggah Gambar untuk Deteksi')
    uploaded_image = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_pil = Image.open(uploaded_image)

        # --- PERBAIKAN DI SINI ---
        # Konversi gambar ke mode RGB untuk memastikan hanya ada 3 channel (menghapus channel Alpha/transparansi)
        image_rgb = image_pil.convert('RGB')
        
        # Lanjutkan proses dengan gambar yang sudah dikonversi
        image_np = np.array(image_rgb)
        
        st.image(image_rgb, caption='Gambar Asli', use_container_width=True)
        st.write("Mendeteksi...")
        
        # Berikan gambar numpy 3-channel ke model
        results = model(image_np)
        annotated_image = results[0].plot()
        
        st.image(annotated_image, caption='Hasil Deteksi', use_container_width=True)

elif selected_mode == "Unggah Video":
    st.header('Unggah Video untuk Deteksi')
    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        if st.button('Mulai Deteksi pada Video'):
            st.write("Memproses video...")
            
            frames = get_video_frames(tfile.name)
            
            output_video_placeholder = st.empty()
            
            process_video_detection(frames, model, output_video_placeholder)

elif selected_mode == "Deteksi Webcam":
    st.header('Deteksi Bahasa Isyarat via Webcam')
    st.write("Klik tombol 'Mulai' untuk memulai deteksi dan 'Hentikan' untuk berhenti.")

    if 'stop_webcam' not in st.session_state:
        st.session_state.stop_webcam = True 

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Mulai'):
            st.session_state.stop_webcam = False
    with col2:
        if st.button('Hentikan'):
            st.session_state.stop_webcam = True

    if not st.session_state.stop_webcam:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        
        while cap.isOpened() and not st.session_state.stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.write("Webcam tidak dapat diakses atau telah berhenti.")
                st.session_state.stop_webcam = True 
                break
            
            results = model(frame)
            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()