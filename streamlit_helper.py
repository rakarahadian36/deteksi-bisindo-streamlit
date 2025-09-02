import cv2
import streamlit as st
import tempfile
import os

def get_video_frames(video_path):
    """Membaca video dan mengembalikan daftar frame."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def process_video_detection(frames, model, placeholder):
    """Melakukan deteksi pada setiap frame dan menampilkan hasilnya."""
    # Membuat video writer
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))

    progress_bar = st.progress(0)
    for i, frame in enumerate(frames):
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        progress_bar.progress((i + 1) / len(frames))

    out.release()
    
    # Menampilkan video hasil
    placeholder.video(temp_output_path)
    
    # Hapus file sementara
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)