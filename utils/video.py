import os
import shutil
import streamlit as st

def save_uploaded_file(uploaded_file, save_dir):
    """Guarda el archivo subido en el directorio especificado"""
    os.makedirs(save_dir, exist_ok=True)
    filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.'))
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, "wb") as f:
        for chunk in iter(lambda: uploaded_file.read(4096), b""):
            f.write(chunk)
    return file_path

def load_video_as_bytes(video_path):
    """Carga un video como bytes para reproducción en Streamlit"""
    try:
        with open(video_path, "rb") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error cargando video {video_path}: {str(e)}")
        return None

def display_video_with_fallback(video_path, caption="", unique_key=""):
    """Muestra un video con sistema de fallback"""
    if not os.path.exists(video_path):
        st.error(f"Video no encontrado: {video_path}")
        return False
    
    try:
        video_bytes = load_video_as_bytes(video_path)
        if video_bytes:
            st.video(video_bytes, start_time=0)
            if caption:
                st.caption(caption)
            return True
    except Exception as e:
        st.error(f"Error reproduciendo video: {str(e)}")
    
    return False

def create_download_link(video_path, filename, unique_key):
    """Crea un enlace de descarga para un video individual"""
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        return st.download_button(
            label = f"⬇️ Descargar {filename}",
            data = video_bytes,
            file_name = filename,
            mime = "video/mp4",
            key = unique_key,
            use_container_width = True
        )
    except Exception as e:
        st.error(f"Error creando enlace de descarga: {str(e)}")
        return False

def clear_folder(folder):
    """Limpia y recrea una carpeta"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok = True)