import streamlit as st

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'clips_data' not in st.session_state:
        st.session_state.clips_data = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False

def reset_processing_state():
    """Resetea el estado de procesamiento"""
    st.session_state.processing_complete = False
    st.session_state.clips_data = None
    st.session_state.video_processed = False