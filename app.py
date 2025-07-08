import os
import tempfile
import traceback
import numpy as np
import streamlit as st
from ui.preview import show_results
from utils.video import save_uploaded_file
from enhanced_video import process_school_video
from utils.session import initialize_session_state, reset_processing_state

# Importar la nueva página de audio
from audio_analysis import audio_analysis_page

UPLOAD_FOLDER = "temp_uploads"
YOLO_MODEL_PATH = "models/best.pt"

def video_analysis_page():
    """
    Página original de análisis de video
    """
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">📊 Análisis de Video Escolar</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Análisis Inteligente de Videos Educativos con IA</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuración del Análisis")
        
        # Configuración de procesamiento
        with st.expander("🎬 Configuración de Video", expanded=True):
            interval_min = st.slider("Intervalo entre clips (min)", 1, 30, 5, key="interval_slider")
            interval_sec = interval_min * 60

            clip_duration = st.slider("Duración del clip (seg)", 5, 120, 10, key="duration_slider")
        
        # Configuración YOLO
        with st.expander("🤖 Detección de Personas", expanded=True):
            yolo_analysis = st.checkbox(
                "👥 Análisis YOLO", True, help="Detecta estudiantes y maestros", key="yolo_analysis"
            )
            
            yolo_in_video = st.checkbox(
                "📹 Mostrar detecciones en video", True, help="Dibuja las detecciones en los clips", key="yolo_in_video", disabled=not yolo_analysis
            )
            
            if yolo_analysis:
                confidence_threshold = st.slider(
                    "Umbral de confianza", 0.1, 0.9, 0.5, 0.1, help="Confianza mínima para detecciones", key="confidence_slider"
                )
            else:
                confidence_threshold = 0.5
        
        # Configuración de emociones
        with st.expander("😊 Análisis de Emociones", expanded=True):
            emotion_analysis = st.checkbox("🧠 Análisis emocional", True, key="emotion_check")
            
            if emotion_analysis:
                emotion_model_path = st.text_input(
                    "Ruta al modelo",
                    value="models/best_emotion.h5",
                    help="Ruta al modelo .h5 de emociones",
                    key="emotion_model_path"
                )
            else:
                emotion_model_path = None
        
        st.markdown("---")
        
        # Botón de reset
        if st.button("🔄 Nuevo Análisis", key="reset_button", use_container_width=True, type="primary"):
            reset_processing_state()
            st.rerun()
        
        # Mostrar métricas en tiempo real si hay datos
        if st.session_state.processing_complete and st.session_state.clips_data:
            st.markdown("### 📊 Métricas Rápidas")
            result = st.session_state.clips_data
            summary = result.get("summary", {})
            clips_meta = summary.get("clips_meta", [])
            
            if clips_meta:
                # Calcular métricas rápidas
                avg_students = np.mean(
                    [meta.get("detections", {}).get("students_sitting", 0) + meta.get("detections", {}).get("students_standing", 0) for meta in clips_meta]
                )
                
                emotion_summary = summary.get("emotion_summary", {})
                total_emotions = sum(emotion_summary.values())
                positive_emotions = emotion_summary.get("happy", 0) + emotion_summary.get("surprise", 0)
                engagement_rate = (positive_emotions / total_emotions * 100) if total_emotions > 0 else 0
                
                st.metric("Estudiantes/clip", f"{avg_students:.1f}")
                st.metric("Engagement", f"{engagement_rate:.1f}%")
                st.metric("Clips generados", len(clips_meta))

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Subir Video")
        uploaded_file = st.file_uploader(
            "Selecciona un video para análisis", 
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
            help="Máximo 2GB por archivo",
            key="file_uploader"
        )
        
        if uploaded_file and st.session_state.uploaded_file_name != uploaded_file.name:
            reset_processing_state()
            st.session_state.uploaded_file_name = uploaded_file.name

    with col2:
        st.subheader("📄 Información del Archivo")
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Tamaño", f"{file_size_mb:.1f} MB")
            st.metric("Tipo", uploaded_file.type)
        else:
            st.info("Esperando archivo...")

    # Mostrar resultados
    if st.session_state.processing_complete and st.session_state.clips_data:
        show_results(st.session_state.clips_data)
        return

    # Procesar video
    if uploaded_file and not st.session_state.video_processed:
        with st.spinner("📥 Guardando video..."):
            try:
                video_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
                st.success(f"✅ Video guardado: {os.path.basename(video_path)}")
            except Exception as e:
                st.error(f"❌ Error al guardar el video: {str(e)}")
                return

        output_dir = tempfile.mkdtemp(prefix="clips_", suffix="_temp")

        if st.button("🚀 Procesar Video", type="primary", use_container_width=True, key="process_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎬 Procesando el video..."):
                try:
                    progress_bar.progress(10)
                    
                    result = process_school_video(
                        video_path=video_path,
                        model_path=YOLO_MODEL_PATH, 
                        interval_seconds=interval_sec,
                        output_folder=output_dir,
                        clip_duration_sec=clip_duration,
                        confidence_threshold=confidence_threshold,
                        detection_analysis=yolo_analysis,
                        yolo_in_video=yolo_in_video,
                        emotion_model_path=emotion_model_path if emotion_analysis else None
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("¡Procesamiento completado!")
                    
                    st.session_state.clips_data = result
                    st.session_state.processing_complete = True
                    st.session_state.video_processed = True
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error("❌ Error durante el procesamiento.")
                    st.error(f"Detalles del error: {str(e)}")
                    with st.expander("Ver traceback completo"):
                        st.code(traceback.format_exc())
                    return

def main():
    """
    Función principal que maneja la navegación entre páginas
    """
    st.set_page_config(
        page_title="Dashboard de Análisis Escolar", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navegación principal
    st.markdown("""
    <style>
    .main-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .nav-button {
        background: rgba(255,255,255,0.2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
    }
    .nav-button:hover {
        background: rgba(255,255,255,0.3);
    }
    .nav-button.active {
        background: rgba(255,255,255,0.4);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Crear tabs para navegación
    tab1, tab2 = st.tabs(["📊 Análisis de Video", "🎵 Análisis de Audio"])
    
    with tab1:
        video_analysis_page()
    
    with tab2:
        # Importar y ejecutar la página de audio
        try:
            audio_analysis_page()
        except Exception as e:
            st.error(f"Error al cargar la página de análisis de audio: {str(e)}")
            st.info("Asegúrate de que todos los módulos del pipeline de audio estén correctamente instalados en la carpeta 'src/'")

if __name__ == "__main__":
    main()