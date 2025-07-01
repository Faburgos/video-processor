import numpy as np
import streamlit as st
from utils.session import reset_processing_state

H5_MODEL_PATH = "models/best_emotion.h5"

def render_header():
    """
    Encabezado superior estilizado
    """
    # Header mejorado
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">📊 Dashboard de Análisis Escolar</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Análisis Inteligente de Videos Educativos con IA</p>
    </div>
    """, unsafe_allow_html = True)

def render_sidebar():
    """
    Sidebar con todos los parámetros de configuración
    """
    # Sidebar mejorado
    with st.sidebar:
        st.markdown("### ⚙️ Configuración del Análisis")
        
        # Configuración de procesamiento
        with st.expander("🎬 Configuración de Video", expanded = True):
            interval_min = st.slider("Intervalo entre clips (min)", 1, 30, 5, key = "interval_slider")
            interval_sec = interval_min * 60

            clip_duration = st.slider("Duración del clip (seg)", 5, 120, 10, key = "duration_slider")

        # Configuración de análisis
        with st.expander("🔍 Configuración de Análisis", expanded = True):
            quality_check = st.checkbox(
                "✅ Análisis de calidad", True, key = "quality_check"
            )
            motion_check = st.checkbox(
                "🏃 Detección de movimiento", True, key = "motion_check"
            )
            smart_filter = st.checkbox(
                "🧠 Filtrado inteligente", True, help = "Solo clips que cumplan TODOS los criterios", key = "smart_filter"
            )

        # Configuración YOLO
        with st.expander("🤖 Detección de Personas", expanded = True):
            yolo_analysis = st.checkbox(
                "👥 Análisis YOLO", True, help = "Detecta estudiantes y maestros", key = "yolo_analysis"
            )

            yolo_in_video = st.checkbox(
                "📹 Mostrar detecciones en video", True, help = "Dibuja las detecciones en los clips", key = "yolo_in_video", disabled = not yolo_analysis
            )

            if yolo_analysis:
                confidence_threshold = st.slider(
                    "Umbral de confianza", 0.1, 0.9, 0.5, 0.1, help = "Confianza mínima para detecciones", key = "confidence_slider"
                )
            else:
                confidence_threshold = 0.5
        
        # Configuración de emociones
        with st.expander("😊 Análisis de Emociones", expanded = True):
            emotion_analysis = st.checkbox("🧠 Análisis emocional", True, key = "emotion_check")
            
            if emotion_analysis:
                emotion_model_path = st.text_input(
                    "Ruta al modelo",
                    value = H5_MODEL_PATH,
                    help = "Ruta al modelo .h5 de emociones",
                    key = "emotion_model_path"
                )
            else:
                emotion_model_path = None
        
        # Configuración de compresión
        with st.expander("⚡ Optimización", expanded = False):
            scale = st.slider(
                "Escala de resolución", 0.1, 1.0, 1.0, 0.1, help = "1.0 = resolución original", key = "scale_slider"
            )
            reduce_fps = st.checkbox(
                "📉 Reducir FPS", False, help = "Reduce tamaño del archivo", key = "reduce_fps_check"
            )

        st.markdown("---")
        
        # Botón de reset mejorado
        if st.button("🔄 Nuevo Análisis", key = "reset_button", use_container_width = True, type = "primary"):
            reset_processing_state()
            st.rerun()
        
        render_sidebar_metrics()

def render_sidebar_metrics():
    """
    Muestra métricas rápidas en la barra lateral
    """
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