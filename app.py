import os
import shutil
import tempfile
import traceback
import base64
import streamlit as st
import pandas as pd
from enhanced_video import process_school_video, zip_results

# Constants
UPLOAD_FOLDER = "temp_uploads"

def save_uploaded_file(uploaded_file, save_dir):
    """Guarda el archivo subido en el directorio especificado"""
    os.makedirs(save_dir, exist_ok=True)
    # Limpiar nombre del archivo
    filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.'))
    file_path = os.path.join(save_dir, filename)
    
    # Guardar archivo por chunks para manejar archivos grandes
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
        # Método 1: Cargar como bytes (más confiable)
        video_bytes = load_video_as_bytes(video_path)
        if video_bytes:
            st.video(video_bytes, start_time=0)
            if caption:
                st.caption(caption)
            return True
    except Exception as e:
        st.error(f"Error reproduciendo video: {str(e)}")
    
    return False

def clear_folder(folder):
    """Limpia y recrea una carpeta"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def format_time(seconds):
    """Convierte segundos a formato mm:ss"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def create_download_link(video_path, filename, unique_key):
    """Crea un enlace de descarga para un video individual"""
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        return st.download_button(
            label=f"⬇️ Descargar {filename}",
            data=video_bytes,
            file_name=filename,
            mime="video/mp4",
            key=unique_key,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error creando enlace de descarga: {str(e)}")
        return False

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

def main():
    st.set_page_config(
        page_title="Procesador de Videos", 
        page_icon="🎥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar estado de sesión
    initialize_session_state()
    
    st.title("🎥 Video Processor MVP")
    st.markdown("Extrae clips inteligentes de tus videos largos")

    # Sidebar de configuración (actualizar la sección de análisis de contenido)
    with st.sidebar:
        st.header("⚙️ Configuración del procesamiento")
        
        # Parámetros principales
        interval_min = st.slider("Intervalo entre clips (minutos)", 1, 30, 5, key="interval_slider")
        interval_sec = interval_min * 60
        
        clip_duration = st.slider("Duración del clip (segundos)", 5, 120, 10, key="duration_slider")
        
        st.markdown("---")
        st.subheader("🔍 Análisis de contenido")
        quality_check = st.checkbox("Analizar calidad de imagen", True, key="quality_check")
        motion_check = st.checkbox("Detectar movimiento", True, key="motion_check")
        smart_filter = st.checkbox("Extracción inteligente", True, 
                                help="Solo extrae clips que cumplan TODOS los criterios", key="smart_filter")
        
        st.markdown("---")
        st.subheader("🤖 Detecciones YOLO") 
        yolo_analysis = st.checkbox("Activar análisis YOLO", True, 
                                help="Detecta estudiantes y maestros", key="yolo_analysis")
        
        yolo_in_video = st.checkbox("Mostrar detecciones en el video", True, 
                                help="Dibuja las detecciones directamente en los clips generados", 
                                key="yolo_in_video",
                                disabled=not yolo_analysis)
        
        if yolo_analysis:
            confidence_threshold = st.slider("Umbral de confianza", 0.1, 0.9, 0.5, 0.1,
                                        help="Confianza mínima para mostrar detecciones", 
                                        key="confidence_slider")
        else:
            confidence_threshold = 0.5
        
        st.markdown("---")
        st.subheader("🎬 Compresión opcional")
        scale = st.slider("Escala de resolución", 0.1, 1.0, 1.0, 0.1,
                        help="1.0 = resolución original", key="scale_slider")
        reduce_fps = st.checkbox("Reducir FPS a la mitad", False,
                            help="Reduce el tamaño del archivo", key="reduce_fps_check")
        
        st.markdown("---")
        if st.button("🔄 Nuevo Video", key="reset_button", use_container_width=True):
            reset_processing_state()
            st.rerun()

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Subir Video")
        uploaded_file = st.file_uploader(
            "Selecciona un video", 
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
            help="Máximo 2GB por archivo",
            key="file_uploader"
        )
        
        # Detectar cambio de archivo
        if uploaded_file and st.session_state.uploaded_file_name != uploaded_file.name:
            reset_processing_state()
            st.session_state.uploaded_file_name = uploaded_file.name
        
        if not uploaded_file and not st.session_state.processing_complete:
            st.info("👆 Por favor, sube un archivo de video para comenzar.")
            return

    with col2:
        st.subheader("📊 Información del archivo")
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Tamaño", f"{file_size_mb:.1f} MB")
            st.metric("Tipo", uploaded_file.type)

    # Si ya se procesó el video, mostrar resultados
    if st.session_state.processing_complete and st.session_state.clips_data:
        show_results(st.session_state.clips_data)
        return

    # Procesar video solo si no se ha procesado aún
    if uploaded_file and not st.session_state.video_processed:
        # Guardar video
        with st.spinner("📥 Guardando video..."):
            try:
                video_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
                st.success(f"✅ Video guardado: {os.path.basename(video_path)}")
            except Exception as e:
                st.error(f"❌ Error al guardar el video: {str(e)}")
                return

        # Crear carpeta clips temporal
        output_dir = tempfile.mkdtemp(prefix="clips_", suffix="_temp")

        # Botón de procesamiento
        if st.button("🚀 Procesar Video", type="primary", use_container_width=True, key="process_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎬 Procesando el video..."):
                try:
                    progress_bar.progress(10)
                    
                    result = process_school_video(
                        video_path = video_path,
                        model_path = "models/best.pt", 
                        interval_seconds = interval_sec,
                        output_folder = output_dir,
                        clip_duration_sec = clip_duration,
                        analyze_quality = quality_check,
                        detect_motion = motion_check,
                        smart_extraction = smart_filter,
                        scale = scale,
                        reduce_fps = reduce_fps
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("¡Procesamiento completado!")
                    
                    # Guardar resultados en session state
                    st.session_state.clips_data = result
                    st.session_state.processing_complete = True
                    st.session_state.video_processed = True
                    
                    # Forzar actualización de la interfaz
                    st.rerun()
                    
                except Exception as e:
                    st.error("❌ Error durante el procesamiento.")
                    st.error(f"Detalles del error: {str(e)}")
                    with st.expander("Ver traceback completo"):
                        st.code(traceback.format_exc())
                    return

def show_results(result):
    """Muestra los resultados del procesamiento"""
    clips = result.get("clips", [])
    summary = result.get("summary", {})
    clips_meta = summary.get("clips_meta", [])

    if not clips:
        st.warning("⚠️ No se generaron clips con los criterios seleccionados.")
        st.info("💡 Intenta ajustar los parámetros de análisis o reducir los criterios.")
        return

    st.success(f"✅ Video procesado correctamente! Se generaron {len(clips)} clips.")

    # Dashboard de métricas
    st.subheader("📊 Estadísticas del Procesamiento")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎞️ Clips generados", summary.get("total_clips", 0))
    with col2:
        st.metric("🎯 Promedio nitidez", f"{summary.get('avg_sharpness', 0):.1f}")
    with col3:
        st.metric("🏃 Promedio movimiento", f"{summary.get('avg_motion_score', 0):.3f}")
    with col4:
        if summary.get("best_quality_clip"):
            st.metric("⭐ Mejor clip", os.path.splitext(summary.get("best_quality_clip", "N/A"))[0])

    # Vista previa de clips (primeros 3)
    st.subheader("🎬 Vista Previa - Primeros 3 Clips")
    
    if len(clips) > 0:
        preview_cols = st.columns(min(3, len(clips)))
        
        for i in range(min(3, len(clips))):
            with preview_cols[i]:
                st.write(f"**Clip {i+1}**")
                
                clip_name = os.path.basename(clips[i])
                clip_meta = next((m for m in clips_meta if m["filename"] == clip_name), {})

                # Usar el nuevo sistema de reproducción
                success = display_video_with_fallback(clips[i], unique_key=f"preview_{i}")
                
                if success:
                    st.caption(f"⏰ {format_time(clip_meta.get('start_time', 0))}")
                    st.caption(f"🎯 Nitidez: {clip_meta.get('quality_score', 0):.1f}")
                    create_download_link(clips[i], clip_name, f"preview_download_{i}")
                else:
                    st.error(f"No se pudo reproducir el clip {i+1}")

    # Tabla de todos los clips con metadatos
    st.subheader("📋 Todos los Clips Generados")
    
    if clips_meta:
        df_data = []
        for i, meta in enumerate(clips_meta, 1):
            df_data.append({
                "#": i,
                "Archivo": meta["filename"],
                "Inicio": format_time(meta["start_time"]),
                "Duración": f"{meta['duration']:.1f}s",
                "Nitidez": f"{meta['quality_score']:.1f}",
                "Movimiento": f"{meta['motion_score']:.3f}"
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

    # Sección expandible con todos los clips
    with st.expander("🎞️ Ver todos los clips individualmente", expanded=False):
        for i, meta in enumerate(clips_meta):
            clip_path = os.path.join(os.path.dirname(clips[0]), meta["filename"])
            
            st.markdown(f"### 🎬 Clip {i+1}: {meta['filename']}")

            col_video, col_info = st.columns([2, 1])
            
            with col_video:
                success = display_video_with_fallback(clip_path, unique_key=f"full_clip_{i}")
                if not success:
                    st.warning("⚠️ No se pudo cargar este clip para reproducción")

            with col_info:
                st.write(f"⏰ **Inicio:** {format_time(meta['start_time'])}")
                st.write(f"⏱️ **Duración:** {meta['duration']:.1f}s")
                st.write(f"🎯 **Nitidez:** {meta['quality_score']:.2f}")
                st.write(f"🏃 **Movimiento:** {meta['motion_score']:.4f}")

                if os.path.exists(clip_path):
                    create_download_link(clip_path, meta["filename"], f"individual_download_{i}")
            
            st.markdown("---")

    # Descargar resultados
    st.subheader("📦 Descargar Resultados")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        try:
            output_dir = os.path.dirname(clips[0]) if clips else ""
            if output_dir:
                zip_data = zip_results(output_dir)
                st.download_button(
                    label="📥 Descargar todos los clips (ZIP)",
                    data=zip_data,
                    file_name=f"clips_results_{st.session_state.uploaded_file_name.split('.')[0] if st.session_state.uploaded_file_name else 'video'}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="download_zip"
                )
        except Exception as e:
            st.error(f"Error creando ZIP: {str(e)}")
    
    with col_download2:
        metadata_path = result.get("metadata_path")
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                json_data = f.read()
            st.download_button(
                label="📊 Descargar metadatos (JSON)",
                data=json_data,
                file_name=f"metadata_{st.session_state.uploaded_file_name.split('.')[0] if st.session_state.uploaded_file_name else 'video'}.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"
            )

    # Información adicional
    with st.expander("ℹ️ Información técnica"):
        st.json({
            "Configuración": {
                "Intervalo entre clips": f"{st.session_state.get('interval_min', 5)} minutos",
                "Duración de clips": f"{st.session_state.get('clip_duration', 10)} segundos",
                "Análisis de calidad": st.session_state.get('quality_check', True),
                "Detección de movimiento": st.session_state.get('motion_check', True),
                "Extracción inteligente": st.session_state.get('smart_filter', True),
                "Escala": st.session_state.get('scale', 1.0),
                "FPS reducido": st.session_state.get('reduce_fps', False)
            },
            "Resultados": {
                "Total de clips": len(clips),
                "Mejor clip (calidad)": summary.get("best_quality_clip", "N/A"),
                "Clip con más movimiento": summary.get("most_detections_clip", "N/A")
            }
        })

if __name__ == "__main__":
    main()