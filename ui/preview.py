import os
import streamlit as st
from utils.helpers import format_time
from enhanced_video import zip_results
from ui.dashboard import create_metrics_dashboard
from utils.video import display_video_with_fallback, create_download_link

def show_results(result):
    """Muestra los resultados del procesamiento con dashboard integrado"""
    clips = result.get("clips", [])
    summary = result.get("summary", {})
    clips_meta = summary.get("clips_meta", [])

    if not clips:
        st.warning("⚠️ No se generaron clips con los criterios seleccionados.")
        st.info("💡 Intenta ajustar los parámetros de análisis o reducir los criterios.")
        return

    st.success(f"✅ Video procesado correctamente! Se generaron {len(clips)} clips.")

    # === DASHBOARD PRINCIPAL ===
    create_metrics_dashboard(result)
    
    st.markdown("---")
    
    # === VISTA PREVIA DE CLIPS (MEJORADA) ===
    st.subheader("🎬 Vista Previa - Top 3 Clips por Engagement")
    
    # Ordenar clips por engagement score
    clips_with_scores = []
    for i, meta in enumerate(clips_meta):
        emotion_stats = meta.get("detections", {}).get("emotion_stats", {})
        positive_count = int(emotion_stats.get("happy", 0) + emotion_stats.get("surprise", 0))
        total_emotion_count = sum(int(v) for v in emotion_stats.values())
        engagement_score = (positive_count / total_emotion_count * 100) if total_emotion_count > 0 else 0
        clips_with_scores.append((i, engagement_score, meta))
    
    # Ordenar por engagement score
    top_clips = sorted(clips_with_scores, key = lambda x: x[1], reverse = True)[:3]
    
    if top_clips:
        preview_cols = st.columns(len(top_clips))
        
        for col_idx, (clip_idx, engagement, clip_meta) in enumerate(top_clips):
            with preview_cols[col_idx]:
                st.write(f"**Top {col_idx+1} Clip** (Engagement: {engagement:.1f}%)")
                
                clip_name = clip_meta["filename"]
                clip_path = os.path.join(os.path.dirname(clips[0]), clip_name)

                success = display_video_with_fallback(clip_path, unique_key=f"top_preview_{col_idx}")
                
                if success:
                    st.caption(f"⏰ {format_time(clip_meta.get('start_time', 0))}")
                    
                    # Calcular total de personas de forma segura
                    detections = clip_meta.get('detections', {})
                    total_people = (
                        int(detections.get('students_sitting', 0)) + int(detections.get('students_standing', 0)) + int(detections.get('teachers_sitting', 0)) + int(detections.get('teachers_standing', 0))
                    )
                    st.caption(f"👥 Personas: {total_people}")
                    
                    create_download_link(clip_path, clip_name, f"top_preview_download_{col_idx}")
                else:
                    st.error(f"No se pudo reproducir el clip")

    # === TODOS LOS CLIPS ===
    with st.expander("🎞️ Ver todos los clips individualmente", expanded = False):
        for i, meta in enumerate(clips_meta):
            clip_path = os.path.join(os.path.dirname(clips[0]), meta["filename"])
            
            st.markdown(f"### 🎬 Clip {i+1}: {meta['filename']}")

            col_video, col_info = st.columns([2, 1])
            
            with col_video:
                success = display_video_with_fallback(clip_path, unique_key = f"full_clip_{i}")
                if not success:
                    st.warning("⚠️ No se pudo cargar este clip para reproducción")

            with col_info:
                st.write(f"⏰ **Inicio:** {format_time(meta['start_time'])}")
                st.write(f"⏱️ **Duración:** {float(meta['duration']):.1f}s")
                
                if "detections" in meta:
                    detections = meta["detections"]
                    st.write("👥 **Detecciones:**")
                    st.write(f"- Estudiantes: {int(detections.get('students_sitting', 0))} sentados, {int(detections.get('students_standing', 0))} parados")
                    st.write(f"- Maestros: {int(detections.get('teachers_sitting', 0))} sentados, {int(detections.get('teachers_standing', 0))} parados")
                    
                    if "emotion_stats" in detections:
                        st.write("😊 **Emociones detectadas:**")
                        for emotion, count in detections["emotion_stats"].items():
                            if int(count) > 0:
                                st.write(f"- {emotion}: {int(count)}")

                if os.path.exists(clip_path):
                    create_download_link(clip_path, meta["filename"], f"individual_download_{i}")
            
            st.markdown("---")

    # === DESCARGAS ===
    st.subheader("📦 Descargar Resultados")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        try:
            output_dir = os.path.dirname(clips[0]) if clips else ""
            if output_dir:
                zip_data = zip_results(output_dir)
                st.download_button(
                    label = "📥 Descargar todos los clips (ZIP)",
                    data = zip_data,
                    file_name = f"clips_results_{st.session_state.uploaded_file_name.split('.')[0] if st.session_state.uploaded_file_name else 'video'}.zip",
                    mime = "application/zip",
                    use_container_width = True,
                    key = "download_zip"
                )
        except Exception as e:
            st.error(f"Error creando ZIP: {str(e)}")
    
    with col_download2:
        metadata_path = result.get("metadata_path")
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                json_data = f.read()
            st.download_button(
                label = "📊 Descargar metadatos (JSON)",
                data = json_data,
                file_name = f"metadata_{st.session_state.uploaded_file_name.split('.')[0] if st.session_state.uploaded_file_name else 'video'}.json",
                mime = "application/json",
                use_container_width = True,
                key = "download_json"
            )