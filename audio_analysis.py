import os
import tempfile
import traceback
import streamlit as st
from utils.video import save_uploaded_file
from utils.session import initialize_session_state

# Importar módulos del pipeline de audio
from src.audio_extractor import extraer_audio
from src.audio_chunker import chunk_audio_with_overlap
from src.gcs_manager import subir_archivo_a_gcs
from src.gemini_transcriber import transcribir_con_gemini 
from src.transcript_stitcher import stitch_transcripts
from src.gemini_analyzer import analizar_con_rag_y_citas
from src.report_parser import extraer_calificaciones
from src.pdf_generator import crear_informe_pdf

# Configuración GCP
ID_PROYECTO = "g-tele-educacion-dev-prj-d18a"
REGION_GCP = "us-east1"
GCS_BUCKET_NAME = "ia_tele_educacion"
RAG_CORPUS_PATH = "projects/g-tele-educacion-dev-prj-d18a/locations/us-central1/ragCorpora/4611686018427387904"

UPLOAD_FOLDER = "temp_uploads"

def process_audio_pipeline(video_path, progress_bar, status_text):
    """
    Procesa el pipeline completo de análisis de audio
    """
    try:
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        ruta_base = os.getcwd()
        
        # Configurar rutas de salida
        ruta_audio_local_completo = os.path.join(ruta_base, "output", "audio", "streamlit", f"{nombre_base}.wav")
        directorio_salida_chunks = os.path.join(ruta_base, "output", "audio_chunks", nombre_base)
        ruta_texto_salida = os.path.join(ruta_base, "output", "transcripciones_finales", f"TRANSCRIPCION - {nombre_base}.txt")
        ruta_prompt = "prompts/generacion_diagnostico.txt"
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(ruta_audio_local_completo), exist_ok=True)
        os.makedirs(directorio_salida_chunks, exist_ok=True)
        os.makedirs(os.path.dirname(ruta_texto_salida), exist_ok=True)
        
        # Fase 1: Extraer audio
        progress_bar.progress(10)
        status_text.text("🎵 Extrayendo audio del video...")
        
        audio_extraido_path = extraer_audio(video_path, ruta_audio_local_completo)
        
        if not audio_extraido_path:
            raise Exception("No se pudo extraer el audio del video")
        
        # Fase 2: Crear chunks de audio
        progress_bar.progress(20)
        status_text.text("✂️ Dividiendo audio en chunks...")
        
        lista_chunks_locales = chunk_audio_with_overlap(
            input_file=ruta_audio_local_completo,
            output_dir=directorio_salida_chunks
        )
        
        if not lista_chunks_locales:
            raise Exception("No se pudieron crear chunks de audio")
        
        # Fase 3: Transcribir chunks
        progress_bar.progress(30)
        status_text.text("🎙️ Transcribiendo chunks de audio...")
        
        transcripciones_de_chunks = []
        total_chunks = len(lista_chunks_locales)
        
        for i, chunk_path in enumerate(lista_chunks_locales):
            # Actualizar progreso
            chunk_progress = 30 + (40 * (i + 1) / total_chunks)
            progress_bar.progress(int(chunk_progress))
            status_text.text(f"🎙️ Transcribiendo chunk {i+1}/{total_chunks}...")
            
            # Subir chunk a GCS
            ruta_destino_gcs = f"audio_chunks/{nombre_base}/{os.path.basename(chunk_path)}"
            gcs_uri = subir_archivo_a_gcs(chunk_path, GCS_BUCKET_NAME, ruta_destino_gcs)
            
            if gcs_uri:
                texto_chunk = transcribir_con_gemini(ID_PROYECTO, REGION_GCP, gcs_uri)
                if texto_chunk:
                    transcripciones_de_chunks.append(texto_chunk)
                else:
                    st.warning(f"No se pudo transcribir el chunk {i+1}")
            else:
                st.warning(f"No se pudo subir el chunk {i+1} a GCS")
        
        if not transcripciones_de_chunks:
            raise Exception("No se pudo transcribir ningún chunk")
        
        # Fase 4: Unir transcripciones
        progress_bar.progress(70)
        status_text.text("🔗 Uniendo transcripciones...")
        
        transcripcion_final = stitch_transcripts(transcripciones_de_chunks)
        
        # Guardar transcripción
        with open(ruta_texto_salida, "w", encoding="utf-8") as f:
            f.write(transcripcion_final)
        
        # Fase 5: Análisis con RAG
        progress_bar.progress(80)
        status_text.text("🧠 Analizando con IA...")
        
        respuestas_gemini = analizar_con_rag_y_citas(
            project_id=ID_PROYECTO,
            location=REGION_GCP,
            rag_corpus_path=RAG_CORPUS_PATH,
            ruta_prompt=ruta_prompt,
            transcripcion_texto=transcripcion_final
        )
        
        if not respuestas_gemini:
            raise Exception("No se pudo generar el análisis con Gemini")
        
        # Fase 6: Extraer calificaciones
        progress_bar.progress(90)
        status_text.text("📊 Extrayendo calificaciones...")
        
        calificaciones, promedio = extraer_calificaciones(respuestas_gemini)
        
        # Fase 7: Generar PDF
        progress_bar.progress(95)
        status_text.text("📄 Generando informe PDF...")
        
        ruta_informe_pdf = os.path.join(
            ruta_base, "output", "informes_pdf", f"INFORME - {nombre_base}.pdf"
        )
        
        os.makedirs(os.path.dirname(ruta_informe_pdf), exist_ok=True)
        
        crear_informe_pdf(
            titulo=f"Informe de Tutoría: {nombre_base}",
            informe_texto=respuestas_gemini,
            calificaciones=calificaciones,
            promedio=promedio,
            ruta_salida=ruta_informe_pdf
        )
        
        progress_bar.progress(100)
        status_text.text("✅ ¡Procesamiento completado!")
        
        return {
            "transcripcion": transcripcion_final,
            "analisis": respuestas_gemini,
            "calificaciones": calificaciones,
            "promedio": promedio,
            "ruta_transcripcion": ruta_texto_salida,
            "ruta_pdf": ruta_informe_pdf,
            "nombre_base": nombre_base
        }
        
    except Exception as e:
        raise Exception(f"Error en el pipeline de audio: {str(e)}")

def show_audio_results(results):
    """
    Muestra los resultados del análisis de audio
    """
    st.success("🎉 ¡Análisis de audio completado exitosamente!")
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Promedio General", f"{results['promedio']:.1f}/10" if results['promedio'] else "N/A")
    
    with col2:
        st.metric("Aspectos Evaluados", len(results['calificaciones']))
    
    with col3:
        transcripcion_palabras = len(results['transcripcion'].split()) if results['transcripcion'] else 0
        st.metric("Palabras Transcritas", transcripcion_palabras)
    
    # Tabs para diferentes secciones
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Calificaciones", "📝 Transcripción", "🧠 Análisis", "📁 Archivos"])
    
    with tab1:
        st.subheader("📊 Calificaciones por Aspecto")
        if results['calificaciones']:
            for aspecto, calificacion in results['calificaciones'].items():
                # Crear una barra de progreso visual
                progress = calificacion / 10
                color = "🟢" if calificacion >= 7 else "🟡" if calificacion >= 5 else "🔴"
                st.metric(f"{color} {aspecto}", f"{calificacion}/10")
        else:
            st.info("No se encontraron calificaciones en el análisis")
    
    with tab2:
        st.subheader("📝 Transcripción Completa")
        if results['transcripcion']:
            st.text_area("Transcripción", results['transcripcion'], height=400)
            
            # Botón para descargar transcripción
            st.download_button(
                label="⬇️ Descargar Transcripción",
                data=results['transcripcion'],
                file_name=f"transcripcion_{results['nombre_base']}.txt",
                mime="text/plain"
            )
        else:
            st.warning("No se pudo obtener la transcripción")
    
    with tab3:
        st.subheader("🧠 Análisis Detallado")
        if results['analisis']:
            st.markdown(results['analisis'])
        else:
            st.warning("No se pudo obtener el análisis")
    
    with tab4:
        st.subheader("📁 Archivos Generados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Transcripción:**\n{results['ruta_transcripcion']}")
            
        with col2:
            st.info(f"**Informe PDF:**\n{results['ruta_pdf']}")
        
        # Botón para descargar PDF si existe
        if os.path.exists(results['ruta_pdf']):
            with open(results['ruta_pdf'], "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Descargar Informe PDF",
                    data=pdf_file.read(),
                    file_name=f"informe_{results['nombre_base']}.pdf",
                    mime="application/pdf"
                )

def audio_analysis_page():
    """
    Página principal para el análisis de audio
    """
    # Inicializar session state
    if 'audio_processing_complete' not in st.session_state:
        st.session_state.audio_processing_complete = False
    if 'audio_results' not in st.session_state:
        st.session_state.audio_results = None
    if 'audio_uploaded_file_name' not in st.session_state:
        st.session_state.audio_uploaded_file_name = None
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🎵 Análisis de Audio Educativo</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Transcripción y Análisis Inteligente de Videos con IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuración del Análisis")
        
        # Información del pipeline
        with st.expander("🔄 Pipeline de Procesamiento", expanded=True):
            st.markdown("""
            **Fases del análisis:**
            1. 🎵 Extracción de audio
            2. ✂️ Segmentación en chunks
            3. 🎙️ Transcripción con Gemini
            4. 🔗 Unión de transcripciones
            5. 🧠 Análisis con RAG
            6. 📊 Extracción de calificaciones
            7. 📄 Generación de informe PDF
            """)
        
        # Configuración GCP
        with st.expander("☁️ Configuración GCP", expanded=False):
            st.text_input("ID Proyecto", value=ID_PROYECTO, disabled=True)
            st.text_input("Región", value=REGION_GCP, disabled=True)
            st.text_input("GCS Bucket", value=GCS_BUCKET_NAME, disabled=True)
        
        st.markdown("---")
        
        # Botón de reset
        if st.button("🔄 Nuevo Análisis", use_container_width=True, type="primary"):
            st.session_state.audio_processing_complete = False
            st.session_state.audio_results = None
            st.session_state.audio_uploaded_file_name = None
            st.rerun()
        
        # Mostrar métricas si hay resultados
        if st.session_state.audio_processing_complete and st.session_state.audio_results:
            st.markdown("### 📊 Métricas Rápidas")
            results = st.session_state.audio_results
            
            st.metric("Promedio", f"{results['promedio']:.1f}/10" if results['promedio'] else "N/A")
            st.metric("Aspectos", len(results['calificaciones']))
            palabras = len(results['transcripcion'].split()) if results['transcripcion'] else 0
            st.metric("Palabras", palabras)
    
    # Mostrar resultados si están disponibles
    if st.session_state.audio_processing_complete and st.session_state.audio_results:
        show_audio_results(st.session_state.audio_results)
        return
    
    # Área principal - Subir archivo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Subir Video para Análisis de Audio")
        uploaded_file = st.file_uploader(
            "Selecciona un video para analizar su audio", 
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
            help="El sistema extraerá y analizará el audio del video",
            key="audio_file_uploader"
        )
        
        # Reset si se cambia el archivo
        if uploaded_file and st.session_state.audio_uploaded_file_name != uploaded_file.name:
            st.session_state.audio_processing_complete = False
            st.session_state.audio_results = None
            st.session_state.audio_uploaded_file_name = uploaded_file.name
    
    with col2:
        st.subheader("📄 Información del Archivo")
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Tamaño", f"{file_size_mb:.1f} MB")
            st.metric("Tipo", uploaded_file.type)
            st.metric("Nombre", uploaded_file.name)
        else:
            st.info("Esperando archivo...")
    
    # Procesar video
    if uploaded_file and not st.session_state.audio_processing_complete:
        # Guardar archivo subido
        with st.spinner("📥 Guardando video..."):
            try:
                video_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
                st.success(f"✅ Video guardado: {os.path.basename(video_path)}")
            except Exception as e:
                st.error(f"❌ Error al guardar el video: {str(e)}")
                return
        
        # Botón para iniciar procesamiento
        if st.button("🚀 Iniciar Análisis de Audio", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Procesar pipeline de audio
                results = process_audio_pipeline(video_path, progress_bar, status_text)
                
                # Guardar resultados en session state
                st.session_state.audio_results = results
                st.session_state.audio_processing_complete = True
                
                st.rerun()
                
            except Exception as e:
                st.error("❌ Error durante el procesamiento del audio.")
                st.error(f"Detalles del error: {str(e)}")
                with st.expander("Ver traceback completo"):
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    audio_analysis_page()