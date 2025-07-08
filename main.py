# main.py

import os
import glob

from src.audio_extractor import extraer_audio
from src.audio_chunker import chunk_audio_with_overlap
from src.gcs_manager import subir_archivo_a_gcs
from src.gemini_transcriber import transcribir_con_gemini 
from src.transcript_stitcher import stitch_transcripts
from src.gemini_analyzer import analizar_con_rag_y_citas
from src.report_parser import extraer_calificaciones
from src.pdf_generator import crear_informe_pdf

if __name__ == "__main__":
    # --- 1. CONFIGURACIÓN ---
    # nombre_video_entrada = "Grabación CE OEI 1 - 2025_05_28 07_53 CST - Recording.mp4"
    subcarpeta_fuente = "mic_solapa" 
    ruta_prompt = "prompts/generacion_diagnostico.txt"
    
    ID_PROYECTO = "g-tele-educacion-dev-prj-d18a"
    REGION_GCP = "us-east1"
    GCS_BUCKET_NAME = "ia_tele_educacion"
    RAG_CORPUS_PATH = "projects/g-tele-educacion-dev-prj-d18a/locations/us-central1/ragCorpora/4611686018427387904"
    
    # --- GENERACIÓN DE RUTAS ---
    ruta_base = os.getcwd()
    directorio_videos = os.path.join(ruta_base, "data", subcarpeta_fuente)
    
    # Usamos glob para encontrar todos los archivos .mp4 en ese directorio
    lista_videos_a_procesar = glob.glob(os.path.join(directorio_videos, '*.mp4'))

    if not lista_videos_a_procesar:
        print(f"No se encontraron archivos .mp4 en la carpeta: {directorio_videos}")
    else:
        print(f"Se encontraron {len(lista_videos_a_procesar)} videos para procesar.")
        
    # Iteramos sobre cada video encontrado
    for ruta_video_local in lista_videos_a_procesar:    
        
        nombre_video_entrada = os.path.basename(ruta_video_local)
        nombre_base = os.path.splitext(nombre_video_entrada)[0]
        
        print(f"\n=====================================================================")
        print(f">>> INICIANDO PIPELINE COMPLETO PARA: {nombre_video_entrada} <<<")
        print(f"=====================================================================")
    
        ruta_video_local = os.path.join(ruta_base, "data", subcarpeta_fuente, nombre_video_entrada)
        ruta_audio_local_completo = os.path.join(ruta_base, "output", "audio", subcarpeta_fuente, f"{nombre_base}.wav")
        directorio_salida_chunks = os.path.join(ruta_base, "output", "audio_chunks", nombre_base)
        # Cambié el nombre de ruta_informe_final a ruta_texto_salida para que coincida con el guardado
        ruta_informe_final = os.path.join(ruta_base, "output", "analisis_qa", f"ANALISIS - {nombre_base}.md")
        ruta_texto_salida = os.path.join(ruta_base, "output", "transcripciones_finales", f"TRANSCRIPCION - {nombre_base}.txt")
        
        # --- INICIO DEL PIPELINE ---
        print(f"\n>>> INICIANDO PIPELINE COMPLETO PARA: {nombre_video_entrada} <<<")

        audio_extraido_path = extraer_audio(ruta_video_local, ruta_audio_local_completo)

        if not audio_extraido_path:
            print("FALLO EN FASE 1: No se pudo extraer el audio. Proceso detenido.")
        else:
            lista_chunks_locales = chunk_audio_with_overlap(
                input_file=ruta_audio_local_completo,
                output_dir=directorio_salida_chunks
            )
            if not lista_chunks_locales:
                print("No se crearon chunks. Terminando el proceso.")
            else:
                transcripciones_de_chunks = []
                for i, chunk_path in enumerate(lista_chunks_locales):
                    print(f"\n--- Procesando Chunk {i+1}/{len(lista_chunks_locales)}: {os.path.basename(chunk_path)} ---")
                    
                    # Ahora la ruta en GCS incluye el nombre base del video para ser única
                    ruta_destino_gcs = f"audio_chunks/{nombre_base}/{os.path.basename(chunk_path)}"
                    
                    gcs_uri = subir_archivo_a_gcs(chunk_path, GCS_BUCKET_NAME, ruta_destino_gcs)
                    

                    if gcs_uri:
                        texto_chunk = transcribir_con_gemini(ID_PROYECTO, REGION_GCP, gcs_uri)
                        if texto_chunk:
                            transcripciones_de_chunks.append(texto_chunk)
                        else:
                            print(f"El chunk {i+1} no pudo ser transcrito.")
                    else:
                        print(f"Fallo al subir el chunk {i+1}.")

                if transcripciones_de_chunks:
                    transcripcion_final = stitch_transcripts(transcripciones_de_chunks)
                    
                    # Usamos la variable ruta_texto_salida definida al principio
                    os.makedirs(os.path.dirname(ruta_texto_salida), exist_ok=True)
                    
                    with open(ruta_texto_salida, "w", encoding="utf-8") as f:
                        f.write(transcripcion_final)
                        
                    
                    print(f"\nTranscripción completa guardada en: {ruta_texto_salida}")


                # --- NUEVA FASE 5: ANÁLISIS Q&A CON GEMINI ---
                if transcripcion_final:
                    respuestas_gemini = analizar_con_rag_y_citas(
                        project_id=ID_PROYECTO,
                        location=REGION_GCP,
                        rag_corpus_path=RAG_CORPUS_PATH,
                        ruta_prompt=ruta_prompt,
                        transcripcion_texto=transcripcion_final
                    )

                    if not respuestas_gemini:
                        print("FALLO EN FASE 5: No se generó el informe de Gemini.")
                    else:
                        # --- NUEVA FASE 6: PARSEAR CALIFICACIONES ---
                        calificaciones, promedio = extraer_calificaciones(respuestas_gemini)

                        # --- NUEVA FASE 7: GENERAR EL INFORME EN PDF ---
                        ruta_informe_pdf = os.path.join(
                            ruta_base, "output", "informes_pdf", f"INFORME - {nombre_base}.pdf"
                        )
                        
                        crear_informe_pdf(
                            titulo=f"Informe de Tutoría: {nombre_base}",
                            informe_texto=respuestas_gemini,
                            calificaciones=calificaciones,
                            promedio=promedio,
                            ruta_salida=ruta_informe_pdf
                        )
                        
                        print(f"\n\n>>> ¡PIPELINE FINALIZADO CON ÉXITO! <<<")
                        print(f"El informe final en PDF ha sido guardado en: {ruta_informe_pdf}")
                else:
                    print("No se generaron transcripciones para unir.")