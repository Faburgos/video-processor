# src/audio_chunker.py

import os
import ffmpeg
import math

def chunk_audio_with_overlap(
    input_file: str,
    output_dir: str,
    chunk_duration_sec: int = 300,  # 5 minutos
    overlap_sec: int = 2
) -> list:
    """
    Divide un archivo de audio en chunks con traslape.

    Args:
        input_file: Ruta al archivo de audio de entrada.
        output_dir: Directorio donde se guardarán los chunks.
        chunk_duration_sec: Duración de cada chunk en segundos.
        overlap_sec: Duración del traslape en segundos.

    Returns:
        Una lista ordenada de las rutas a los chunks creados.
    """
    print(f"--- [Chunker] Iniciando división en chunks de {chunk_duration_sec}s con traslape de {overlap_sec}s ---")
    
    if not os.path.exists(input_file):
        print(f"Error: El archivo de entrada no existe en {input_file}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Obtener la duración total del audio
        probe = ffmpeg.probe(input_file)
        total_duration = float(probe['format']['duration'])
        print(f"Duración total del audio: {total_duration:.2f} segundos.")
    except ffmpeg.Error as e:
        print("Error al obtener la duración del audio:", e.stderr)
        return []

    chunk_paths = []
    start_time = 0
    chunk_index = 0
    
    while start_time < total_duration:
        output_filename = os.path.join(output_dir, f"chunk_{chunk_index:04d}.wav")
        
        # Usamos -ss para el inicio y -t para la duración
        (
            ffmpeg
            .input(input_file, ss=start_time)
            .output(output_filename, t=chunk_duration_sec, acodec='pcm_s16le', ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        
        chunk_paths.append(output_filename)
        print(f"Creado chunk: {output_filename}")
        
        # Avanzar el tiempo para el siguiente chunk
        start_time += chunk_duration_sec - overlap_sec
        chunk_index += 1

    print(f"División completada. Se crearon {len(chunk_paths)} chunks.")
    return chunk_paths