# src/audio_extractor.py

import os
import ffmpeg

def extraer_audio(input_video_path, output_audio_path):
    """
    Extrae el audio de un archivo de video y lo guarda en formato WAV.

    Args:
        input_video_path (str): Ruta al archivo de video de entrada.
        output_audio_path (str): Ruta donde se guardará el audio extraído.

    Returns:
        str: La ruta al archivo de audio si la extracción fue exitosa, None en caso contrario.
    """
    print("--- [Módulo Extractor] Iniciando extracción de audio ---")
    
    # Asegurarse de que el directorio de salida exista
    output_directory = os.path.dirname(output_audio_path)
    os.makedirs(output_directory, exist_ok=True)

    try:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"El archivo de video no se encuentra en la ruta: {input_video_path}")

        print(f"Procesando: {os.path.basename(input_video_path)}")
        (
            ffmpeg
            .input(input_video_path)
            # --- CAMBIO: Volvemos a pcm_s16le para generar un archivo WAV ---
            .output(output_audio_path, 
                    acodec='pcm_s16le', 
                    ac=1, 
                    ar='16000',
                    )
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print(f"¡Éxito! Audio WAV guardado en: {output_audio_path}\n")
        return output_audio_path

    except FileNotFoundError as e:
        print(f"Error de archivo: {e}")
        return None
    except ffmpeg.Error as e:
        print(f"Ocurrió un error al ejecutar FFmpeg: {e.stderr.decode('utf8')}")
        return None