# src/transcribir_audio.py

from google.cloud import speech
import os

def transcribir_archivo_audio(ruta_audio):
    """
    Transcribe un archivo de audio utilizando la API de Google Cloud Speech-to-Text.

    Args:
        ruta_audio (str): La ruta al archivo de audio local (p. ej., MP3, WAV, FLAC).

    Returns:
        str: El texto transcrito, o None si ocurre un error.
    """
    try:
        # Crea una instancia del cliente de Speech-to-Text
        client = speech.SpeechClient()

        print(f"Cargando archivo de audio para transcribir: '{ruta_audio}'...")

        # Carga el archivo de audio en memoria
        with open(ruta_audio, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        # Configura la solicitud de reconocimiento
        # Asegúrate de que el language_code coincida con el idioma del audio
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3, # Cambia si tu audio es WAV, FLAC, etc.
            sample_rate_hertz=16000, # Frecuencia de muestreo común. Moviepy puede exportar a esto.
            language_code="es-ES", # Código de idioma (Español de España)
            enable_automatic_punctuation=True # Habilita la puntuación automática
        )

        print("Enviando audio a la API de Google Cloud Speech-to-Text...")
        
        # Detecta y transcribe el habla
        response = client.recognize(config=config, audio=audio)

        print("Transcripción recibida.")

        # Concatena los resultados para obtener la transcripción completa
        transcripcion_completa = ""
        for result in response.results:
            transcripcion_completa += result.alternatives[0].transcript + " "

        return transcripcion_completa.strip()

    except Exception as e:
        print(f"Error al transcribir el audio '{ruta_audio}': {e}")
        return None

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Asumimos que ya has extraído el audio de un video y lo has guardado aquí
    # Por ejemplo, el audio de 'data/categoria_1/video1.mp4' -> 'output/audio/video1.mp3'
    ruta_ejemplo_audio = os.path.join("notebooks", "audio_extraido.wav")
    ruta_salida_texto = os.path.join("output", "transcripciones", "audio_extraido.txt")

    # Para que este ejemplo funcione, crea un archivo de audio de prueba en esa ruta
    if not os.path.exists(ruta_ejemplo_audio):
        print(f"Archivo de audio de ejemplo no encontrado en '{ruta_ejemplo_audio}'.")
        print("Asegúrate de extraer primero el audio de un video y guardarlo ahí.")
    else:
        # 1. Llama a la función de transcripción
        texto_resultado = transcribir_archivo_audio(ruta_ejemplo_audio)

        # 2. Guarda el resultado en un archivo de texto
        if texto_resultado:
            os.makedirs(os.path.dirname(ruta_salida_texto), exist_ok=True)
            with open(ruta_salida_texto, "w", encoding="utf-8") as f:
                f.write(texto_resultado)
            print(f"Transcripción guardada exitosamente en '{ruta_salida_texto}'")
            print("\n--- INICIO DE LA TRANSCRIPCIÓN ---")
            print(texto_resultado)
            print("--- FIN DE LA TRANSCRIPCIÓN ---")