# src/transcriber.py

from google.cloud import speech
from google.api_core import exceptions
import datetime

def format_timedelta(td: datetime.timedelta) -> str:
    """Convierte un objeto timedelta a un formato de tiempo MM:SS.ms."""
    minutes, seconds = divmod(td.total_seconds(), 60)
    return f"{int(minutes):02d}:{int(seconds):02d}.{td.microseconds // 1000:03d}"

def transcribir_audio_largo_desde_gcs(gcs_uri, idioma="es-419"):
    """
    Transcribe, diariza y añade marcas de tiempo a un archivo de audio LARGO desde GCS.
    """
    print("--- [Módulo Transcriptor] Iniciando transcripción con Timestamps y Diarización ---")
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        config = speech.RecognitionConfig(
            # --- CAMBIO: Especificamos que la codificación de audio es FLAC ---
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=idioma,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            diarization_config=speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True, min_speaker_count=2, max_speaker_count=6
            ),
        )

        print("Enviando solicitud de operación de larga duración...")
        operation = client.long_running_recognize(config=config, audio=audio)

        print("Esperando a que la operación se complete...")
        response = operation.result(timeout=3600) # Timeout de 1 hora
        print("Respuesta recibida.")

        if not response.results or not response.results[-1].alternatives:
             print("La API no devolvió resultados para la transcripción.")
             return ""

        # --- LÓGICA PARA AGRUPAR PALABRAS EN FRASES ---
        result = response.results[-1]
        words_info = result.alternatives[0].words
        
        transcript_builder = []
        current_speaker_tag = None
        
        sentence_text = ""
        sentence_start_time = None

        for word_info in words_info:
            # Si el hablante cambia, forzamos el cierre de la frase anterior
            if word_info.speaker_tag != current_speaker_tag:
                if sentence_text: # Si hay una frase en construcción, la guardamos
                    # El tiempo final es el de la palabra ANTERIOR a la actual
                    sentence_end_time = words_info[words_info.index(word_info) - 1].end_time
                    line = f"[{format_timedelta(sentence_start_time)} - {format_timedelta(sentence_end_time)}] {sentence_text.strip()}"
                    transcript_builder.append(line)
                
                # Reiniciamos para el nuevo hablante
                current_speaker_tag = word_info.speaker_tag
                transcript_builder.append(f"\n\nHablante {current_speaker_tag}:")
                sentence_text = ""
                sentence_start_time = None

            # Si es la primera palabra de una nueva frase, guardamos su tiempo de inicio
            if not sentence_start_time:
                sentence_start_time = word_info.start_time

            # Añadimos la palabra actual a la frase en construcción
            sentence_text += f"{word_info.word} "
            
            # Si la palabra termina con un signo de puntuación, cerramos la frase
            if word_info.word.endswith(('.', '?', '!')):
                sentence_end_time = word_info.end_time
                line = f"[{format_timedelta(sentence_start_time)} - {format_timedelta(sentence_end_time)}] {sentence_text.strip()}"
                transcript_builder.append(line)
                
                # Reiniciamos para la siguiente frase
                sentence_text = ""
                sentence_start_time = None

        # Guardar la última frase si el audio termina sin puntuación
        if sentence_text:
            last_word_end_time = words_info[-1].end_time
            line = f"[{format_timedelta(sentence_start_time)} - {format_timedelta(last_word_end_time)}] {sentence_text.strip()}"
            transcript_builder.append(line)
            
        return "\n".join(transcript_builder)

    except Exception as e:
        print(f"Ocurrió un error inesperado durante la transcripción: {e}")
        return None