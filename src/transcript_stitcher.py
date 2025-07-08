# src/transcript_stitcher.py

from difflib import SequenceMatcher

def stitch_transcripts(transcripts: list) -> str:
    """
    Une múltiples transcripciones, eliminando el texto de traslape.

    Args:
        transcripts: Una lista de strings, donde cada string es la transcripción de un chunk.

    Returns:
        Un único string con la transcripción completa y unida.
    """
    print("--- [Stitcher] Uniendo transcripciones y eliminando traslapes ---")
    if not transcripts:
        return ""

    full_transcript = transcripts[0]
    
    for i in range(len(transcripts) - 1):
        text1 = full_transcript
        text2 = transcripts[i+1]
        
        # Encontrar el punto de coincidencia entre el final del texto acumulado y el inicio del nuevo
        match = SequenceMatcher(None, text1, text2).find_longest_match(0, len(text1), 0, len(text2))
        
        # La parte del texto nuevo que no se traslapa
        non_overlapping_part = text2[match.b + match.size:]
        
        full_transcript += non_overlapping_part
        
    print("Unión completada.")
    return full_transcript