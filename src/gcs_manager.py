# src/gcs_manager.py

import os
from google.cloud import storage
from google.api_core import exceptions

def subir_archivo_a_gcs(ruta_archivo_local, bucket_nombre, destino_blob_nombre):
    """
    Sube un archivo local a un bucket de Google Cloud Storage.

    Args:
        ruta_archivo_local (str): Ruta al archivo en tu máquina.
        bucket_nombre (str): El nombre de tu bucket en GCS (ej. "mi-bucket-de-audios").
        destino_blob_nombre (str): La ruta/nombre que tendrá el archivo dentro del bucket.

    Returns:
        str: La URI de GCS del archivo subido (ej. "gs://mi-bucket/audio.wav"), o None si falla.
    """
    print("--- [Módulo GCS] Iniciando subida a Google Cloud Storage ---")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_nombre)
        blob = bucket.blob(destino_blob_nombre)

        print(f"Subiendo '{ruta_archivo_local}' a 'gs://{bucket_nombre}/{destino_blob_nombre}'...")
        
        # --- CRÍTICO: Mantenemos el timeout de 1 hora para el archivo WAV grande ---
        blob.upload_from_filename(ruta_archivo_local, timeout=600)
        
        gcs_uri = f"gs://{bucket_nombre}/{destino_blob_nombre}"
        print(f"¡Éxito! Archivo disponible en: {gcs_uri}\n")
        return gcs_uri
    except Exception as e:
        print(f"Ocurrió un error inesperado al subir el archivo a GCS: {e}")
        return None