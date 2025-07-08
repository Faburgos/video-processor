# gemini_transcriber.py

import os
from google import genai
from google.genai import types

def transcribir_con_gemini(project_id, location, gcs_uri):
    """
    Envía un archivo de audio en GCS a Gemini 1.5 Pro para su transcripción y diarización.

    Args:
        project_id (str): ID de tu proyecto de GCP.
        location (str): Región de tu proyecto (ej. "us-central1").
        gcs_uri (str): La URI del archivo de audio en GCS.

    Returns:
        str: La transcripción generada por el modelo.
    """
    print("--- Iniciando transcripción con Gemini 2.5 Pro ---")
    
    # Inicializar Vertex AI
    client = genai.Client(
        vertexai=True,
        project="g-tele-educacion-dev-prj-d18a",
        location="us-east1",
        )

    # Cargar el modelo
    # Asegúrate de usar un modelo que soporte audio. Gemini 1.5 Pro es la mejor opción.
    model = "gemini-2.5-pro"
    
    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        seed = 0,
        max_output_tokens = 65535,
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
        thinking_config=types.ThinkingConfig(
        thinking_budget=-1,
        ),
    )

    # 3. Preparar el contenido: el audio y el prompt
    # Esta es la parte clave que faltaba en tu código de ejemplo.
    audio1 = types.Part.from_uri(
        file_uri=gcs_uri,
        mime_type="audio/wav",
    )
    text1 = types.Part.from_text(text="""Por favor, transcribe el siguiente audio con la mayor precisión posible.
                                 El audio de una clase en El Salvador en español entre los estudiantes y el maestro.
                                 Tu tarea es identificar y separar a los diferentes hablantes en el texto, incluyendo cuando hablen todos al mismo tiempo
                                 coloca cuando el audio es inentendible.
                                 Formatea la transcripción como si fuera el guion de un diálogo,
                                 indicando cuándo cambia de hablante.""")

    contents = [
        types.Content(
        role="user",
        parts=[
            audio1,
            text1
        ]
        )
    ]

    print(f"Enviando audio '{gcs_uri}' a Gemini. Esto puede tomar un momento...")
    
    try:
        # Generar la respuesta
        response = client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
            )
        print("Respuesta recibida de Gemini.")
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Ocurrió un error al contactar a la API de Gemini: {e}")
        return None
