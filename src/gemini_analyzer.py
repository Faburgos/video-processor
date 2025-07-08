# src/gemini_analyzer.py

from google import genai
from google.genai import types
import os

def analizar_con_rag_y_citas(project_id: str, location: str, rag_corpus_path: str, ruta_prompt: str, transcripcion_texto: str) -> str:    
    """
    Usa RAG de Vertex AI para analizar una transcripción, cargando el prompt desde un archivo.

    Args:
        project_id: ID de tu proyecto de GCP.
        location: Región de tu proyecto.
        rag_corpus_path: Ruta completa a tu corpus de RAG.
        ruta_prompt: Ruta al archivo .txt que contiene el prompt.
        transcripcion_texto: texto con la transcripcion obtenida.

    Returns:
        El informe generado por Gemini, incluyendo las citas.
    """
    print("--- [Módulo Gemini] Analizando transcripción con apoyo de RAG ---")

    # Inicializar Vertex AI
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        )
    
    model = "gemini-2.5-pro"
    contents = [
        types.Content(
        role="user",
        parts=[
        ]
        )
    ]
    tools = [
        types.Tool(
        retrieval=types.Retrieval(
            vertex_rag_store=types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(
                rag_corpus=rag_corpus_path
                )
            ],
            )
        )
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0.5,
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
        tools = tools,
        thinking_config=types.ThinkingConfig(
        thinking_budget=-1,
        ),
    )

    # Construimos un prompt muy específico para forzar al modelo a basarse solo en el texto.
    try:
        with open(ruta_prompt, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"Plantilla de prompt cargada desde: {ruta_prompt}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de prompt en la ruta: {ruta_prompt}")
        return None
    
    prompt = prompt_template.format(transcripcion_texto=transcripcion_texto)

    text1 = types.Part.from_text(text=prompt)
    
    contents = [
        types.Content(
        role="user",
        parts=[
            text1
        ]
        )
    ]
    
    print("Enviando transcripción y preguntas a Gemini...")
    
    try:
        # Generar la respuesta
        response = client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
            )
        print("Respuesta recibida de Gemini.")
        print(response.text)
        # --- LÓGICA PARA EXTRAER CITAS ---
        informe_texto = response.text
        
        try:
            citations = response.candidates[0].citation_metadata.citation_sources
            citas_formateadas = "\n\n--- CITAS DE LA FUENTE ---\n"
            for citation in citations:
                citas_formateadas += f"- Segmento {citation.segment_index} (URI: {citation.uri}): Inicia en el índice {citation.start_index}, termina en {citation.end_index}.\n"
            
            # Devolvemos el informe Y las citas
            return informe_texto + citas_formateadas
        except (AttributeError, IndexError):
            # Si no hay citas, devolvemos solo el texto
            return informe_texto
    except Exception as e:
        print(f"Ocurrió un error al contactar a la API de Gemini: {e}")
        return None