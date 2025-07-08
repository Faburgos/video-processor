# src/report_parser.py
import re

def extraer_calificaciones(informe_texto: str) -> tuple[dict, float]:
    """
    Extrae las calificaciones de un informe de texto generado por Gemini
    y calcula el promedio. El formato esperado es:
    **P1. ...**
    **Cumple/No Cumple. (X/5)**

    Args:
        informe_texto: El texto completo generado por Gemini.

    Returns:
        Una tupla conteniendo:
        - Un diccionario con las calificaciones por criterio (ej. {'P1': 0, 'P2': 3}).
        - El promedio general de las calificaciones.
    """
    print("--- [Parser] Extrayendo calificaciones del informe (formato P/Cumple) ---")

    # Expresión regular actualizada para el nuevo formato.
    # - `^\*\*P(\d+)\.` : Busca una línea que empiece con **P seguido de un número (captura el número).
    # - `.*?`: Coincide con cualquier caracter (incluyendo saltos de línea) de forma no codiciosa.
    # - `\((\d+)/5\)`: Busca el patrón literal (X/5) y captura la puntuación X.
    # - `re.DOTALL` permite que `.` coincida con saltos de línea, `re.MULTILINE` ayuda con `^`.
    patron = re.compile(r"^\*\*P(\d+)\..*?\((\d+)/5\)", re.MULTILINE | re.DOTALL)

    calificaciones = {}
    puntuaciones = []

    # Usamos finditer para encontrar todas las coincidencias en el texto
    for match in patron.finditer(informe_texto):
        pregunta_num = int(match.group(1))
        puntuacion = int(match.group(2))

        # Guardamos la calificación usando la "P" para consistencia
        calificaciones[f"P{pregunta_num}"] = puntuacion
        puntuaciones.append(puntuacion)

    if not puntuaciones:
        print("ADVERTENCIA: No se encontraron calificaciones con el formato esperado en el texto.")
        print("Asegúrate de que la respuesta de Gemini incluya la puntuación como '(X/5)'.")
        return {}, 0.0

    promedio = sum(puntuaciones) / len(puntuaciones)
    print(f"Calificaciones extraídas: {calificaciones}")
    print(f"Promedio calculado: {promedio:.2f}")

    return calificaciones, promedio