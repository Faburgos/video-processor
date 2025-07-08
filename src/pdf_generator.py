# src/pdf_generator.py
from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        # El título se pasará dinámicamente
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def crear_informe_pdf(titulo: str, informe_texto: str, calificaciones: dict, promedio: float, ruta_salida: str):
    """
    Crea un informe en PDF con el análisis de Gemini.
    """
    print("--- [PDF Generator] Creando informe en PDF ---")

    pdf = PDF()
    pdf.title = titulo # Pasamos el título al objeto PDF
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Escribir el informe completo generado por Gemini
    pdf.set_font('Arial', '', 11)
    # Usamos 'latin-1' para manejar caracteres especiales como tildes
    # El método 'encode' convierte el string a un formato que FPDF puede manejar
    pdf.multi_cell(0, 5, informe_texto.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.add_page()
    
    # Añadir la tabla resumen de calificaciones
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Resumen de Calificaciones", 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    for criterio, nota in calificaciones.items():
        pdf.cell(0, 8, f"{criterio}: {nota}/5", 0, 1, 'L')
        
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Promedio General: {promedio:.2f} / 5.00", 0, 1, 'L')

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    pdf.output(ruta_salida)
    print(f"PDF guardado exitosamente en: {ruta_salida}")