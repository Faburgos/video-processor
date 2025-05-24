# 🎥 Video Processor MVP

## Descripción del Proyecto

**Video Processor MVP** es una aplicación web desarrollada en Python que permite extraer clips inteligentes de videos largos de manera automatizada. La aplicación utiliza técnicas de análisis de imagen y detección de movimiento para identificar los segmentos más relevantes de un video y extraerlos como clips independientes.

### ¿Qué hace este proyecto?

El sistema procesa videos largos (conferencias, tutoriales, grabaciones de reuniones, etc.) y extrae automáticamente clips cortos basándose en criterios de calidad y movimiento. Esto es especialmente útil para:

- Crear highlights automáticos de videos largos
- Extraer momentos clave de conferencias o presentaciones
- Generar contenido para redes sociales a partir de videos extensos
- Crear resúmenes visuales de material educativo

## 🚀 Características Principales

### ✨ Extracción Inteligente de Clips
- **Análisis automático**: El sistema evalúa cada segmento del video antes de extraerlo
- **Criterios múltiples**: Combina análisis de calidad de imagen y detección de movimiento
- **Filtrado inteligente**: Solo extrae clips que cumplan con los criterios establecidos

### 🔍 Análisis de Calidad de Imagen
- **Algoritmo de nitidez**: Utiliza el operador Laplaciano para medir la nitidez de cada frame
- **Puntuación de calidad**: Asigna una puntuación numérica a cada segmento
- **Filtrado automático**: Descarta clips borrosos o de baja calidad

### 🏃 Detección de Movimiento
- **Análisis frame-a-frame**: Compara frames consecutivos para detectar cambios
- **Puntuación de movimiento**: Cuantifica la cantidad de movimiento en cada segmento
- **Filtrado de contenido estático**: Evita extraer clips de escenas estáticas

### 📱 Interfaz Web Intuitiva
- **Streamlit**: Interfaz web moderna y fácil de usar
- **Vista previa**: Muestra los primeros clips generados
- **Descarga masiva**: Permite descargar todos los clips en un archivo ZIP
- **Metadatos detallados**: Información técnica de cada clip extraído

## ⚙️ Parámetros Ajustables Detallados

### 🕐 Intervalo entre Clips
**Parámetro**: `interval_seconds` (1-30 minutos)
**Ubicación en código**: `enhanced_video.py` línea 15

**¿Qué es?**: Define cada cuánto tiempo se evalúa el video para extraer un nuevo clip.

**¿Cómo funciona?**: 
```python
current_time += self.interval_seconds  # Avanza al siguiente punto de evaluación
```

**¿Para qué sirve?**: 
- **Intervalos cortos (1-5 min)**: Más clips, mayor cobertura del video, archivos más grandes
- **Intervalos largos (15-30 min)**: Menos clips, menor cobertura, archivos más pequeños
- **Uso recomendado**: 5 minutos para videos educativos, 10-15 minutos para conferencias largas

### ⏱️ Duración del Clip
**Parámetro**: `clip_duration_sec` (5-120 segundos)
**Ubicación en código**: `enhanced_video.py` línea 16

**¿Qué es?**: La duración en segundos de cada clip extraído.

**¿Cómo funciona?**:
```python
end_time = min(current_time + self.clip_duration_sec, self.duration_sec)
target_frames = int((end_time - start_time) * self.fps)
```

**¿Para qué sirve?**:
- **Clips cortos (5-15s)**: Ideales para redes sociales, highlights rápidos
- **Clips medianos (30-60s)**: Buenos para resúmenes, momentos clave
- **Clips largos (90-120s)**: Para contexto completo, explicaciones detalladas

### 🎬 FPS (Frames Por Segundo)
**Parámetro**: `reduce_fps` (True/False)
**Ubicación en código**: `enhanced_video.py` líneas 18, 44-45

**¿Qué son los FPS?**: Los FPS (Frames Per Second) indican cuántas imágenes se muestran por segundo en un video. Un video típico tiene 24-30 FPS.

**¿Cómo funciona en el código?**:
```python
self.original_fps = original_fps
self.fps = original_fps / 2 if self.reduce_fps else original_fps

# Durante la extracción:
if self.reduce_fps:
    cap.read()  # Saltar el siguiente frame
```

**¿Para qué sirve?**:
- **FPS original**: Mantiene la fluidez original del video
- **FPS reducido**: Reduce el tamaño del archivo a la mitad, útil para:
  - Ahorrar espacio de almacenamiento
  - Acelerar la transferencia de archivos
  - Contenido donde la fluidez no es crítica (presentaciones, tutoriales)

**Impacto en el archivo**:
- Reducir FPS puede disminuir el tamaño del archivo en un 40-50%
- La calidad visual se mantiene, pero el movimiento es menos fluido

### 📏 Escala de Resolución
**Parámetro**: `scale` (0.1 - 1.0)
**Ubicación en código**: `enhanced_video.py` líneas 19, 52-53

**¿Qué es?**: Factor multiplicador que ajusta la resolución del video de salida.

**¿Cómo funciona?**:
```python
self.width = int(original_width * self.scale)
self.height = int(original_height * self.scale)

# Durante el procesamiento:
if self.scale != 1.0:
    frame = cv2.resize(frame, (self.width, self.height))
```

**¿Para qué sirve?**:
- **scale = 1.0**: Resolución original (1920x1080 → 1920x1080)
- **scale = 0.5**: Mitad de resolución (1920x1080 → 960x540)
- **scale = 0.25**: Cuarto de resolución (1920x1080 → 480x270)

**Casos de uso**:
- **1.0**: Máxima calidad, archivos grandes
- **0.7-0.8**: Buena calidad, archivos medianos
- **0.5**: Calidad aceptable, archivos pequeños
- **0.3-0.4**: Baja calidad, archivos muy pequeños (para previsualizaciones)

### 🎯 Análisis de Calidad
**Parámetro**: `analyze_quality` (True/False)
**Ubicación en código**: `enhanced_video.py` líneas 17, 60-65

**¿Qué es?**: Sistema que evalúa la nitidez y claridad de cada frame usando el operador Laplaciano.

**¿Cómo funciona?**:
```python
def _analyze_frame_quality(self, frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
```

**Algoritmo explicado**:
1. **Conversión a escala de grises**: Simplifica el análisis
2. **Operador Laplaciano**: Detecta bordes y detalles finos
3. **Varianza**: Mide la dispersión de valores (mayor varianza = más nitidez)

**¿Para qué sirve?**:
- **Activado**: Solo extrae clips con buena nitidez (umbral > 50.0)
- **Desactivado**: Extrae todos los clips sin filtrar por calidad
- **Útil para**: Eliminar clips borrosos, desenfocados o de baja calidad

### 🏃 Detección de Movimiento
**Parámetro**: `detect_motion` (True/False)
**Ubicación en código**: `enhanced_video.py` líneas 17, 67-76

**¿Qué es?**: Sistema que detecta cambios entre frames consecutivos para identificar movimiento.

**¿Cómo funciona?**:
```python
def _detect_motion(self, prev_frame, curr_frame) -> float:
    diff = cv2.absdiff(prev_frame, curr_frame)  # Diferencia absoluta
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh) / 255
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return motion_pixels / total_pixels
```

**Algoritmo explicado**:
1. **Diferencia absoluta**: Calcula cambios entre frames
2. **Umbralización**: Identifica píxeles que cambiaron significativamente
3. **Proporción**: Calcula qué porcentaje del frame cambió

**¿Para qué sirve?**:
- **Activado**: Solo extrae clips con movimiento (umbral > 0.01)
- **Desactivado**: Extrae clips sin considerar el movimiento
- **Útil para**: Evitar clips estáticos, encontrar momentos dinámicos

### 🧠 Extracción Inteligente
**Parámetro**: `smart_extraction` (True/False)
**Ubicación en código**: `enhanced_video.py` líneas 18, 78-92

**¿Qué es?**: Define la lógica para combinar los criterios de calidad y movimiento.

**¿Cómo funciona?**:
```python
def _should_extract_clip(self, quality_score: float, motion_score: float) -> bool:
    conditions = []
    
    if self.analyze_quality:
        conditions.append(quality_score > 50.0)
    
    if self.detect_motion:
        conditions.append(motion_score > 0.01)
    
    # Lógica de decisión:
    return all(conditions) if self.smart_extraction else any(conditions)
```

**Modos de operación**:
- **Smart = True (AND)**: El clip debe cumplir TODOS los criterios
  - Calidad > 50.0 **Y** Movimiento > 0.01
  - Resultado: Menos clips, pero de mayor calidad
  
- **Smart = False (OR)**: El clip debe cumplir AL MENOS UN criterio
  - Calidad > 50.0 **O** Movimiento > 0.01
  - Resultado: Más clips, criterios más flexibles

**¿Cuándo usar cada modo?**:
- **Smart = True**: Para contenido de alta calidad, highlights selectivos
- **Smart = False**: Para máxima cobertura, cuando necesitas más material

## 🛠️ Instalación y Uso

### Requisitos del Sistema
- Python 3.8 o superior
- OpenCV compatible con tu sistema
- Al menos 4GB de RAM (recomendado 8GB para videos grandes)
- Espacio en disco suficiente para los clips generados

### Instalación

1. **Clonar el repositorio**:
```bash
git clone <url-del-repositorio>
cd video-pipeline
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Ejecutar la aplicación**:
```bash
streamlit run app.py
```

4. **Abrir en el navegador**:
La aplicación se abrirá automáticamente en `http://localhost:8501`

### Uso Básico

1. **Subir video**: Arrastra o selecciona tu archivo de video
2. **Configurar parámetros**: Ajusta los valores en la barra lateral
3. **Procesar**: Haz clic en "🚀 Procesar Video"
4. **Descargar**: Descarga los clips individuales o el ZIP completo

---

### ⚠️ Configuración de FFmpeg (Windows)

Para que los clips extraídos incluyan **audio**, necesitas el ejecutable de FFmpeg. Este proyecto incluye el archivo comprimido `ffmpeg.zip` para tu comodidad.

#### ¿Cómo configurar FFmpeg?

1. Ubica el archivo `ffmpeg.zip` en la raíz del proyecto.
2. Descomprime su contenido en la carpeta `bin` en la raíz del proyecto.
   - Después de descomprimir, deberías tener: `bin/ffmpeg.exe`
3. ¡Listo! La aplicación usará automáticamente el ejecutable local de FFmpeg.

Si prefieres usar tu propia versión de FFmpeg, también puedes colocar tu `ffmpeg.exe` en la carpeta `bin`, o instalar FFmpeg globalmente y agregarlo al PATH del sistema.

#### ¿Qué pasa si no descomprimes FFmpeg?
- Los clips se generarán **sin audio** (solo video).
- El sistema te avisará en la consola si FFmpeg no está disponible.

---

## 📁 Estructura del Proyecto

```
video-pipeline/
├── app.py                 # Interfaz web principal (Streamlit)
├── enhanced_video.py      # Motor de procesamiento de video
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
├── temp_uploads/         # Carpeta temporal para videos subidos
└── temp_clips/           # Carpeta temporal para clips generados
```

### Archivos Principales

#### `app.py`
- **Función**: Interfaz de usuario web con Streamlit
- **Características**:
  - Subida de archivos de video
  - Configuración de parámetros mediante sliders
  - Vista previa de clips generados
  - Descarga de resultados
  - Dashboard con métricas del procesamiento

#### `enhanced_video.py`
- **Función**: Motor de procesamiento de video
- **Clase principal**: `EnhancedVideoPipeline`
- **Características**:
  - Análisis de calidad de imagen
  - Detección de movimiento
  - Extracción de clips
  - Generación de metadatos
  - Compresión y redimensionamiento

## 🔄 Flujo de Procesamiento

1. **Carga del video**: Se valida y cargan las propiedades del video
2. **Configuración**: Se establecen los parámetros de procesamiento
3. **Análisis por segmentos**: 
   - Se evalúa cada segmento según el intervalo configurado
   - Se calcula la puntuación de calidad y movimiento
4. **Filtrado**: Se decide si extraer cada clip según los criterios
5. **Extracción**: Se generan los clips que pasan el filtro
6. **Metadatos**: Se guardan las estadísticas y información de cada clip
7. **Empaquetado**: Se preparan los archivos para descarga

## 💡 Ejemplos de Uso

### Caso 1: Conferencia Académica (90 minutos)
```
Configuración recomendada:
- Intervalo: 10 minutos
- Duración: 45 segundos
- Calidad: Activada
- Movimiento: Activada
- Smart: True
- Escala: 0.8
- FPS reducido: True

Resultado esperado: 6-9 clips de momentos clave
```

### Caso 2: Tutorial de Programación (30 minutos)
```
Configuración recomendada:
- Intervalo: 3 minutos
- Duración: 20 segundos
- Calidad: Activada
- Movimiento: Desactivada (código estático)
- Smart: False
- Escala: 1.0
- FPS reducido: False

Resultado esperado: 8-10 clips de explicaciones importantes
```

### Caso 3: Video de Ejercicios (60 minutos)
```
Configuración recomendada:
- Intervalo: 5 minutos
- Duración: 30 segundos
- Calidad: Activada
- Movimiento: Activada
- Smart: True
- Escala: 0.7
- FPS reducido: False

Resultado esperado: 10-12 clips de ejercicios dinámicos
```

## 📊 Interpretación de Métricas

### Puntuación de Calidad (Nitidez)
- **> 100**: Excelente calidad, imagen muy nítida
- **50-100**: Buena calidad, imagen clara
- **20-50**: Calidad aceptable, ligero desenfoque
- **< 20**: Baja calidad, imagen borrosa

### Puntuación de Movimiento
- **> 0.1**: Movimiento muy alto (cambio de escena, acción)
- **0.01-0.1**: Movimiento moderado (gestos, movimientos sutiles)
- **0.001-0.01**: Movimiento bajo (pequeños cambios)
- **< 0.001**: Prácticamente estático

## 🔧 Solución de Problemas

### Error: "No se generaron clips"
- **Causa**: Criterios muy estrictos
- **Solución**: Desactivar "Extracción inteligente" o reducir umbrales

### Error: "Archivo muy grande"
- **Causa**: Video de alta resolución
- **Solución**: Reducir escala a 0.5-0.7 y activar FPS reducido

### Error: "Procesamiento muy lento"
- **Causa**: Video muy largo o alta resolución
- **Solución**: Aumentar intervalo entre clips y reducir escala

## 🚀 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **OpenCV**: Procesamiento de video e imagen
- **Streamlit**: Interfaz web interactiva
- **NumPy**: Cálculos numéricos y análisis de arrays
- **Pandas**: Manipulación de datos y metadatos

## 📝 Notas Técnicas

- Los clips se guardan en formato MP4 con codec H.264
- Los metadatos se exportan en formato JSON
- La aplicación maneja archivos de hasta 2GB
- Los archivos temporales se limpian automáticamente
- Compatible con formatos: MP4, AVI, MOV, MKV, FLV, WMV

---

**Desarrollado para facilitar el procesamiento inteligente de videos**