# Imagen base oficial
FROM python:3.12.7-slim

# Variables de entorno
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# Instalar dependencias del sistema con codecs completos
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg con soporte completo de codecs
    ffmpeg \
    # Librerías multimedia esenciales
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libswresample-dev \
    # Codecs de video específicos
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # Compilación
    gcc \
    g++ \
    build-essential \
    pkg-config \
    # Limpiar cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY . /app

# Actualizar pip y instalar dependencias
RUN pip install --upgrade pip setuptools wheel

# Instalar OpenCV con soporte completo de codecs
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Verificar instalación de FFmpeg
RUN ffmpeg -version && ffmpeg -codecs | grep h264

# Exponer puerto
EXPOSE 8080

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.headless=true", "--server.enableCORS=false"]