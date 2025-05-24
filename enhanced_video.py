import os
import cv2
import json
import zipfile
import subprocess
import numpy as np
from io import BytesIO
from typing import Dict, List, Tuple

class EnhancedVideoPipeline:
    def __init__(
        self,
        video_path: str,
        output_folder: str = 'temp_clips',
        interval_seconds: int = 300,
        clip_duration_sec: int = 10,
        analyze_quality: bool = True,
        detect_motion: bool = True,
        smart_extraction: bool = True,
        scale: float = 1.0,
        reduce_fps: bool = False
    ):
        self.video_path = video_path
        self.output_folder = output_folder
        self.interval_seconds = interval_seconds
        self.clip_duration_sec = clip_duration_sec
        self.analyze_quality = analyze_quality
        self.detect_motion = detect_motion
        self.smart_extraction = smart_extraction
        self.scale = scale
        self.reduce_fps = reduce_fps
        
        # Definir la ruta al ejecutable de ffmpeg
        self.ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe')
        self.has_ffmpeg = self._check_ffmpeg_available()  # <--- Solo se verifica una vez

        os.makedirs(self.output_folder, exist_ok=True)
        self._prepare()

    def _prepare(self):
        """Prepara las propiedades del video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"No se puede abrir el video: {self.video_path}")
        
        # Obtener propiedades del video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0  # FPS por defecto si no se puede leer
            
        self.original_fps = original_fps
        self.fps = max(15.0, original_fps / 2) if self.reduce_fps else original_fps
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = total_frames / original_fps if original_fps else 0
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.width = int(original_width * self.scale)
        self.height = int(original_height * self.scale)
        
        # Asegurar dimensiones pares (requerido para H.264)
        if self.width % 2 != 0:
            self.width += 1
        if self.height % 2 != 0:
            self.height += 1
        
        cap.release()

        if self.duration_sec <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Propiedades de video inválidas")
            
        print(f"Video preparado: {self.duration_sec:.1f}s, {self.width}x{self.height}, {self.fps:.1f}fps")

    def _analyze_frame_quality(self, frame) -> float:
        """Analiza la nitidez del frame usando varianza del Laplaciano"""
        if frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _detect_motion(self, prev_frame, curr_frame) -> float:
        """Detecta movimiento entre dos frames"""
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        diff = cv2.absdiff(prev_frame, curr_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh) / 255
        total_pixels = thresh.shape[0] * thresh.shape[1]
        return motion_pixels / total_pixels if total_pixels > 0 else 0.0

    def _should_extract_clip(self, quality_score: float, motion_score: float) -> bool:
        """Determina si se debe extraer un clip basado en los criterios"""
        conditions = []
        
        if self.analyze_quality:
            conditions.append(quality_score > 50.0)
        
        if self.detect_motion:
            conditions.append(motion_score > 0.01)
        
        if not conditions:
            return True
        
        return all(conditions) if self.smart_extraction else any(conditions)

    def _check_ffmpeg_available(self) -> bool:
        """Verifica si ffmpeg está disponible en la carpeta bin"""
        try:
            if not os.path.exists(self.ffmpeg_path):
                print(f"ffmpeg no encontrado en: {self.ffmpeg_path}")
                return False
                
            subprocess.run([self.ffmpeg_path, '-version'], 
                         capture_output=True, check=True)
            print(f"ffmpeg encontrado en: {self.ffmpeg_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error verificando ffmpeg: {str(e)}")
            return False

    def _extract_clip_with_ffmpeg(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando ffmpeg (con audio)"""
        try:
            duration = end_time - start_time
            # Comando ffmpeg para extraer con audio usando la ruta local
            cmd = [
                self.ffmpeg_path,  # Usar la ruta local de ffmpeg
                '-ss', str(start_time),  # SEEK antes de -i para mayor precisión
                '-i', self.video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-y',  # Sobrescribir si existe
                output_path
            ]
            # Agregar parámetros de escala si es necesario
            if self.scale != 1.0:
                cmd.extend(['-vf', f'scale={self.width}:{self.height}'])
            # Agregar parámetros de FPS si es necesario
            if self.reduce_fps:
                cmd.extend(['-r', str(self.fps)])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"  ✓ Clip con audio creado exitosamente usando ffmpeg")
                return True
            else:
                print(f"  ✗ Error con ffmpeg: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ✗ Error ejecutando ffmpeg: {str(e)}")
            return False

    def _extract_clip_opencv_fallback(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando OpenCV (sin audio) como fallback"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        # Configurar posición inicial
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        # CONFIGURACIÓN CRÍTICA PARA STREAMLIT
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 alternativo
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'XVID')   # XVID como último recurso
        ]
        
        writer = None
        for fourcc_option in fourcc_options:
            writer = cv2.VideoWriter(output_path, fourcc_option, self.fps, (self.width, self.height))
            if writer.isOpened():
                print(f"  Usando codec: {fourcc_option}")
                break
            else:
                writer.release()
        
        if writer is None or not writer.isOpened():
            cap.release()
            print(f"  Error: No se pudo crear el writer para {output_path}")
            return False
        
        frames_written = 0
        target_frames = int((end_time - start_time) * self.fps)
        
        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar si es necesario
            if self.scale != 1.0 or frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Asegurar que el frame sea válido
            if frame is not None and frame.size > 0:
                writer.write(frame)
                frames_written += 1
            
            # Si reducimos FPS, saltamos frames
            if self.reduce_fps:
                ret, _ = cap.read()  # Saltar el siguiente frame
                if not ret:
                    break
        
        writer.release()
        cap.release()
        
        # Verificar que el archivo se creó correctamente
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"  ✓ Clip sin audio creado: {frames_written} frames")
            return True
        else:
            print(f"  ✗ Error: Archivo no válido o muy pequeño")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

    def _extract_clip(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip del video priorizando ffmpeg para incluir audio"""
        # Intentar primero con ffmpeg (incluye audio)
        if self._check_ffmpeg_available():
            if self._extract_clip_with_ffmpeg(start_time, end_time, output_path):
                return True
            else:
                print("  Ffmpeg falló, intentando con OpenCV...")
        else:
            print("  Ffmpeg no disponible, usando OpenCV (sin audio)...")
        
        # Fallback a OpenCV (sin audio)
        return self._extract_clip_opencv_fallback(start_time, end_time, output_path)

    def _analyze_segment(self, start_time: float) -> Tuple[float, float]:
        """Analiza un segmento del video para determinar calidad y movimiento"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0.0, 0.0
        
        # Posicionarse en el tiempo de inicio
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        quality_scores = []
        motion_scores = []
        prev_frame = None
        
        # Analizar algunos frames del segmento
        frames_to_analyze = min(10, int(self.clip_duration_sec * self.original_fps / 10))
        
        for i in range(frames_to_analyze):
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.scale != 1.0:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Analizar calidad
            if self.analyze_quality:
                q_score = self._analyze_frame_quality(frame)
                quality_scores.append(q_score)
            
            # Analizar movimiento
            if self.detect_motion and prev_frame is not None:
                m_score = self._detect_motion(prev_frame, frame)
                motion_scores.append(m_score)
            
            prev_frame = frame
            
            # Saltar algunos frames para eficiencia
            for _ in range(int(self.original_fps / 2)):
                ret, _ = cap.read()
                if not ret:
                    break
        
        cap.release()
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        
        return avg_quality, avg_motion

    def extract_clips(self) -> List[Dict]:
        """Extrae clips según los intervalos y criterios configurados"""
        clips_meta = []
        current_time = 0.0
        
        print(f"Iniciando extracción de clips cada {self.interval_seconds}s")
        
        # Verificar disponibilidad de ffmpeg
        has_ffmpeg = self.has_ffmpeg
        if has_ffmpeg:
            print("✓ FFmpeg detectado - Los clips incluirán audio")
        else:
            print("⚠ FFmpeg no disponible - Los clips NO tendrán audio")
        
        while current_time < self.duration_sec:
            end_time = min(current_time + self.clip_duration_sec, self.duration_sec)
            
            # Si el clip sería muy corto, saltar
            if (end_time - current_time) < (self.clip_duration_sec * 0.5):
                break
            
            print(f"Analizando segmento: {current_time:.1f}s - {end_time:.1f}s")
            
            # Analizar el segmento
            quality_score, motion_score = self._analyze_segment(current_time)
            
            # Determinar si extraer el clip
            should_extract = self._should_extract_clip(quality_score, motion_score)
            
            print(f"  Calidad: {quality_score:.2f}, Movimiento: {motion_score:.4f}, Extraer: {should_extract}")
            
            if should_extract:
                # Generar nombre del archivo
                clip_filename = f"clip_{int(current_time):04d}s.mp4"
                output_path = os.path.join(self.output_folder, clip_filename)
                
                # Extraer el clip
                if self._extract_clip(current_time, end_time, output_path):
                    clips_meta.append({
                        "filename": clip_filename,
                        "start_time": round(current_time, 2),
                        "duration": round(end_time - current_time, 2),
                        "quality_score": round(quality_score, 2),
                        "motion_score": round(motion_score, 6),
                        "has_audio": has_ffmpeg  # Indicar si tiene audio
                    })
                    print(f"  ✓ Clip extraído: {clip_filename}")
                else:
                    print(f"  ✗ Error extrayendo clip: {clip_filename}")
            
            # Avanzar al siguiente intervalo
            current_time += self.interval_seconds
        
        return clips_meta

    def run(self) -> Dict:
        """Ejecuta el pipeline completo"""
        print(f"Procesando video: {self.video_path}")
        print(f"Configuración: intervalo={self.interval_seconds}s, duración={self.clip_duration_sec}s")
        
        clips_meta = self.extract_clips()
        
        # Guardar metadatos
        video_metadata = {
            "video": {
                "path": self.video_path,
                "duration": self.duration_sec,
                "original_fps": self.original_fps,
                "output_fps": self.fps,
                "resolution": f"{self.width}x{self.height}",
                "ffmpeg_available": self.has_ffmpeg
            },
            "settings": {
                "interval_seconds": self.interval_seconds,
                "clip_duration_sec": self.clip_duration_sec,
                "analyze_quality": self.analyze_quality,
                "detect_motion": self.detect_motion,
                "smart_extraction": self.smart_extraction,
                "scale": self.scale,
                "reduce_fps": self.reduce_fps
            },
            "clips_meta": clips_meta
        }
        
        metadata_path = os.path.join(self.output_folder, "clips_metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False)
        
        # Calcular estadísticas resumen
        if clips_meta:
            quality_scores = [c["quality_score"] for c in clips_meta]
            motion_scores = [c["motion_score"] for c in clips_meta]
            
            best_quality_clip = max(clips_meta, key=lambda x: x["quality_score"])
            most_motion_clip = max(clips_meta, key=lambda x: x["motion_score"])
            
            summary = {
                "total_clips": len(clips_meta),
                "avg_sharpness": float(np.mean(quality_scores)),
                "avg_motion_score": float(np.mean(motion_scores)),
                "best_quality_clip": best_quality_clip["filename"],
                "most_motion_clip": most_motion_clip["filename"],
                "clips_meta": clips_meta,
                "audio_included": any(c.get("has_audio", False) for c in clips_meta)
            }
        else:
            summary = {
                "total_clips": 0,
                "avg_sharpness": 0.0,
                "avg_motion_score": 0.0,
                "best_quality_clip": None,
                "most_motion_clip": None,
                "clips_meta": [],
                "audio_included": False
            }
        
        # Rutas de los clips generados
        clip_paths = [os.path.join(self.output_folder, c["filename"]) for c in clips_meta]
        
        return {
            "clips": clip_paths,
            "summary": summary,
            "metadata_path": metadata_path
        }


def process_video(
    video_path: str,
    interval_seconds: int = 300,
    output_folder: str = "temp_clips",
    clip_duration_sec: int = 10,
    analyze_quality: bool = True,
    detect_motion: bool = True,
    smart_extraction: bool = True,
    scale: float = 1.0,
    reduce_fps: bool = False
) -> Dict:
    """
    Procesa un video y extrae clips según los criterios especificados.
    """
    pipeline = EnhancedVideoPipeline(
        video_path = video_path,
        output_folder = output_folder,
        interval_seconds = interval_seconds,
        clip_duration_sec = clip_duration_sec,
        analyze_quality = analyze_quality,
        detect_motion = detect_motion,
        smart_extraction = smart_extraction,
        scale = scale,
        reduce_fps = reduce_fps
    )
    return pipeline.run()

def zip_results(output_dir: str) -> bytes:
    """
    Crea un archivo ZIP con todos los clips y metadatos.
    """
    buffer = BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = file
                zip_file.write(file_path, arcname)
                print(f"Agregado al ZIP: {arcname}")
    
    buffer.seek(0)
    return buffer.getvalue()