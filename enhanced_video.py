import os
import tempfile
import cv2
import json
import zipfile
import subprocess
import numpy as np
from io import BytesIO
from typing import Dict, List, Tuple
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class SchoolYOLOVideoPipeline:
    def __init__(
        self,
        video_path: str,
        model_path: str,  # Ruta a tu modelo YOLO entrenado
        output_folder: str = 'temp_clips',
        interval_seconds: int = 300,
        clip_duration_sec: int = 10,
        analyze_quality: bool = True,
        detect_motion: bool = True,
        smart_extraction: bool = True,
        scale: float = 1.0,
        reduce_fps: bool = False,
        confidence_threshold: float = 0.5,
        detection_analysis: bool = True,
        yolo_in_video: bool = True
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_folder = output_folder
        self.interval_seconds = interval_seconds
        self.clip_duration_sec = clip_duration_sec
        self.analyze_quality = analyze_quality
        self.detect_motion = detect_motion
        self.smart_extraction = smart_extraction
        self.scale = scale
        self.reduce_fps = reduce_fps
        self.confidence_threshold = confidence_threshold
        self.detection_analysis = detection_analysis
        self.yolo_in_video = yolo_in_video
        
        # Cargar modelo YOLO
        try:
            self.model = YOLO(model_path)
            print(f"✓ Modelo YOLO cargado desde: {model_path}")
            print(f"  Clases detectables: {self.model.names}")
        except Exception as e:
            print(f"✗ Error cargando modelo YOLO: {str(e)}")
            self.model = None
        
        # Definir la ruta al ejecutable de ffmpeg
        self.ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe')
        self.has_ffmpeg = self._check_ffmpeg_available()

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
            original_fps = 30.0
            
        self.original_fps = original_fps
        self.fps = max(15.0, original_fps / 2) if self.reduce_fps else original_fps
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = total_frames / original_fps if original_fps else 0
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.width = int(original_width * self.scale)
        self.height = int(original_height * self.scale)
        
        # Asegurar dimensiones pares
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

    def _analyze_detections(self, frame) -> Dict:
        """Analiza detecciones YOLO en el frame"""
        if self.model is None:
            return {
                'total_detections': 0,
                'students_sitting': 0,
                'students_standing': 0,
                'teachers_sitting': 0,
                'teachers_standing': 0,
                'detection_score': 0.0,
                'detections': []
            }
        
        # Realizar detección
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Procesar resultados
        detections = {
            'student_sitting': 0,
            'student_standing': 0,
            'teacher_sitting': 0,
            'teacher_standing': 0
        }
        
        detection_details = []
        total_confidence = 0.0
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    # Obtener clase y confianza
                    cls_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if cls_id < len(self.model.names):
                        class_name = self.model.names[cls_id]
                        
                        # Contar detecciones por tipo
                        if class_name in detections:
                            detections[class_name] += 1
                            total_confidence += confidence
                            
                            # Guardar detalles de la detección
                            xyxy = box.xyxy.cpu().numpy()[0]
                            detection_details.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': xyxy.tolist()
                            })
        
        total_detections = sum(detections.values())
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
        
        return {
            'total_detections': total_detections,
            'students_sitting': detections.get('student_sitting', 0),
            'students_standing': detections.get('student_standing', 0),
            'teachers_sitting': detections.get('teacher_sitting', 0),
            'teachers_standing': detections.get('teacher_standing', 0),
            'detection_score': avg_confidence,
            'detections': detection_details
        }

    def _should_extract_clip(self, quality_score: float, motion_score: float, detection_data: Dict) -> bool:
        """Determina si se debe extraer un clip basado en los criterios mejorados"""
        conditions = []
        
        if self.analyze_quality:
            conditions.append(quality_score > 50.0)
        
        if self.detect_motion:
            conditions.append(motion_score > 0.01)
        
        if self.detection_analysis and self.model is not None:
            # Extraer clips que tengan al menos una detección relevante
            has_meaningful_detections = detection_data['total_detections'] > 0
            has_good_confidence = detection_data['detection_score'] > self.confidence_threshold
            conditions.append(has_meaningful_detections and has_good_confidence)
        
        if not conditions:
            return True
        
        return all(conditions) if self.smart_extraction else any(conditions)

    def _check_ffmpeg_available(self) -> bool:
        """Verifica si ffmpeg está disponible"""
        try:
            if os.path.exists(self.ffmpeg_path):
                subprocess.run([self.ffmpeg_path, '-version'], 
                             capture_output=True, check=True)
                return True
            else:
                # Probar ffmpeg del sistema
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, check=True)
                self.ffmpeg_path = 'ffmpeg'
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _extract_clip_with_ffmpeg(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando ffmpeg (con audio)"""
        try:
            duration = end_time - start_time
            cmd = [
                self.ffmpeg_path,
                '-ss', str(start_time),
                '-i', self.video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-y',
                output_path
            ]
            
            if self.scale != 1.0:
                cmd.extend(['-vf', f'scale={self.width}:{self.height}'])
            
            if self.reduce_fps:
                cmd.extend(['-r', str(self.fps)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True
            else:
                return False
        except Exception as e:
            return False

    def _extract_clip_opencv_fallback(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando OpenCV (sin audio) como fallback"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'H264'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID')
        ]
        
        writer = None
        for fourcc_option in fourcc_options:
            writer = cv2.VideoWriter(output_path, fourcc_option, self.fps, (self.width, self.height))
            if writer.isOpened():
                break
            else:
                writer.release()
        
        if writer is None or not writer.isOpened():
            cap.release()
            return False
        
        frames_written = 0
        target_frames = int((end_time - start_time) * self.fps)
        
        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.scale != 1.0 or frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            if frame is not None and frame.size > 0:
                writer.write(frame)
                frames_written += 1
            
            if self.reduce_fps:
                ret, _ = cap.read()
                if not ret:
                    break
        
        writer.release()
        cap.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def _extract_clip_opencv_with_yolo(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando OpenCV aplicando detecciones YOLO en tiempo real"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'H264'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID')
        ]

         # 1. Crear video temporal sin audio
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_video_path = tmpfile.name
        
        writer = None
        for fourcc_option in fourcc_options:
            writer = cv2.VideoWriter(temp_video_path, fourcc_option, self.fps, (self.width, self.height))
            if writer.isOpened():
                break
            else:
                writer.release()
        
        if writer is None or not writer.isOpened():
            cap.release()
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return False
        
        frames_written = 0
        target_frames = int((end_time - start_time) * self.fps)
        
        # Colores para diferentes clases
        colors = {
            'student_sitting': (255, 0, 0),      # Azul
            'student_standing': (255, 255, 0),   # Cian
            'teacher_sitting': (0, 165, 255),    # Naranja
            'teacher_standing': (0, 0, 255)      # Rojo
        }
        
        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar frame si es necesario
            if self.scale != 1.0 or frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Aplicar detecciones YOLO si el modelo está disponible
            if self.model is not None:
                try:
                    # Realizar detección en el frame actual
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                    
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None:
                            for box in boxes:
                                # Obtener información de la detección
                                cls_id = int(box.cls.cpu().numpy()[0])
                                confidence = float(box.conf.cpu().numpy()[0])
                                xyxy = box.xyxy.cpu().numpy()[0]
                                
                                if cls_id < len(self.model.names):
                                    class_name = self.model.names[cls_id]
                                    
                                    # Obtener coordenadas
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    
                                    # Elegir color según la clase
                                    color = colors.get(class_name, (0, 255, 0))  # Verde por defecto
                                    
                                    # Dibujar bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Preparar etiqueta
                                    label = f"{class_name}: {confidence:.2f}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    # Dibujar fondo para la etiqueta
                                    cv2.rectangle(frame, 
                                                (x1, y1 - label_size[1] - 10), 
                                                (x1 + label_size[0], y1), 
                                                color, -1)
                                    
                                    # Dibujar texto de la etiqueta
                                    cv2.putText(frame, label, 
                                            (x1, y1 - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                            (255, 255, 255), 2)
                                    
                except Exception as e:
                    print(f"Error aplicando YOLO al frame {frames_written}: {str(e)}")
                    # Continúa con el frame original si hay error
            
            # Escribir frame (con o sin detecciones)
            if frame is not None and frame.size > 0:
                writer.write(frame)
                frames_written += 1
            
            # Saltar frame si se reduce FPS
            if self.reduce_fps:
                ret, _ = cap.read()
                if not ret:
                    break
        
        writer.release()
        cap.release()
        
        # 2. Usar FFmpeg para combinar el video temporal con el audio original
        if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1000:
            try:
                duration = end_time - start_time
                ffmpeg_cmd = [
                    self.ffmpeg_path,
                    '-y',
                    '-ss', str(start_time),
                    '-i', self.video_path,
                    '-i', temp_video_path,
                    '-t', str(duration),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '1:v:0',
                    '-map', '0:a:0?',
                    '-shortest',
                    output_path
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                os.remove(temp_video_path)
                if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    return True
                else:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return False
            except Exception as e:
                print(f"Error combinando video y audio con FFmpeg: {e}")
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                return False
        else:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return False

    def _extract_clip(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip del video priorizando ffmpeg"""
        # Si queremos detecciones en el video Y tenemos modelo YOLO
        if self.model is not None and self.detection_analysis:
            # Usar OpenCV con YOLO aplicado frame por frame
            print(f"  Extrayendo con detecciones YOLO en tiempo real...")
            return self._extract_clip_opencv_with_yolo(start_time, end_time, output_path)
        
        # Comportamiento original: priorizar ffmpeg para clips sin detecciones
        if self.has_ffmpeg:
            if self._extract_clip_with_ffmpeg(start_time, end_time, output_path):
                return True
        
        return self._extract_clip_opencv_fallback(start_time, end_time, output_path)

    def _create_detection_visualization(self, frame, detections, output_path: str):
        """Crea una imagen con las detecciones visualizadas"""
        try:
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Colores para diferentes clases
            colors = {
                'student_sitting': 'blue',
                'student_standing': 'cyan',
                'teacher_sitting': 'red',
                'teacher_standing': 'orange'
            }
            
            # Dibujar bounding boxes
            for det in detections:
                bbox = det['bbox']
                class_name = det['class']
                confidence = det['confidence']
                
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                color = colors.get(class_name, 'green')
                
                # Dibujar rectángulo
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Añadir etiqueta
                label = f"{class_name}: {confidence:.2f}"
                ax.text(x1, y1-10, label, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
            
            ax.set_title(f"Detecciones YOLO - Total: {len(detections)}")
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error creando visualización: {str(e)}")
            return False

    def _analyze_segment(self, start_time: float) -> Tuple[float, float, Dict]:
        """Analiza un segmento del video incluyendo detecciones YOLO"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0.0, 0.0, {}
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        quality_scores = []
        motion_scores = []
        detection_results = []
        prev_frame = None
        
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
            
            # Analizar detecciones YOLO
            if self.detection_analysis and self.model is not None:
                det_result = self._analyze_detections(frame)
                detection_results.append(det_result)
            
            prev_frame = frame
            
            # Saltar algunos frames para eficiencia
            for _ in range(int(self.original_fps / 2)):
                ret, _ = cap.read()
                if not ret:
                    break
        
        cap.release()
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        
        # Agregar estadísticas de detección
        if detection_results:
            avg_detections = {
                'total_detections': np.mean([d['total_detections'] for d in detection_results]),
                'students_sitting': np.mean([d['students_sitting'] for d in detection_results]),
                'students_standing': np.mean([d['students_standing'] for d in detection_results]),
                'teachers_sitting': np.mean([d['teachers_sitting'] for d in detection_results]),
                'teachers_standing': np.mean([d['teachers_standing'] for d in detection_results]),
                'detection_score': np.mean([d['detection_score'] for d in detection_results]),
                'best_frame_detections': max(detection_results, key=lambda x: x['total_detections'])['detections']
            }
        else:
            avg_detections = {
                'total_detections': 0,
                'students_sitting': 0,
                'students_standing': 0,
                'teachers_sitting': 0,
                'teachers_standing': 0,
                'detection_score': 0.0,
                'best_frame_detections': []
            }
        
        return avg_quality, avg_motion, avg_detections

    def extract_clips(self) -> List[Dict]:
        """Extrae clips según los intervalos y criterios configurados"""
        clips_meta = []
        current_time = 0.0
        
        print(f"Iniciando extracción de clips cada {self.interval_seconds}s")
        
        if self.has_ffmpeg:
            print("✓ FFmpeg detectado - Los clips incluirán audio")
        else:
            print("⚠ FFmpeg no disponible - Los clips NO tendrán audio")
        
        if self.model:
            print(f"✓ Modelo YOLO activo - Analizando detecciones escolares")
        
        while current_time < self.duration_sec:
            end_time = min(current_time + self.clip_duration_sec, self.duration_sec)
            
            if (end_time - current_time) < (self.clip_duration_sec * 0.5):
                break
            
            print(f"Analizando segmento: {current_time:.1f}s - {end_time:.1f}s")
            
            # Analizar el segmento (incluyendo detecciones YOLO)
            quality_score, motion_score, detection_data = self._analyze_segment(current_time)
            
            # Determinar si extraer el clip
            should_extract = self._should_extract_clip(quality_score, motion_score, detection_data)
            
            print(f"  Calidad: {quality_score:.2f}")
            print(f"  Movimiento: {motion_score:.4f}")
            print(f"  Detecciones: {detection_data.get('total_detections', 0):.1f}")
            print(f"  Estudiantes sentados: {detection_data.get('students_sitting', 0):.1f}")
            print(f"  Estudiantes parados: {detection_data.get('students_standing', 0):.1f}")
            print(f"  Maestros sentados: {detection_data.get('teachers_sitting', 0):.1f}")
            print(f"  Maestros parados: {detection_data.get('teachers_standing', 0):.1f}")
            print(f"  Extraer: {should_extract}")
            
            if should_extract:
                clip_filename = f"clip_{int(current_time):04d}s.mp4"
                output_path = os.path.join(self.output_folder, clip_filename)
                
                # Extraer el clip
                if self._extract_clip(current_time, end_time, output_path):
                    # Crear visualización de detecciones si es relevante
                    visualization_path = None
                    if detection_data.get('best_frame_detections') and len(detection_data['best_frame_detections']) > 0:
                        # Obtener un frame representativo para la visualización
                        cap = cv2.VideoCapture(self.video_path)
                        cap.set(cv2.CAP_PROP_POS_MSEC, (current_time + self.clip_duration_sec/2) * 1000)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            viz_filename = f"detections_{int(current_time):04d}s.jpg"
                            visualization_path = os.path.join(self.output_folder, viz_filename)
                            self._create_detection_visualization(
                                frame, 
                                detection_data['best_frame_detections'], 
                                visualization_path
                            )
                    
                    clips_meta.append({
                        "filename": clip_filename,
                        "start_time": round(current_time, 2),
                        "duration": round(end_time - current_time, 2),
                        "quality_score": round(quality_score, 2),
                        "motion_score": round(motion_score, 6),
                        "has_audio": self.has_ffmpeg,
                        "detections": {
                            "total": round(detection_data.get('total_detections', 0), 1),
                            "students_sitting": round(detection_data.get('students_sitting', 0), 1),
                            "students_standing": round(detection_data.get('students_standing', 0), 1),
                            "teachers_sitting": round(detection_data.get('teachers_sitting', 0), 1),
                            "teachers_standing": round(detection_data.get('teachers_standing', 0), 1),
                            "avg_confidence": round(detection_data.get('detection_score', 0), 3)
                        },
                        "visualization": viz_filename if visualization_path and os.path.exists(visualization_path) else None
                    })
                    print(f"  ✓ Clip extraído: {clip_filename}")
                else:
                    print(f"  ✗ Error extrayendo clip: {clip_filename}")
            
            current_time += self.interval_seconds
        
        return clips_meta

    def run(self) -> Dict:
        """Ejecuta el pipeline completo"""
        print(f"Procesando video: {self.video_path}")
        print(f"Modelo YOLO: {self.model_path}")
        print(f"Configuración: intervalo={self.interval_seconds}s, duración={self.clip_duration_sec}s")
        
        clips_meta = self.extract_clips()
        
        # Guardar metadatos extendidos
        video_metadata = {
            "video": {
                "path": self.video_path,
                "duration": self.duration_sec,
                "original_fps": self.original_fps,
                "output_fps": self.fps,
                "resolution": f"{self.width}x{self.height}",
                "ffmpeg_available": self.has_ffmpeg
            },
            "model": {
                "path": self.model_path,
                "available": self.model is not None,
                "classes": list(self.model.names.values()) if self.model else [],
                "confidence_threshold": self.confidence_threshold
            },
            "settings": {
                "interval_seconds": self.interval_seconds,
                "clip_duration_sec": self.clip_duration_sec,
                "analyze_quality": self.analyze_quality,
                "detect_motion": self.detect_motion,
                "detection_analysis": self.detection_analysis,
                "smart_extraction": self.smart_extraction,
                "scale": self.scale,
                "reduce_fps": self.reduce_fps
            },
            "clips_meta": clips_meta
        }
        
        metadata_path = os.path.join(self.output_folder, "clips_metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False)
        
        # Calcular estadísticas resumen mejoradas
        if clips_meta:
            quality_scores = [c["quality_score"] for c in clips_meta]
            motion_scores = [c["motion_score"] for c in clips_meta]
            
            # Estadísticas de detección
            total_students_sitting = sum(c["detections"]["students_sitting"] for c in clips_meta)
            total_students_standing = sum(c["detections"]["students_standing"] for c in clips_meta)
            total_teachers_sitting = sum(c["detections"]["teachers_sitting"] for c in clips_meta)
            total_teachers_standing = sum(c["detections"]["teachers_standing"] for c in clips_meta)
            
            best_quality_clip = max(clips_meta, key=lambda x: x["quality_score"])
            most_detections_clip = max(clips_meta, key=lambda x: x["detections"]["total"])
            
            summary = {
                "total_clips": len(clips_meta),
                "avg_sharpness": float(np.mean(quality_scores)),
                "avg_motion_score": float(np.mean(motion_scores)),
                "best_quality_clip": best_quality_clip["filename"],
                "most_detections_clip": most_detections_clip["filename"],
                "detection_summary": {
                    "total_students_sitting": total_students_sitting,
                    "total_students_standing": total_students_standing,
                    "total_teachers_sitting": total_teachers_sitting,
                    "total_teachers_standing": total_teachers_standing,
                    "clips_with_students": len([c for c in clips_meta if c["detections"]["students_sitting"] + c["detections"]["students_standing"] > 0]),
                    "clips_with_teachers": len([c for c in clips_meta if c["detections"]["teachers_sitting"] + c["detections"]["teachers_standing"] > 0])
                },
                "clips_meta": clips_meta,
                "audio_included": any(c.get("has_audio", False) for c in clips_meta)
            }
        else:
            summary = {
                "total_clips": 0,
                "avg_sharpness": 0.0,
                "avg_motion_score": 0.0,
                "best_quality_clip": None,
                "most_detections_clip": None,
                "detection_summary": {
                    "total_students_sitting": 0,
                    "total_students_standing": 0,
                    "total_teachers_sitting": 0,
                    "total_teachers_standing": 0,
                    "clips_with_students": 0,
                    "clips_with_teachers": 0
                },
                "clips_meta": [],
                "audio_included": False
            }
        
        clip_paths = [os.path.join(self.output_folder, c["filename"]) for c in clips_meta]
        
        return {
            "clips": clip_paths,
            "summary": summary,
            "metadata_path": metadata_path
        }


def process_school_video(
    video_path: str,
    model_path: str,
    interval_seconds: int = 300,
    output_folder: str = "temp_clips",
    clip_duration_sec: int = 10,
    analyze_quality: bool = True,
    detect_motion: bool = True,
    smart_extraction: bool = True,
    scale: float = 1.0,
    reduce_fps: bool = False,
    confidence_threshold: float = 0.5,
    detection_analysis: bool = True,
    yolo_in_video: bool = True
) -> Dict:
    """
    Procesa un video escolar y extrae clips con análisis YOLO.
    
    Args:
        video_path: Ruta al video de entrada
        model_path: Ruta al modelo YOLO entrenado
        interval_seconds: Intervalo entre clips en segundos
        output_folder: Carpeta de salida
        clip_duration_sec: Duración de cada clip
        analyze_quality: Analizar calidad de imagen
        detect_motion: Detectar movimiento
        smart_extraction: Extracción inteligente (todos los criterios)
        scale: Factor de escala de resolución
        reduce_fps: Reducir FPS
        confidence_threshold: Umbral de confianza para detecciones
        detection_analysis: Incluir análisis de detecciones YOLO
    
    Returns:
        Diccionario con clips, resumen y metadatos
    """
    pipeline = SchoolYOLOVideoPipeline(
        video_path=video_path,
        model_path=model_path,
        output_folder=output_folder,
        interval_seconds=interval_seconds,
        clip_duration_sec=clip_duration_sec,
        analyze_quality=analyze_quality,
        detect_motion=detect_motion,
        smart_extraction=smart_extraction,
        scale=scale,
        reduce_fps=reduce_fps,
        confidence_threshold=confidence_threshold,
        detection_analysis=detection_analysis,
        yolo_in_video=yolo_in_video
    )
    return pipeline.run()


def zip_results(output_dir: str) -> bytes:
    """
    Crea un archivo ZIP con todos los clips, visualizaciones y metadatos.
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