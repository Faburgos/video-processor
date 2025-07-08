import os
import cv2
import json
import zipfile
import tempfile
import platform
import subprocess
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from typing import Dict, List
from emotion_analyzer import ImprovedEmotionAnalyzer

class SchoolYOLOVideoPipeline:
    """
    A pipeline for processing school videos using YOLOv8 and Emotion Analyzer.
    """
    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_folder: str = 'temp_clips',
        interval_seconds: int = 300,
        clip_duration_sec: int = 10,
        confidence_threshold: float = 0.5,
        detection_analysis: bool = True,
        yolo_in_video: bool = True,
        emotion_model_path: str = None
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_folder = output_folder
        self.interval_seconds = interval_seconds
        self.clip_duration_sec = clip_duration_sec
        self.confidence_threshold = confidence_threshold
        self.detection_analysis = detection_analysis
        self.yolo_in_video = yolo_in_video
        self.emotion_model_path = emotion_model_path
        
        # Cargar modelo YOLO
        try:
            self.model = YOLO(model_path)
            print(f"✓ Modelo YOLO cargado desde: {model_path}")
            print(f"  Clases detectables: {self.model.names}")
        except Exception as e:
            print(f"✗ Error cargando modelo YOLO: {str(e)}")
            self.model = None
        
        # Cargar modelo de emociones
        self.emotion_analyzer = None
        if emotion_model_path and os.path.exists(emotion_model_path):
            try:
                self.emotion_analyzer = ImprovedEmotionAnalyzer(
                    emotion_model_path,
                    confidence_threshold = 0.5  # Ajustable
                )
                print("✓ Modelo de emociones cargado correctamente")
            except Exception as e:
                print(f"✗ Error cargando modelo de emociones: {str(e)}")
                self.emotion_analyzer = None
        
        # Configurar FFmpeg
        if platform.system().lower().startswith("win"):
            self.ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe')
        else:
            self.ffmpeg_path = "ffmpeg"
        self.has_ffmpeg = self._check_ffmpeg_available()

        os.makedirs(self.output_folder, exist_ok=True)
        self._prepare()

    def _prepare(self):
        """Prepara las propiedades del video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"No se puede abrir el video: {self.video_path}")
        
        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0:
            self.original_fps = 30.0
            
        self.fps = self.original_fps
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = total_frames / self.original_fps if self.original_fps else 0
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Asegurar dimensiones pares para codecs
        if self.width % 2 != 0:
            self.width += 1
        if self.height % 2 != 0:
            self.height += 1
        
        cap.release()

        if self.duration_sec <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Propiedades de video inválidas")
            
        print(f"Video preparado: {self.duration_sec:.1f}s, {self.width}x{self.height}, {self.fps:.1f}fps")

    def _analyze_detections(self, frame) -> Dict:
        """Analiza detecciones YOLO y emociones en un frame"""
        if self.model is None:
            return {
                'total_detections': 0,
                'students_sitting': 0,
                'students_standing': 0,
                'teachers_sitting': 0,
                'teachers_standing': 0,
                'detection_score': 0.0,
                'detections': [],
                'emotion_stats': {},
                'student_status_stats': {}
            }
        
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = {
            'student_sitting': 0,
            'student_standing': 0,
            'teacher_sitting': 0,
            'teacher_standing': 0
        }
        
        detection_details = []
        total_confidence = 0.0
        emotion_counts = {}
        status_counts = {}
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                student_detections = []
                
                for box in boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if cls_id < len(self.model.names):
                        class_name = self.model.names[cls_id]
                        
                        if class_name in detections:
                            detections[class_name] += 1
                            total_confidence += confidence
                            
                            xyxy = box.xyxy.cpu().numpy()[0]
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            detection_info = {
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2]
                            }
                            
                            # Agregar estudiantes para análisis de emociones
                            if 'student' in class_name:
                                student_detections.append(detection_info)
                            
                            detection_details.append(detection_info)
                
                # Análisis de emociones
                if self.emotion_analyzer and student_detections:
                    try:
                        emotion_results = self.emotion_analyzer.batch_analyze_detections(
                            frame, student_detections
                        )
                        
                        for result in emotion_results:
                            detection_info = result['detection']
                            emotion_analysis = result['emotion_analysis']
                            student_status = result.get('student_status', {})
                            
                            for det in detection_details:
                                if (det['bbox'] == detection_info['bbox'] and 
                                    det['class'] == detection_info['class']):
                                    
                                    det['emotion_analysis'] = emotion_analysis
                                    det['student_status'] = student_status
                                    
                                    if emotion_analysis.get('success'):
                                        emotion = emotion_analysis['emotion']
                                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                                    
                                    if student_status.get('status'):
                                        status = student_status['status']
                                        status_counts[status] = status_counts.get(status, 0) + 1
                                    
                                    break
                    
                    except Exception as e:
                        print(f"Error en análisis de emociones: {str(e)}")
        
        total_detections = sum(detections.values())
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
        
        return {
            'total_detections': total_detections,
            'students_sitting': detections.get('student_sitting', 0),
            'students_standing': detections.get('student_standing', 0),
            'teachers_sitting': detections.get('teacher_sitting', 0),
            'teachers_standing': detections.get('teacher_standing', 0),
            'detection_score': avg_confidence,
            'detections': detection_details,
            'emotion_stats': emotion_counts,
            'student_status_stats': status_counts
        }

    def _should_extract_clip(self, detection_data: Dict) -> bool:
        """Determina si se debe extraer un clip basado en detecciones"""
        if not self.detection_analysis or self.model is None:
            return True
        
        # Criterios simplificados: debe tener detecciones con buena confianza
        has_detections = detection_data['total_detections'] > 0
        has_good_confidence = detection_data['detection_score'] > self.confidence_threshold
        
        return has_detections and has_good_confidence

    def _check_ffmpeg_available(self) -> bool:
        """Verifica si ffmpeg está disponible y tiene codecs necesarios"""
        try:
            if os.path.exists(self.ffmpeg_path):
                result = subprocess.run([self.ffmpeg_path, '-version'], 
                                    capture_output=True, check=True, text=True)
                # Verificar soporte H.264
                codec_result = subprocess.run([self.ffmpeg_path, '-codecs'], 
                                            capture_output=True, text=True)
                if 'h264' in codec_result.stdout.lower():
                    print("✓ FFmpeg con soporte H.264 disponible")
                    return True
                else:
                    print("⚠ FFmpeg sin soporte H.264")
                    return False
            else:
                result = subprocess.run(['ffmpeg', '-version'], 
                                    capture_output=True, check=True, text=True)
                self.ffmpeg_path = 'ffmpeg'
                codec_result = subprocess.run(['ffmpeg', '-codecs'], 
                                            capture_output=True, text=True)
                if 'h264' in codec_result.stdout.lower():
                    print("✓ FFmpeg con soporte H.264 disponible")
                    return True
                else:
                    print("⚠ FFmpeg sin soporte H.264")
                    return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ FFmpeg no disponible")
            return False

    def _extract_clip_opencv_with_yolo(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip usando OpenCV aplicando detecciones YOLO"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmpfile:
            temp_video_path = tmpfile.name
        
        fourcc_options = [
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ]

        writer = None
        codec_used = None
        
        for codec_name, fourcc_option in fourcc_options:
            try:
                writer = cv2.VideoWriter(temp_video_path, fourcc_option, self.fps, (self.width, self.height))
                if writer.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        if test_frame.shape[1] != self.width or test_frame.shape[0] != self.height:
                            test_frame = cv2.resize(test_frame, (self.width, self.height))
                        writer.write(test_frame)
                        codec_used = codec_name
                        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                        break
                else:
                    writer.release()
            except Exception as e:
                if writer:
                    writer.release()
                continue
        
        if writer is None or not writer.isOpened():
            cap.release()
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return False
        
        print(f"✓ Procesando con YOLO usando codec: {codec_used}")
        
        frames_written = 0
        target_frames = int((end_time - start_time) * self.fps)
        
        colors = {
            'student_sitting': (255, 0, 0),
            'student_standing': (255, 255, 0),
            'teacher_sitting': (0, 165, 255),
            'teacher_standing': (0, 0, 255)
        }
        
        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Aplicar detecciones YOLO
            if self.model is not None:
                try:
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                    
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None:
                            for box in boxes:
                                cls_id = int(box.cls.cpu().numpy()[0])
                                confidence = float(box.conf.cpu().numpy()[0])
                                xyxy = box.xyxy.cpu().numpy()[0]
                                
                                if cls_id < len(self.model.names):
                                    class_name = self.model.names[cls_id]
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    
                                    color = colors.get(class_name, (0, 255, 0))
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    label = f"{class_name}: {confidence:.2f}"
                                    
                                    # Análisis de emociones
                                    emotion_text = ""
                                    if self.emotion_analyzer and class_name in ['student_sitting', 'student_standing']:
                                        try:
                                            emotion_result = self.emotion_analyzer.analyze_face_in_detection(
                                                frame, (x1, y1, x2, y2)
                                            )
                                            
                                            if emotion_result.get('success'):
                                                emotion = emotion_result['emotion']
                                                
                                                posture = 'sitting' if 'sitting' in class_name else 'standing'
                                                status_result = self.emotion_analyzer.determine_student_status_enhanced(
                                                    posture, emotion_result
                                                )
                                                
                                                emotion_text = f" ({emotion})"
                                                if status_result.get('status'):
                                                    emotion_text += f" [{status_result['status']}]"
                                            
                                        except Exception as e:
                                            print(f"Error analizando emociones: {str(e)}")
                                    
                                    full_label = f"{label}{emotion_text}"
                                    label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    cv2.rectangle(frame, 
                                                (x1, y1 - label_size[1] - 10), 
                                                (x1 + label_size[0], y1), 
                                                color, -1)
                                    
                                    cv2.putText(frame, full_label, 
                                            (x1, y1 - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                            (255, 255, 255), 2)
                                    
                except Exception as e:
                    print(f"Error aplicando YOLO al frame {frames_written}: {str(e)}")
            
            if frame is not None and frame.size > 0:
                writer.write(frame)
                frames_written += 1
        
        writer.release()
        cap.release()
        
        # Convertir a MP4 con FFmpeg si está disponible
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
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-map', '1:v:0',
                    '-map', '0:a:0?',
                    '-shortest',
                    '-movflags', '+faststart',
                    output_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                os.remove(temp_video_path)
                
                if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"✓ Video final con audio: {output_path}")
                    return True
                else:
                    print(f"✗ Error FFmpeg: {result.stderr}")
                    if os.path.exists(temp_video_path):
                        import shutil
                        shutil.move(temp_video_path, output_path)
                        return True
                    return False
                    
            except Exception as e:
                print(f"Error combinando video y audio: {e}")
                if os.path.exists(temp_video_path):
                    import shutil
                    shutil.move(temp_video_path, output_path)
                    return True
                return False
        else:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return False

    def _extract_clip(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extrae un clip del video"""
        if self.model is not None and self.detection_analysis and (self.yolo_in_video or self.emotion_analyzer):
            return self._extract_clip_opencv_with_yolo(start_time, end_time, output_path)
        return False

    def _analyze_segment(self, start_time: float) -> Dict:
        """Analiza un segmento del video para detecciones"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {}
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        detection_results = []
        frames_to_analyze = min(5, int(self.clip_duration_sec * self.original_fps / 20))
        
        for i in range(frames_to_analyze):
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.detection_analysis and self.model is not None:
                det_result = self._analyze_detections(frame)
                detection_results.append(det_result)
            
            # Saltar algunos frames para acelerar el análisis
            for _ in range(int(self.original_fps / 2)):
                ret, _ = cap.read()
                if not ret:
                    break
        
        cap.release()
        
        if detection_results:
            return {
                'total_detections': np.mean([d['total_detections'] for d in detection_results]),
                'students_sitting': np.mean([d['students_sitting'] for d in detection_results]),
                'students_standing': np.mean([d['students_standing'] for d in detection_results]),
                'teachers_sitting': np.mean([d['teachers_sitting'] for d in detection_results]),
                'teachers_standing': np.mean([d['teachers_standing'] for d in detection_results]),
                'detection_score': np.mean([d['detection_score'] for d in detection_results]),
                'best_frame_detections': max(detection_results, key=lambda x: x['total_detections'])['detections'],
                'emotion_stats': self._combine_emotion_stats([d.get('emotion_stats', {}) for d in detection_results])
            }
        else:
            return {
                'total_detections': 0,
                'students_sitting': 0,
                'students_standing': 0,
                'teachers_sitting': 0,
                'teachers_standing': 0,
                'detection_score': 0.0,
                'best_frame_detections': [],
                'emotion_stats': {}
            }

    def _combine_emotion_stats(self, emotion_stats_list):
        """Combina estadísticas de emociones de múltiples frames"""
        combined = {}
        for stats in emotion_stats_list:
            for emotion, count in stats.items():
                combined[emotion] = combined.get(emotion, 0) + count
        return combined

    def extract_clips(self) -> List[Dict]:
        """Extrae clips según los intervalos configurados"""
        clips_meta = []
        current_time = 0.0
        
        print(f"Iniciando extracción de clips cada {self.interval_seconds}s")
        
        if self.has_ffmpeg:
            print("✓ FFmpeg detectado - Los clips incluirán audio")
        else:
            print("⚠ FFmpeg no disponible - Los clips NO tendrán audio")
        
        if self.model:
            print(f"✓ Modelo YOLO activo - Analizando detecciones escolares")
        
        if self.emotion_analyzer:
            print("✓ Modelo de emociones activo - Analizando estados emocionales")
        
        while current_time < self.duration_sec:
            end_time = min(current_time + self.clip_duration_sec, self.duration_sec)
            
            if (end_time - current_time) < (self.clip_duration_sec * 0.5):
                break
            
            print(f"Analizando segmento: {current_time:.1f}s - {end_time:.1f}s")
            
            detection_data = self._analyze_segment(current_time)
            should_extract = self._should_extract_clip(detection_data)
            
            print(f"  Detecciones: {detection_data.get('total_detections', 0):.1f}")
            print(f"  Estudiantes sentados: {detection_data.get('students_sitting', 0):.1f}")
            print(f"  Estudiantes parados: {detection_data.get('students_standing', 0):.1f}")
            print(f"  Maestros sentados: {detection_data.get('teachers_sitting', 0):.1f}")
            print(f"  Maestros parados: {detection_data.get('teachers_standing', 0):.1f}")
            
            if self.emotion_analyzer:
                print("  Emociones detectadas:")
                for emotion, count in detection_data.get('emotion_stats', {}).items():
                    if count > 0:
                        print(f"    {emotion}: {count}")
            
            print(f"  Extraer: {should_extract}")
            
            if should_extract:
                clip_filename = f"clip_{int(current_time):04d}s.mp4"
                output_path = os.path.join(self.output_folder, clip_filename)
                
                if self._extract_clip(current_time, end_time, output_path):
                    clips_meta.append({
                        "filename": clip_filename,
                        "start_time": round(current_time, 2),
                        "duration": round(end_time - current_time, 2),
                        "has_audio": self.has_ffmpeg,
                        "detections": {
                            "total": round(detection_data.get('total_detections', 0), 1),
                            "students_sitting": round(detection_data.get('students_sitting', 0), 1),
                            "students_standing": round(detection_data.get('students_standing', 0), 1),
                            "teachers_sitting": round(detection_data.get('teachers_sitting', 0), 1),
                            "teachers_standing": round(detection_data.get('teachers_standing', 0), 1),
                            "avg_confidence": round(detection_data.get('detection_score', 0), 3),
                            "emotion_stats": detection_data.get('emotion_stats', {})
                        }
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
        if self.emotion_analyzer:
            print(f"Modelo de emociones: {self.emotion_model_path}")
        print(f"Configuración: intervalo={self.interval_seconds}s, duración={self.clip_duration_sec}s")
        
        clips_meta = self.extract_clips()
        
        # Guardar metadatos
        video_metadata = {
            "video": {
                "path": self.video_path,
                "duration": self.duration_sec,
                "fps": self.fps,
                "resolution": f"{self.width}x{self.height}",
                "ffmpeg_available": self.has_ffmpeg
            },
            "model": {
                "path": self.model_path,
                "available": self.model is not None,
                "classes": list(self.model.names.values()) if self.model else [],
                "confidence_threshold": self.confidence_threshold
            },
            "emotion_model": {
                "path": self.emotion_model_path,
                "available": self.emotion_analyzer is not None,
                "classes": list(self.emotion_analyzer.emotion_labels.values()) if self.emotion_analyzer else []
            },
            "settings": {
                "interval_seconds": self.interval_seconds,
                "clip_duration_sec": self.clip_duration_sec,
                "detection_analysis": self.detection_analysis,
                "yolo_in_video": self.yolo_in_video
            },
            "clips_meta": clips_meta
        }
        
        metadata_path = os.path.join(self.output_folder, "clips_metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False)
        
        # Calcular estadísticas resumen
        if clips_meta:
            total_students_sitting = sum(c["detections"]["students_sitting"] for c in clips_meta)
            total_students_standing = sum(c["detections"]["students_standing"] for c in clips_meta)
            total_teachers_sitting = sum(c["detections"]["teachers_sitting"] for c in clips_meta)
            total_teachers_standing = sum(c["detections"]["teachers_standing"] for c in clips_meta)
            
            emotion_totals = {}
            if self.emotion_analyzer:
                for emotion in self.emotion_analyzer.emotion_labels.values():
                    emotion_totals[emotion] = sum(
                        c["detections"]["emotion_stats"].get(emotion, 0) 
                        for c in clips_meta
                    )
            
            most_detections_clip = max(clips_meta, key=lambda x: x["detections"]["total"])
            
            summary = {
                "total_clips": len(clips_meta),
                "most_detections_clip": most_detections_clip["filename"],
                "detection_summary": {
                    "total_students_sitting": total_students_sitting,
                    "total_students_standing": total_students_standing,
                    "total_teachers_sitting": total_teachers_sitting,
                    "total_teachers_standing": total_teachers_standing,
                    "clips_with_students": len([c for c in clips_meta if c["detections"]["students_sitting"] + c["detections"]["students_standing"] > 0]),
                    "clips_with_teachers": len([c for c in clips_meta if c["detections"]["teachers_sitting"] + c["detections"]["teachers_standing"] > 0])
                },
                "emotion_summary": emotion_totals,
                "clips_meta": clips_meta,
                "audio_included": any(c.get("has_audio", False) for c in clips_meta)
            }
        else:
            summary = {
                "total_clips": 0,
                "most_detections_clip": None,
                "detection_summary": {
                    "total_students_sitting": 0,
                    "total_students_standing": 0,
                    "total_teachers_sitting": 0,
                    "total_teachers_standing": 0,
                    "clips_with_students": 0,
                    "clips_with_teachers": 0
                },
                "emotion_summary": {},
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
    confidence_threshold: float = 0.5,
    detection_analysis: bool = True,
    yolo_in_video: bool = True,
    emotion_model_path: str = None
) -> Dict:
    """
    Procesa un video escolar y extrae clips con análisis YOLO y de emociones.
    """
    pipeline = SchoolYOLOVideoPipeline(
        video_path = video_path,
        model_path = model_path,
        output_folder = output_folder,
        interval_seconds = interval_seconds,
        clip_duration_sec = clip_duration_sec,
        confidence_threshold = confidence_threshold,
        detection_analysis = detection_analysis,
        yolo_in_video = yolo_in_video,
        emotion_model_path = emotion_model_path
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