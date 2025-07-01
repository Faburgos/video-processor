import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from typing import Tuple, Optional, List, Dict

class ImprovedEmotionAnalyzer:
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        """
        Analizador de emociones mejorado con múltiples validaciones.
        
        Args:
            model_path: Ruta al modelo de emociones (.h5)
            confidence_threshold: Umbral mínimo de confianza para considerar válida una predicción
        """
        self.confidence_threshold = confidence_threshold
        self.emotion_labels = {
            0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
            4: 'sad', 5: 'surprised', 6: 'neutral'
        }
        
        # Cargar modelo de emociones
        try:
            self.emotion_model = tf.keras.models.load_model(model_path, compile = False)
            print(f"✓ Modelo de emociones cargado: {model_path}")
        except Exception as e:
            print(f"✗ Error cargando modelo de emociones: {e}")
            self.emotion_model = None
        
        # Inicializar MediaPipe para detección de rostros más precisa
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection = 1,  # Modelo optimizado para rostros distantes
                min_detection_confidence = 0.7
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode = True,
                max_num_faces = 5,
                refine_landmarks = True,
                min_detection_confidence = 0.7,
                min_tracking_confidence = 0.5
            )
            print("✓ MediaPipe Face Detection inicializado")
        except Exception as e:
            print(f"⚠ MediaPipe no disponible, usando fallback: {e}")
            self.mp_face_detection = None
            self.face_detection = None
            self.face_mesh = None
        
        # Fallback: Haar Cascades mejorado
        try:
            self.haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.haar_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception as e:
            print(f"⚠ Error inicializando Haar Cascades: {e}")
            self.haar_face = None
            self.haar_profile = None

    def _validate_face_quality(self, face_img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Valida la calidad de un rostro detectado.
        
        Args:
            face_img: Región del rostro extraída
            bbox: Coordenadas de la caja delimitadora (x1, y1, x2, y2)
            
        Returns:
            Dict con métricas de calidad
        """
        if face_img is None or face_img.size == 0:
            return {'valid': False, 'reason': 'imagen_vacia'}
        
        h, w = face_img.shape[:2]
        
        # 1. Validar tamaño mínimo
        if w < 30 or h < 30:
            return {'valid': False, 'reason': 'muy_pequeño', 'size': (w, h)}
        
        # 2. Validar relación de aspecto (rostros humanos ~0.75-1.3)
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.5:
            return {'valid': False, 'reason': 'aspecto_invalido', 'aspect_ratio': aspect_ratio}
        
        # 3. Validar nitidez (varianza del Laplaciano)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 50:  # Muy borroso
            return {'valid': False, 'reason': 'muy_borroso', 'sharpness': sharpness}
        
        # 4. Validar contraste
        std_deviation = np.std(gray)
        if std_deviation < 15:  # Muy poco contraste
            return {'valid': False, 'reason': 'bajo_contraste', 'std': std_deviation}
        
        # 5. Validar iluminación (evitar rostros muy oscuros o muy claros)
        mean_brightness = np.mean(gray)
        if mean_brightness < 20 or mean_brightness > 240:
            return {'valid': False, 'reason': 'iluminacion_extrema', 'brightness': mean_brightness}
        
        return {
            'valid': True,
            'size': (w, h),
            'aspect_ratio': aspect_ratio,
            'sharpness': sharpness,
            'contrast': std_deviation,
            'brightness': mean_brightness
        }

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detecta rostros usando MediaPipe (más preciso).
        
        Returns:
            Lista de (x1, y1, x2, y2, confidence)
        """
        if not self.face_detection:
            return []
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    
                    # Convertir coordenadas relativas a absolutas
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    # Validar coordenadas
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2, y2, confidence))
            
            return sorted(faces, key = lambda x: x[4], reverse = True)  # Ordenar por confianza
            
        except Exception as e:
            print(f"Error en detección MediaPipe: {e}")
            return []

    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detecta rostros usando Haar Cascades como fallback.
        
        Returns:
            Lista de (x1, y1, x2, y2, confidence)
        """
        if not self.haar_face:
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros frontales
            faces_front = self.haar_face.detectMultiScale(
                gray, scaleFactor = 1.1, minNeighbors = 5,
                minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detectar rostros de perfil
            faces_profile = []
            if self.haar_profile:
                faces_profile = self.haar_profile.detectMultiScale(
                    gray, scaleFactor = 1.1, minNeighbors = 5,
                    minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE
                )
            
            # Combinar detecciones
            all_faces = []
            
            for (x, y, w, h) in faces_front:
                all_faces.append((x, y, x+w, y+h, 0.8))  # Confianza estimada
                
            for (x, y, w, h) in faces_profile:
                all_faces.append((x, y, x+w, y+h, 0.7))  # Menor confianza para perfiles

            return sorted(all_faces, key = lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse = True)  # Por tamaño

        except Exception as e:
            print(f"Error en detección Haar: {e}")
            return []

    def _find_best_face_in_bbox(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Encuentra el mejor rostro dentro de una detección YOLO.
        
        Args:
            image: Imagen completa
            yolo_bbox: Caja delimitadora de YOLO (x1, y1, x2, y2)
            
        Returns:
            Tuple de (face_image, quality_metrics) o None si no encuentra rostro válido
        """
        x1, y1, x2, y2 = yolo_bbox
        
        # Expandir ligeramente la región YOLO para capturar rostros en los bordes
        padding = 10
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extraer región de interés
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Detectar rostros en la ROI
        faces_mp = self._detect_faces_mediapipe(roi)
        faces_haar = self._detect_faces_haar(roi) if not faces_mp else []
        
        all_faces = faces_mp + faces_haar
        
        if not all_faces:
            return None
        
        # Evaluar cada rostro detectado
        best_face = None
        best_score = 0
        best_quality = None
        
        for face_coords in all_faces:
            fx1, fy1, fx2, fy2 = face_coords[:4]
            face_confidence = face_coords[4] if len(face_coords) > 4 else 0.5
            
            # Extraer rostro
            face_img = roi[fy1:fy2, fx1:fx2]
            
            # Validar calidad
            quality = self._validate_face_quality(face_img, (fx1, fy1, fx2, fy2))
            
            if not quality['valid']:
                continue
            
            # Calcular puntuación combinada
            score = (
                face_confidence * 0.3 +
                min(quality['sharpness'] / 200, 1.0) * 0.25 +
                min(quality['contrast'] / 50, 1.0) * 0.25 +
                min((fx2-fx1) * (fy2-fy1) / 10000, 1.0) * 0.2  # Tamaño
            )
            
            if score > best_score:
                best_score = score
                best_face = face_img
                best_quality = quality
        
        return (best_face, best_quality) if best_face is not None else None

    def _preprocess_face_for_emotion(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocesa el rostro para el modelo de emociones con técnicas avanzadas.
        """
        if face_img is None or face_img.size == 0:
            return None
        
        # Convertir a escala de grises si es necesario
        if len(face_img.shape) == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img.copy()
        
        # 1. Ecualización de histograma adaptativo para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        enhanced_face = clahe.apply(gray_face)
        
        # 2. Filtro bilateral para reducir ruido pero mantener bordes
        denoised_face = cv2.bilateralFilter(enhanced_face, 9, 75, 75)
        
        # 3. Redimensionar con interpolación de alta calidad
        resized_face = cv2.resize(denoised_face, (48, 48), interpolation=cv2.INTER_CUBIC)
        
        # 4. Normalización robusta
        face_float = resized_face.astype(np.float32)
        
        # Normalización por percentiles (más robusta que min-max)
        p1, p99 = np.percentile(face_float, (1, 99))
        face_normalized = np.clip((face_float - p1) / (p99 - p1), 0, 1)
        
        # 5. Preparar para el modelo
        face_batch = np.expand_dims(face_normalized, axis = (0, -1))
        
        return face_batch

    def analyze_face_in_detection(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analiza emociones en una detección YOLO específica.
        
        Args:
            image: Imagen completa
            yolo_bbox: Coordenadas de la detección YOLO (x1, y1, x2, y2)
            
        Returns:
            Dict con resultados del análisis
        """
        if self.emotion_model is None:
            return {
                'success': False,
                'error': 'modelo_no_disponible',
                'emotion': None,
                'confidence': 0.0,
                'face_quality': None
            }
        
        # Buscar el mejor rostro en la detección
        face_result = self._find_best_face_in_bbox(image, yolo_bbox)
        
        if not face_result:
            return {
                'success': False,
                'error': 'no_face_detected',
                'emotion': None,
                'confidence': 0.0,
                'face_quality': None
            }
        
        face_img, quality_metrics = face_result
        
        try:
            # Preprocesar rostro
            processed_face = self._preprocess_face_for_emotion(face_img)
            
            if processed_face is None:
                return {
                    'success': False,
                    'error': 'preprocessing_failed',
                    'emotion': None,
                    'confidence': 0.0,
                    'face_quality': quality_metrics
                }
            
            # Realizar predicción
            predictions = self.emotion_model.predict(processed_face, verbose=0)
            emotion_probs = predictions[0]
            
            # Obtener emoción más probable
            emotion_id = np.argmax(emotion_probs)
            confidence = float(emotion_probs[emotion_id])
            emotion_name = self.emotion_labels[emotion_id]
            
            # Validar confianza mínima
            if confidence < self.confidence_threshold:
                return {
                    'success': False,
                    'error': 'low_confidence',
                    'emotion': emotion_name,
                    'confidence': confidence,
                    'face_quality': quality_metrics,
                    'all_probabilities': {self.emotion_labels[i]: float(emotion_probs[i]) for i in range(len(emotion_probs))}
                }
            
            return {
                'success': True,
                'emotion': emotion_name,
                'confidence': confidence,
                'face_quality': quality_metrics,
                'all_probabilities': {self.emotion_labels[i]: float(emotion_probs[i]) for i in range(len(emotion_probs))}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'prediction_error: {str(e)}',
                'emotion': None,
                'confidence': 0.0,
                'face_quality': quality_metrics
            }

    def determine_student_status_enhanced(self, posture: str, emotion_result: Dict, context: Dict = None) -> Dict:
        """
        Determina el estado del estudiante con análisis contextual mejorado.
        
        Args:
            posture: 'sitting' o 'standing'
            emotion_result: Resultado del análisis de emociones
            context: Información contextual adicional (opcional)
            
        Returns:
            Dict con el estado interpretado y confianza
        """
        if not emotion_result['success']:
            return {
                'status': 'indeterminado',
                'confidence': 0.0,
                'reason': f"emoción no detectada: {emotion_result.get('error', 'unknown')}"
            }
        
        emotion = emotion_result['emotion']
        emotion_confidence = emotion_result['confidence']
        face_quality = emotion_result.get('face_quality', {})
        
        # Mapeo de estados mejorado
        status_map = {
            ('sitting', 'neutral'): ('atento', 0.8),
            ('sitting', 'happy'): ('participativo/contento', 0.9),
            ('sitting', 'surprised'): ('interesado/sorprendido', 0.7),
            ('sitting', 'sad'): ('desanimado/distraído', 0.8),
            ('sitting', 'angry'): ('frustrado/molesto', 0.8),
            ('sitting', 'disgusted'): ('desinteresado/aburrido', 0.7),
            ('sitting', 'fearful'): ('ansioso/preocupado', 0.7),
            
            ('standing', 'happy'): ('participativo activo', 0.8),
            ('standing', 'neutral'): ('esperando/transitorio', 0.6),
            ('standing', 'surprised'): ('interrumpido/alterado', 0.8),
            ('standing', 'angry'): ('disruptivo/confrontativo', 0.9),
            ('standing', 'sad'): ('necesita atención', 0.7),
            ('standing', 'fearful'): ('nervioso/intimidado', 0.8),
            ('standing', 'disgusted'): ('resistente/desafiante', 0.7)
        }
        
        # Buscar estado correspondiente
        key = (posture, emotion)
        if key in status_map:
            status, base_confidence = status_map[key]
        else:
            status = 'indeterminado'
            base_confidence = 0.3
        
        # Ajustar confianza basada en calidad del rostro y emoción
        quality_factor = 1.0
        if face_quality and face_quality.get('valid'):
            sharpness_factor = min(face_quality.get('sharpness', 100) / 100, 1.0)
            contrast_factor = min(face_quality.get('contrast', 30) / 30, 1.0)
            quality_factor = (sharpness_factor + contrast_factor) / 2
        
        final_confidence = base_confidence * emotion_confidence * quality_factor
        
        return {
            'status': status,
            'confidence': round(final_confidence, 3),
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'face_quality_score': quality_factor,
            'reasoning': f"Postura: {posture}, Emoción: {emotion} ({emotion_confidence:.2f}), Calidad rostro: {quality_factor:.2f}"
        }

    def batch_analyze_detections(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Analiza múltiples detecciones en lote para mejor eficiencia.
        
        Args:
            image: Imagen completa
            detections: Lista de detecciones YOLO con 'bbox' y 'class'
            
        Returns:
            Lista de resultados de análisis
        """
        results = []
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            class_name = detection.get('class', '')
            
            if len(bbox) != 4 or 'student' not in class_name:
                results.append({
                    'detection': detection,
                    'emotion_analysis': {'success': False, 'error': 'not_student_or_invalid_bbox'}
                })
                continue
            
            # Determinar postura
            posture = 'sitting' if 'sitting' in class_name else 'standing'
            
            # Analizar emoción
            emotion_result = self.analyze_face_in_detection(image, tuple(bbox))
            
            # Determinar estado
            status_result = self.determine_student_status_enhanced(posture, emotion_result)
            
            results.append({
                'detection': detection,
                'emotion_analysis': emotion_result,
                'student_status': status_result,
                'posture': posture
            })
        
        return results