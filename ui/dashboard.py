import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from utils.helpers import format_time, safe_convert_metric_value

@dataclass
class EducationalMetrics:
    """Clase para encapsular métricas educacionales calculadas"""
    attendance_rate: float
    engagement_score: float
    attention_score: float
    participation_level: float
    emotional_wellness: float
    class_dynamics: str
    peak_attention_time: str
    recommendations: List[str]

class EducationalAnalyzer:
    """Analizador avanzado de métricas educacionales"""
    
    def __init__(self, clips_meta: List[Dict], emotion_summary: Dict):
        self.clips_meta = clips_meta
        self.emotion_summary = emotion_summary
        self.total_clips = len(clips_meta)
        
    def calculate_attendance_metrics(self) -> Dict[str, float]:
        """Calcula métricas de asistencia y presencia"""
        total_students = sum(
            self._get_total_students(meta) for meta in self.clips_meta
        )
        total_teachers = sum(
            self._get_total_teachers(meta) for meta in self.clips_meta
        )
        
        avg_students = total_students / self.total_clips if self.total_clips > 0 else 0
        avg_teachers = total_teachers / self.total_clips if self.total_clips > 0 else 0
        
        # Calcular consistencia de asistencia (variabilidad)
        student_counts = [self._get_total_students(meta) for meta in self.clips_meta]
        attendance_consistency = (
            1 - (np.std(student_counts) / np.mean(student_counts))
            if np.mean(student_counts) > 0 else 0
        )
        
        return {
            'avg_students': avg_students,
            'avg_teachers': avg_teachers,
            'total_students': total_students,
            'total_teachers': total_teachers,
            'attendance_consistency': max(0, min(1, attendance_consistency))
        }
    
    def calculate_engagement_score(self) -> Dict[str, float]:
        """Calcula un score de engagement basado en emociones y participación"""
        if not self.emotion_summary:
            return {'engagement_score': 0, 'emotional_wellness': 0}
        
        total_emotions = sum(self.emotion_summary.values())
        if total_emotions == 0:
            return {'engagement_score': 0, 'emotional_wellness': 0}
        
        # Pesos para diferentes emociones (basado en investigación educacional)
        emotion_weights = {
            'happy': 1.0,      # Muy positivo para el aprendizaje
            'surprise': 0.8,   # Indica curiosidad/interés
            'neutral': 0.4,    # Neutral, no negativo pero tampoco muy positivo
            'sad': -0.3,       # Ligeramente negativo
            'fear': -0.5,      # Puede inhibir el aprendizaje
            'angry': -0.7      # Muy negativo para el ambiente
        }
        
        # Calcular score ponderado
        weighted_score = sum(
            self.emotion_summary.get(emotion, 0) * weight 
            for emotion, weight in emotion_weights.items()
        )
        
        # Normalizar a 0-100
        engagement_score = max(0, min(100, (weighted_score / total_emotions) * 100 + 50))
        
        # Calcular bienestar emocional (solo emociones positivas vs negativas)
        positive_emotions = (
            self.emotion_summary.get('happy', 0) + 
            self.emotion_summary.get('surprise', 0)
        )
        negative_emotions = (
            self.emotion_summary.get('sad', 0) + 
            self.emotion_summary.get('angry', 0) + 
            self.emotion_summary.get('fear', 0)
        )
        
        emotional_wellness = (
            positive_emotions / (positive_emotions + negative_emotions) * 100
            if (positive_emotions + negative_emotions) > 0 else 50
        )
        
        return {
            'engagement_score': engagement_score,
            'emotional_wellness': emotional_wellness,
            'positive_ratio': positive_emotions / total_emotions if total_emotions > 0 else 0
        }
    
    def calculate_attention_patterns(self) -> Dict[str, any]:
        """Analiza patrones de atención basado en movimiento y postura"""
        sitting_students = []
        standing_students = []
        motion_scores = []
        times = []
        
        for meta in self.clips_meta:
            detections = meta.get("detections", {})
            sitting = detections.get("students_sitting", 0)
            standing = detections.get("students_standing", 0)
            motion = meta.get("motion_score", 0)
            
            sitting_students.append(sitting)
            standing_students.append(standing)
            motion_scores.append(motion)
            times.append(meta.get("start_time", 0))
        
        # Calcular score de atención
        # Asumimos que estudiantes sentados + bajo movimiento = mayor atención
        attention_scores = []
        for i in range(len(sitting_students)):
            total_students = sitting_students[i] + standing_students[i]
            if total_students > 0:
                sitting_ratio = sitting_students[i] / total_students
                # Atención = (% sentados) * (1 - movimiento_excesivo)
                motion_factor = max(0, 1 - motion_scores[i] * 10)  # Penalizar movimiento excesivo
                attention_score = sitting_ratio * motion_factor * 100
            else:
                attention_score = 0
            attention_scores.append(attention_score)
        
        avg_attention = np.mean(attention_scores) if attention_scores else 0
        
        # Encontrar momento de mayor atención
        if attention_scores and times:
            peak_attention_idx = np.argmax(attention_scores)
            peak_attention_time = format_time(times[peak_attention_idx])
        else:
            peak_attention_time = "N/A"
        
        return {
            'avg_attention_score': avg_attention,
            'attention_scores': attention_scores,
            'peak_attention_time': peak_attention_time,
            'attention_trend': self._calculate_trend(attention_scores)
        }
    
    def analyze_class_dynamics(self) -> Dict[str, any]:
        """Analiza la dinámica de la clase"""
        teacher_student_ratios = []
        participation_levels = []
        
        for meta in self.clips_meta:
            detections = meta.get("detections", {})
            total_students = self._get_total_students(meta)
            total_teachers = self._get_total_teachers(meta)
            
            # Ratio maestro-estudiante
            ratio = total_teachers / total_students if total_students > 0 else 0
            teacher_student_ratios.append(ratio)
            
            # Nivel de participación (basado en estudiantes de pie vs sentados)
            sitting = detections.get("students_sitting", 0)
            standing = detections.get("students_standing", 0)
            
            if total_students > 0:
                participation = standing / total_students
            else:
                participation = 0
            participation_levels.append(participation)
        
        avg_ratio = np.mean(teacher_student_ratios) if teacher_student_ratios else 0
        avg_participation = np.mean(participation_levels) if participation_levels else 0
        
        # Clasificar dinámica de clase
        if avg_ratio > 0.3:
            class_type = "Clase Individual/Grupos Pequeños"
        elif avg_ratio > 0.1:
            class_type = "Clase Estándar"
        else:
            class_type = "Clase Masiva"
        
        return {
            'avg_teacher_student_ratio': avg_ratio,
            'avg_participation_level': avg_participation * 100,
            'class_dynamics': class_type,
            'participation_trend': self._calculate_trend(participation_levels)
        }
    
    def generate_insights(self, metrics: Dict) -> List[str]:
        """Genera insights basados en todas las métricas"""
        insights = []
        
        # Insights de asistencia
        if metrics['attendance']['avg_students'] < 3:
            insights.append("🚨 Asistencia muy baja. Revisar factores que afectan la participación.")
        elif metrics['attendance']['avg_students'] > 20:
            insights.append("⚠️ Clase muy numerosa. Considerar dividir en grupos más pequeños.")
        
        # Insights de engagement
        if metrics['engagement']['engagement_score'] < 40:
            insights.append("📉 Engagement bajo. Implementar actividades más interactivas.")
        elif metrics['engagement']['engagement_score'] > 75:
            insights.append("🎉 Excelente engagement. Las estrategias actuales son efectivas.")
        
        # Insights de atención
        if metrics['attention']['avg_attention_score'] < 50:
            insights.append("💭 Nivel de atención bajo. Considerar descansos más frecuentes.")
        
        # Insights de bienestar emocional
        if metrics['engagement']['emotional_wellness'] < 60:
            insights.append("😟 Bienestar emocional mejorable. Evaluar el ambiente de clase.")
        
        # Insights de participación
        if metrics['dynamics']['avg_participation_level'] < 20:
            insights.append("🤐 Participación baja. Fomentar más interacción estudiante-maestro.")
        elif metrics['dynamics']['avg_participation_level'] > 60:
            insights.append("🗣️ Alta participación detectada. Excelente interacción en clase.")
        
        if not insights:
            insights.append("✅ Todas las métricas dentro de rangos saludables.")
        
        return insights
    
    def _get_total_students(self, meta: Dict) -> int:
        """Helper para obtener total de estudiantes"""
        detections = meta.get("detections", {})
        return (
            detections.get("students_sitting", 0) + 
            detections.get("students_standing", 0)
        )
    
    def _get_total_teachers(self, meta: Dict) -> int:
        """Helper para obtener total de maestros"""
        detections = meta.get("detections", {})
        return (
            detections.get("teachers_sitting", 0) + 
            detections.get("teachers_standing", 0)
        )
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula la tendencia de una serie de valores"""
        if len(values) < 2:
            return "Estable"
        
        # Calcular pendiente usando regresión lineal simple
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "Creciente"
        elif slope < -0.01:
            return "Decreciente"
        else:
            return "Estable"


def create_metrics_dashboard(result):
    """Crea un dashboard avanzado de métricas educacionales"""
    clips = result.get("clips", [])
    summary = result.get("summary", {})
    clips_meta = summary.get("clips_meta", [])
    emotion_summary = summary.get("emotion_summary", {})
    
    if not clips_meta:
        st.warning("⚠️ No hay datos de clips para mostrar métricas.")
        return
    
    # Inicializar el analizador
    analyzer = EducationalAnalyzer(clips_meta, emotion_summary)
    
    # Calcular todas las métricas
    attendance_metrics = analyzer.calculate_attendance_metrics()
    engagement_metrics = analyzer.calculate_engagement_score()
    attention_metrics = analyzer.calculate_attention_patterns()
    dynamics_metrics = analyzer.analyze_class_dynamics()
    
    all_metrics = {
        'attendance': attendance_metrics,
        'engagement': engagement_metrics,
        'attention': attention_metrics,
        'dynamics': dynamics_metrics
    }
    
    # === HEADER PRINCIPAL ===
    st.markdown("## 🎓 Dashboard de Análisis Educacional Avanzado")
    
    # === KPIs PRINCIPALES MEJORADOS ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="👥 Asistencia Promedio",
            value=f"{attendance_metrics['avg_students']:.1f}",
            delta=f"Consistencia: {attendance_metrics['attendance_consistency']:.1%}",
            help="Número promedio de estudiantes por clase y consistencia de asistencia"
        )
    
    with col2:
        engagement_color = "normal"
        if engagement_metrics['engagement_score'] > 75:
            engagement_color = "normal"
        elif engagement_metrics['engagement_score'] < 40:
            engagement_color = "inverse"
        
        st.metric(
            label="🚀 Score de Engagement",
            value=f"{engagement_metrics['engagement_score']:.1f}/100",
            delta=f"Bienestar: {engagement_metrics['emotional_wellness']:.1f}%",
            help="Puntuación basada en emociones positivas, curiosidad y participación"
        )
    
    with col3:
        st.metric(
            label="🎯 Nivel de Atención",
            value=f"{attention_metrics['avg_attention_score']:.1f}/100",
            delta=f"Tendencia: {attention_metrics['attention_trend']}",
            help="Basado en postura de estudiantes y nivel de movimiento"
        )
    
    with col4:
        st.metric(
            label="🗣️ Participación Activa",
            value=f"{dynamics_metrics['avg_participation_level']:.1f}%",
            delta=f"Tendencia: {dynamics_metrics['participation_trend']}",
            help="Porcentaje de estudiantes en participación activa (de pie/interactuando)"
        )
    
    with col5:
        st.metric(
            label="👨‍🏫 Ratio Maestro/Estudiante",
            value=f"1:{int(1/dynamics_metrics['avg_teacher_student_ratio']) if dynamics_metrics['avg_teacher_student_ratio'] > 0 else 'N/A'}",
            delta=f"Tipo: {dynamics_metrics['class_dynamics']}",
            help="Relación entre número de maestros y estudiantes"
        )
    
    st.markdown("---")
    
    # === GRÁFICOS PRINCIPALES MEJORADOS ===
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis Temporal", "🎭 Emociones", "👥 Participación", "📈 Tendencias"])
    
    with tab1:
        st.subheader("⏰ Evolución Temporal de Métricas Clave")
        
        # Preparar datos temporales
        times = [meta.get("start_time", 0) / 60 for meta in clips_meta]  # En minutos
        
        fig_temporal = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Asistencia por Tiempo', 
                'Engagement y Atención', 
                'Participación Activa',
                'Emociones Dominantes'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Datos para los gráficos
        students_data = [analyzer._get_total_students(meta) for meta in clips_meta]
        teachers_data = [analyzer._get_total_teachers(meta) for meta in clips_meta]
        attention_scores = attention_metrics['attention_scores']
        
        # Subplot 1: Asistencia
        fig_temporal.add_trace(
            go.Scatter(x=times, y=students_data, name="Estudiantes", 
                      line=dict(color='#3498db', width=3), mode='lines+markers'),
            row=1, col=1
        )
        fig_temporal.add_trace(
            go.Scatter(x=times, y=teachers_data, name="Maestros", 
                      line=dict(color='#e74c3c', width=2), mode='lines+markers'),
            row=1, col=1
        )
        
        # Subplot 2: Engagement vs Atención
        engagement_per_clip = []
        for meta in clips_meta:
            clip_emotions = meta.get("detections", {}).get("emotion_stats", {})
            pos = clip_emotions.get("happy", 0) + clip_emotions.get("surprise", 0)
            total = sum(clip_emotions.values()) if clip_emotions else 1
            engagement_per_clip.append((pos / total) * 100)
        
        fig_temporal.add_trace(
            go.Scatter(x=times, y=engagement_per_clip, name="Engagement (%)", 
                      line=dict(color='#2ecc71', width=3), mode='lines+markers'),
            row=1, col=2
        )
        fig_temporal.add_trace(
            go.Scatter(x=times, y=attention_scores, name="Atención (%)", 
                      line=dict(color='#f39c12', width=3), mode='lines+markers',
                      yaxis='y2'),
            row=1, col=2
        )
        
        # Subplot 3: Participación
        participation_data = []
        for meta in clips_meta:
            detections = meta.get("detections", {})
            total_students = analyzer._get_total_students(meta)
            standing = detections.get("students_standing", 0)
            participation = (standing / total_students * 100) if total_students > 0 else 0
            participation_data.append(participation)
        
        fig_temporal.add_trace(
            go.Scatter(x=times, y=participation_data, name="Participación Activa (%)", 
                      line=dict(color='#9b59b6', width=3), mode='lines+markers',
                      fill='tonexty'),
            row=2, col=1
        )
        
        # Subplot 4: Emociones dominantes - CORREGIDO
        dominant_emotions = []
        if dominant_emotions:
            emotion_counts = {}
            for emotion in dominant_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Crear gráfico de barras en lugar de scatter
            emotions_list = list(emotion_counts.keys())
            counts_list = list(emotion_counts.values())
            
            colors_map = {'happy': '#2ecc71', 'surprise': '#f39c12', 'neutral': '#95a5a6', 
                         'sad': '#3498db', 'fear': '#9b59b6', 'angry': '#e74c3c'}
            bar_colors = [colors_map.get(emotion, '#bdc3c7') for emotion in emotions_list]
            
            fig_temporal.add_trace(
                go.Bar(x=emotions_list, y=counts_list, name="Frecuencia Emociones", 
                       marker=dict(color=bar_colors), showlegend=False),
                row=2, col=2
            )
        
        # Actualizar layout con títulos de ejes apropiados
        fig_temporal.update_layout(
            height=700, 
            showlegend=True,
            title_text="Análisis Temporal Completo"
        )
        fig_temporal.update_xaxes(title_text="Tiempo (minutos)", row=1, col=1)
        fig_temporal.update_xaxes(title_text="Tiempo (minutos)", row=1, col=2)
        fig_temporal.update_xaxes(title_text="Tiempo (minutos)", row=2, col=1)
        fig_temporal.update_xaxes(title_text="Emociones", row=2, col=2)
        fig_temporal.update_yaxes(title_text="Número de Personas", row=1, col=1)
        fig_temporal.update_yaxes(title_text="Porcentaje (%)", row=1, col=2)
        fig_temporal.update_yaxes(title_text="Participación (%)", row=2, col=1)
        fig_temporal.update_yaxes(title_text="Frecuencia", row=2, col=2)
        
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with tab2:
        st.subheader("🎭 Análisis Emocional Detallado")
        
        if not emotion_summary or not any(emotion_summary.values()):
            st.info("❌ No hay datos de emociones disponibles para mostrar")
            return
        
        col_emotion1, col_emotion2 = st.columns([1.2, 0.8])
        
        with col_emotion1:
            # Gráfico de dona mejorado con mejor formato
            emotions = list(emotion_summary.keys())
            values = list(emotion_summary.values())
            
            # Colores consistentes
            emotion_colors = {
                'happy': '#2ecc71', 'sad': '#3498db', 'angry': '#e74c3c',
                'fear': '#9b59b6', 'surprise': '#f39c12', 'neutral': '#95a5a6'
            }
            colors = [emotion_colors.get(emotion, '#bdc3c7') for emotion in emotions]
            
            fig_emotions = go.Figure(data=[go.Pie(
                labels=[f"{emotion.capitalize()}" for emotion in emotions],
                values=values,
                hole=0.4,
                marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>',
                textfont=dict(size=12)
            )])
            
            fig_emotions.update_layout(
                title={
                    'text': "Distribución de Emociones Detectadas",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=450,
                showlegend=False,  # Evitar duplicar información
                margin=dict(t=50, b=50, l=50, r=50)
            )
            st.plotly_chart(fig_emotions, use_container_width=True)
        
        with col_emotion2:
            # Métricas emocionales con mejor formato
            st.markdown("### 📊 Métricas Emocionales")
            
            total_emotions = sum(emotion_summary.values())
            positive_emotions = emotion_summary.get('happy', 0) + emotion_summary.get('surprise', 0)
            negative_emotions = emotion_summary.get('sad', 0) + emotion_summary.get('angry', 0) + emotion_summary.get('fear', 0)
            neutral_emotions = emotion_summary.get('neutral', 0)
            
            # Métricas con mejor visualización
            st.metric(
                "😊 Positivas", 
                f"{positive_emotions:,}", 
                f"{positive_emotions/total_emotions:.1%}" if total_emotions > 0 else "0%"
            )
            st.metric(
                "😟 Negativas", 
                f"{negative_emotions:,}", 
                f"{negative_emotions/total_emotions:.1%}" if total_emotions > 0 else "0%"
            )
            st.metric(
                "😐 Neutrales", 
                f"{neutral_emotions:,}", 
                f"{neutral_emotions/total_emotions:.1%}" if total_emotions > 0 else "0%"
            )
            
            # Índice de bienestar mejorado
            if total_emotions > 0:
                wellness_index = (positive_emotions - negative_emotions) / total_emotions * 100
                wellness_color = "normal" if wellness_index >= 0 else "inverse"
                st.metric(
                    "💚 Índice Bienestar", 
                    f"{wellness_index:+.1f}%", 
                    "Balance emocional"
                )
                
                # Barra de progreso visual para bienestar
                wellness_normalized = max(0, min(100, wellness_index + 50))  # Normalizar de -50/+50 a 0/100
                st.progress(wellness_normalized/100)
    
    with tab3:
        st.subheader("👥 Análisis de Participación y Dinámica")
        
        # Verificar que tenemos datos
        if not clips_meta:
            st.warning("⚠️ No hay datos de clips disponibles")
            return
        
        # Tabla mejorada con mejor cálculo de métricas
        detailed_data = []
        for i, meta in enumerate(clips_meta, 1):
            detections = meta.get("detections", {})
            emotion_stats = detections.get("emotion_stats", {})
            
            students_sitting = detections.get("students_sitting", 0)
            students_standing = detections.get("students_standing", 0)
            teachers_sitting = detections.get("teachers_sitting", 0)
            teachers_standing = detections.get("teachers_standing", 0)
            
            total_students = students_sitting + students_standing
            total_teachers = teachers_sitting + teachers_standing
            
            # Evitar división por cero
            participation_rate = (students_standing / total_students * 100) if total_students > 0 else 0
            teacher_student_ratio = (1 / total_students * total_teachers) if total_students > 0 else 0
            
            # Engagement del clip - manejo seguro
            if emotion_stats and sum(emotion_stats.values()) > 0:
                positive_count = emotion_stats.get("happy", 0) + emotion_stats.get("surprise", 0)
                total_emotion_count = sum(emotion_stats.values())
                clip_engagement = (positive_count / total_emotion_count * 100)
                dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1])[0].title()
            else:
                clip_engagement = 0
                dominant_emotion = "N/A"
            
            # Score de atención mejorado
            motion_score = meta.get("motion_score", 0)
            if total_students > 0:
                sitting_ratio = students_sitting / total_students
                attention_score = sitting_ratio * max(0, 1 - min(motion_score * 5, 1)) * 100
            else:
                attention_score = 0
            
            row = {
                "Clip": i,
                "Tiempo": format_time(meta.get("start_time", 0)),
                "Estudiantes": total_students,
                "Maestros": total_teachers,
                "Participación": participation_rate,
                "Atención": attention_score,
                "Engagement": clip_engagement,
                "Actividad": motion_score,
                "Emoción": dominant_emotion
            }
            detailed_data.append(row)
        
        if not detailed_data:
            st.warning("⚠️ No se pudieron calcular métricas de participación")
            return
        
        df_detailed = pd.DataFrame(detailed_data)
        
        # Estadísticas clave con verificación de datos
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        participation_values = [row["Participación"] for row in detailed_data]
        attention_values = [row["Atención"] for row in detailed_data]
        engagement_values = [row["Engagement"] for row in detailed_data]
        
        with col_stats1:
            if participation_values and max(participation_values) > 0:
                best_participation_idx = participation_values.index(max(participation_values))
                st.metric("🏆 Mayor Participación", f"Clip {best_participation_idx + 1}", 
                         f"{max(participation_values):.1f}%")
            else:
                st.metric("🏆 Mayor Participación", "N/A", "0%")
        
        with col_stats2:
            if attention_values and max(attention_values) > 0:
                best_attention_idx = attention_values.index(max(attention_values))
                st.metric("🎯 Mayor Atención", f"Clip {best_attention_idx + 1}", 
                         f"{max(attention_values):.1f}%")
            else:
                st.metric("🎯 Mayor Atención", "N/A", "0%")
        
        with col_stats3:
            if engagement_values and max(engagement_values) > 0:
                best_engagement_idx = engagement_values.index(max(engagement_values))
                st.metric("🚀 Mayor Engagement", f"Clip {best_engagement_idx + 1}", 
                         f"{max(engagement_values):.1f}%")
            else:
                st.metric("🚀 Mayor Engagement", "N/A", "0%")
        
        with col_stats4:
            student_counts = [row["Estudiantes"] for row in detailed_data]
            avg_students = np.mean(student_counts) if student_counts else 0
            st.metric("📊 Promedio Estudiantes", f"{avg_students:.1f}", 
                     f"Total clips: {len(detailed_data)}")
        
        # Tabla con formato mejorado
        st.markdown("### 📋 Detalle por Clip")
        
        # Formatear DataFrame para mejor visualización
        df_display = df_detailed.copy()
        df_display["Participación"] = df_display["Participación"].apply(lambda x: f"{x:.1f}%")
        df_display["Atención"] = df_display["Atención"].apply(lambda x: f"{x:.1f}%") 
        df_display["Engagement"] = df_display["Engagement"].apply(lambda x: f"{x:.1f}%")
        df_display["Actividad"] = df_display["Actividad"].apply(lambda x: f"{x:.3f}")
        
        # Mostrar tabla sin el estilo problemático de gradiente
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    with tab4:
        st.subheader("📈 Análisis de Tendencias y Patrones")
        
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.markdown("### 📊 Tendencias Identificadas")
            
            # Mostrar tendencias calculadas
            st.write(f"**📈 Atención:** {attention_metrics['attention_trend']}")
            st.write(f"**🗣️ Participación:** {dynamics_metrics['participation_trend']}")
            
            # Análisis de picos
            if attention_metrics['attention_scores']:
                peak_time = attention_metrics['peak_attention_time']
                st.write(f"**🎯 Pico de Atención:** {peak_time}")
            
            # Análisis de primera vs segunda mitad
            if len(clips_meta) > 2:
                mid_point = len(clips_meta) // 2
                first_half_students = np.mean([analyzer._get_total_students(meta) for meta in clips_meta[:mid_point]])
                second_half_students = np.mean([analyzer._get_total_students(meta) for meta in clips_meta[mid_point:]])
                
                if second_half_students > first_half_students:
                    st.write("**📈 Asistencia:** Incremento durante la clase")
                elif second_half_students < first_half_students:
                    st.write("**📉 Asistencia:** Disminución durante la clase")
                else:
                    st.write("**📊 Asistencia:** Estable durante la clase")
        
        with col_trend2:
            st.markdown("### 🎯 Correlaciones")
            
            # Calcular correlaciones interesantes
            if len(clips_meta) > 3:
                students_counts = [analyzer._get_total_students(meta) for meta in clips_meta]
                motion_scores = [meta.get("motion_score", 0) for meta in clips_meta]
                
                # Correlación entre asistencia y movimiento
                if len(students_counts) == len(motion_scores):
                    correlation = np.corrcoef(students_counts, motion_scores)[0, 1]
                    if not np.isnan(correlation):
                        if correlation > 0.3:
                            st.write("**🔗 Asistencia-Actividad:** Correlación positiva alta")
                        st.write("  📊 Más estudiantes → Más actividad")
                    elif correlation < -0.3:
                        st.write("**🔗 Asistencia-Actividad:** Correlación negativa")
                        st.write("  📊 Más estudiantes → Menos movimiento")
                    else:
                        st.write("**🔗 Asistencia-Actividad:** Sin correlación clara")
                
                # Correlación entre engagement y atención
                if attention_metrics['attention_scores'] and len(engagement_per_clip) > 0:
                    att_eng_corr = np.corrcoef(attention_metrics['attention_scores'], engagement_per_clip)[0, 1]
                    if not np.isnan(att_eng_corr):
                        if att_eng_corr > 0.3:
                            st.write("**🎯 Atención-Engagement:** Correlación positiva")
                        elif att_eng_corr < -0.3:
                            st.write("**🎯 Atención-Engagement:** Correlación negativa")
                        else:
                            st.write("**🎯 Atención-Engagement:** Independientes")
            else:
                st.info("📊 Necesarios más datos para análisis de correlaciones")
    
    st.markdown("---")
    
    # === INSIGHTS Y RECOMENDACIONES AVANZADAS ===
    st.subheader("💡 Insights Inteligentes y Recomendaciones")
    
    # Generar insights usando el analizador
    insights = analyzer.generate_insights(all_metrics)
    
    col_insights1, col_insights2 = st.columns(2)
    
    with col_insights1:
        st.markdown("### 🔍 Análisis Automático")
        for insight in insights:
            st.write(f"• {insight}")
        
        # Análisis adicional de calidad
        avg_quality = np.mean([meta.get("quality_score", 0) for meta in clips_meta])
        if avg_quality < 50:
            st.write("• 📷 Calidad de video baja detectada. Revisar configuración de cámara.")
        elif avg_quality > 80:
            st.write("• ✅ Excelente calidad de video mantenida.")
    
    with col_insights2:
        st.markdown("### 🎯 Recomendaciones Personalizadas")
        
        recommendations = []
        
        # Recomendaciones basadas en engagement
        if engagement_metrics['engagement_score'] < 50:
            recommendations.append("🚀 Implementar actividades más dinámicas e interactivas")
            recommendations.append("🎪 Considerar gamificación en el contenido")
        
        # Recomendaciones basadas en atención
        if attention_metrics['avg_attention_score'] < 60:
            recommendations.append("⏰ Introducir descansos cada 15-20 minutos")
            recommendations.append("🎯 Variar el formato de enseñanza (visual, auditivo, kinestésico)")
        
        # Recomendaciones basadas en participación
        if dynamics_metrics['avg_participation_level'] < 30:
            recommendations.append("🗣️ Fomentar más preguntas y respuestas")
            recommendations.append("👥 Implementar trabajo en grupos pequeños")
        
        # Recomendaciones basadas en emociones
        if engagement_metrics['emotional_wellness'] < 65:
            recommendations.append("😊 Crear un ambiente más positivo y de apoyo")
            recommendations.append("🤝 Revisar la dinámica interpersonal en clase")
        
        # Recomendaciones basadas en tendencias
        if attention_metrics['attention_trend'] == 'Decreciente':
            recommendations.append("📉 Reestructurar el contenido para mantener interés")
            recommendations.append("🔄 Cambiar actividades más frecuentemente")
        
        if not recommendations:
            recommendations.append("✅ Continuar con las estrategias actuales - Resultados óptimos")
            recommendations.append("🎉 Considerar documentar mejores prácticas para replicar")
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # === RESUMEN EJECUTIVO ===
    st.markdown("---")
    st.subheader("📋 Resumen Ejecutivo")
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        st.markdown("#### 📊 Métricas Clave")
        st.write(f"**Asistencia Promedio:** {attendance_metrics['avg_students']:.1f} estudiantes")
        st.write(f"**Engagement Score:** {engagement_metrics['engagement_score']:.1f}/100")
        st.write(f"**Nivel de Atención:** {attention_metrics['avg_attention_score']:.1f}/100")
        st.write(f"**Participación Activa:** {dynamics_metrics['avg_participation_level']:.1f}%")
    
    with col_summary2:
        st.markdown("#### 🎯 Estado General")
        
        # Calcular estado general
        overall_score = np.mean([
            engagement_metrics['engagement_score'],
            attention_metrics['avg_attention_score'],
            dynamics_metrics['avg_participation_level'],
            engagement_metrics['emotional_wellness']
        ])
        
        if overall_score >= 75:
            status = "🟢 EXCELENTE"
            status_desc = "Clase funcionando de manera óptima"
        elif overall_score >= 60:
            status = "🟡 BUENO"
            status_desc = "Clase con buen rendimiento, algunas mejoras posibles"
        elif overall_score >= 40:
            status = "🟠 REGULAR"
            status_desc = "Clase necesita mejoras significativas"
        else:
            status = "🔴 REQUIERE ATENCIÓN"
            status_desc = "Clase necesita intervención inmediata"
        
        st.write(f"**Estado:** {status}")
        st.write(f"**Descripción:** {status_desc}")
        st.write(f"**Score Global:** {overall_score:.1f}/100")
    
    with col_summary3:
        st.markdown("#### 🚀 Próximos Pasos")
        
        # Determinar prioridades
        priorities = []
        
        if engagement_metrics['engagement_score'] < 50:
            priorities.append("1️⃣ Mejorar engagement estudiantil")
        
        if attention_metrics['avg_attention_score'] < 50:
            priorities.append("2️⃣ Aumentar nivel de atención")
        
        if dynamics_metrics['avg_participation_level'] < 30:
            priorities.append("3️⃣ Fomentar participación activa")
        
        if engagement_metrics['emotional_wellness'] < 60:
            priorities.append("4️⃣ Mejorar bienestar emocional")
        
        if not priorities:
            priorities.append("1️⃣ Mantener estándares actuales")
            priorities.append("2️⃣ Documentar mejores prácticas")
            priorities.append("3️⃣ Explorar optimizaciones")
        
        for priority in priorities[:3]:  # Mostrar solo las 3 principales
            st.write(priority)
    
    # === EXPORTAR DATOS (OPCIONAL) ===
    if st.button("📊 Generar Reporte Detallado"):
        # Crear un reporte resumido para download
        report_data = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_clips': len(clips_meta),
            'avg_students': attendance_metrics['avg_students'],
            'engagement_score': engagement_metrics['engagement_score'],
            'attention_score': attention_metrics['avg_attention_score'],
            'participation_level': dynamics_metrics['avg_participation_level'],
            'emotional_wellness': engagement_metrics['emotional_wellness'],
            'overall_score': overall_score,
            'status': status,
            'class_dynamics': dynamics_metrics['class_dynamics'],
            'peak_attention_time': attention_metrics['peak_attention_time'],
            'main_insights': insights[:3],  # Top 3 insights
            'top_recommendations': recommendations[:3]  # Top 3 recommendations
        }
        
        st.success("📋 Reporte generado exitosamente")
        st.json(report_data)
        
        # Opcional: convertir a DataFrame para descarga
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Reporte CSV",
            data=csv,
            file_name=f"reporte_educacional_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )