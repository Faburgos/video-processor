import numpy as np

def format_time(seconds):
    """Convierte segundos a formato mm:ss"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def safe_convert_metric_value(value):
    """Convierte valores numpy a tipos compatibles con st.metric()"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, str):
        return value
    elif value is None:
        return None
    else:
        return str(value)