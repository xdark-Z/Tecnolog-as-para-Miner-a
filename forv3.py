import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import ast
import math
import os
import plotly.express as px
import plotly.graph_objects as go
import heapq
import zipfile
from io import BytesIO
# ---- Orquestador de tareas (Redis) ----
import json
import time
import uuid
try:
    import redis
    REDIS_DISPONIBLE = True
except ImportError:
    REDIS_DISPONIBLE = False

st.set_page_config(page_title="Simulador de Tuneles", layout="wide")
st.title("Simulador de Avance de Tuneles - Monte Carlo")

SHEETS_OPERACION = {
    'Actividades': ['Nivel', 'Secuencia', 'Actividad', 'Recurso', 'Tiempo', 'Distribucion'],
    'Recursos': ['Nivel', 'Recurso', 'Cantidad', 'Cambio', 'Turno_Cambio', 'Cantidad_Cambio'],
    'Frentes': ['Nivel', 'Index', 'Frentes', 'Xi', 'Yi', 'Xf', 'Yf']
}
SHEETS_RESTRICCIONES = {
    'FrenteHundimiento': ['X', 'Y'],
    'ObrasCiviles': ['X', 'Y'],
    'Restriccion': ['Tipo', 'X', 'Y', 'Turno']
}
DEFAULT_METERBLAST = {
    '4.2x4.1': 3.2,
    '4.7x4.3': 3.2,
    '4.5x4.5': 3.2,
    '5.1x4.3': 3.2,
    '5.2x5.1': 3.2,
    '6.2x6.1': 2.3
}
DEFAULT_RESTRICTION_RADII = {
    'Polvorazo': 80.0,
    'Polvorazo B': 80.0,
    'PA FH': 40.0
}
VALID_DISTRIBUTIONS = {'Cte', 'fisk', 'normal', 'weibull', 'gamma', 'lognormal', 'kstwobign',
                       'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                       'burr', 'loglaplace', 'maxwell', 'nakagami'}

# ---- Parametros de cada distribucion, tal como los define scipy.stats ----
# Cada entrada describe los argumentos que scipy.stats.<dist>.pdf()/.rvs() usa
# realmente (nombre del parametro tal cual la libreria y su significado).
# Es la fuente unica para armar los formularios de la UI y el grafico de PDF.
DISTRIBUTION_PARAM_INFO = {
    'Cte': {
        'scipy_name': None,
        'parametros': [
            {'key': 'valor', 'label': 'Valor constante', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'normal': {
        'scipy_name': 'scipy.stats.norm',
        'parametros': [
            {'key': 'mean', 'label': 'mean (promedio, mu)', 'default': 1.0},
            {'key': 'std', 'label': 'std (desviacion estandar, sigma)', 'min': 0.0001, 'default': 0.1},
        ],
    },
    'weibull': {
        'scipy_name': 'scipy.stats.weibull_min',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma k)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala lambda)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'gamma': {
        'scipy_name': 'scipy.stats.gamma',
        'parametros': [
            {'key': 'a', 'label': 'a (shape, parametro de forma k)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (theta, escala; theta = 1/rate)', 'min': 0.0001, 'default': 0.5},
        ],
    },
    'lognormal': {
        'scipy_name': 'scipy.stats.lognorm',
        'parametros': [
            {'key': 's', 'label': 's (sigma, desv. estandar en escala log)', 'min': 0.0001, 'default': 0.5},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (= exp(mu), mu en escala log)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'fisk': {
        'scipy_name': 'scipy.stats.fisk',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma)', 'min': 0.0001, 'default': 6.26},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': -0.42},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 3.92},
        ],
    },
    'kstwobign': {
        'scipy_name': 'scipy.stats.kstwobign',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': -0.48},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 2.62},
        ],
    },
    'rayleigh': {
        'scipy_name': 'scipy.stats.rayleigh',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (sigma, parametro de escala/modo)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'foldcauchy': {
        'scipy_name': 'scipy.stats.foldcauchy',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, ubicacion de la Cauchy antes de plegar)', 'min': 0.0, 'default': 2.20},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'foldnorm': {
        'scipy_name': 'scipy.stats.foldnorm',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, promedio de la normal antes de plegar, en unidades de scale)', 'min': 0.0, 'default': 1.79},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (sigma, desviacion estandar antes de plegar)', 'min': 0.0001, 'default': 1.19},
        ],
    },
    'ncx2': {
        'scipy_name': 'scipy.stats.ncx2',
        'parametros': [
            {'key': 'df', 'label': 'df (grados de libertad)', 'min': 0.0001, 'default': 2.0},
            {'key': 'nc', 'label': 'nc (parametro de no-centralidad)', 'min': 0.0, 'default': 0.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'burr': {
        'scipy_name': 'scipy.stats.burr',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, primer parametro de forma)', 'min': 0.0001, 'default': 4.0},
            {'key': 'd', 'label': 'd (shape, segundo parametro de forma)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'loglaplace': {
        'scipy_name': 'scipy.stats.loglaplace',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'maxwell': {
        'scipy_name': 'scipy.stats.maxwell',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (a, parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'nakagami': {
        'scipy_name': 'scipy.stats.nakagami',
        'parametros': [
            {'key': 'nu', 'label': 'nu (shape, parametro de forma m)', 'min': 0.0001, 'default': 1.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala/spread)', 'min': 0.0001, 'default': 1.0},
        ],
    },
}

SHEETS_MARINA = {
    'Vaciaderos': ['Nivel', 'Capacidad', 'Maximo', 'Tipo']
}
SHEETS_MUCKPILE = {
    'Muckpile': ['Nivel', 'Capacidad Ocupada', 'Maximo', 'Tipo']
}

# ---- Configuración del orquestador de tareas (Redis) ----
# En Streamlit Cloud NO existe un Redis en "localhost": el servidor donde corre
# la app es efímero y compartido, así que la conexión debe apuntar a un Redis
# externo (por ejemplo, Upstash o Redis Cloud, que tienen planes gratuitos).
# Las credenciales se leen primero desde st.secrets (Settings -> Secrets en
# Streamlit Cloud) y, si no existen, desde variables de entorno (útil para
# correr la app en local). Si no se configura nada, la app sigue funcionando
# en "modo degradado" sin orquestación (tal como ya hacía el código original).
def _get_config(nombre, default=None):
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass
    return os.environ.get(nombre, default)

REDIS_HOST = _get_config("REDIS_HOST", "localhost")
REDIS_PORT = int(_get_config("REDIS_PORT", 6379))
REDIS_DB = int(_get_config("REDIS_DB", 0))
REDIS_PASSWORD = _get_config("REDIS_PASSWORD", None)
REDIS_SSL = str(_get_config("REDIS_SSL", "false")).strip().lower() in ("1", "true", "yes")
REDIS_LOCK_KEY = "orquestador:simulacion:lock"
REDIS_ESTADO_KEY = "orquestador:simulacion:estado"
REDIS_HISTORIAL_KEY = "orquestador:simulacion:historial"
REDIS_LOCK_TTL = int(_get_config("REDIS_LOCK_TTL", 1800))  # seg. TTL de seguridad del lock

@st.cache_resource(show_spinner=False)
def get_redis_client():
    """Conecta con Redis (orquestador de tareas). Si no está disponible, la app
    sigue funcionando sin orquestación (modo degradado) en vez de romperse.
    En Streamlit Cloud esto requiere un Redis externo configurado en Secrets
    (REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_SSL=true); si no se define,
    simplemente se corre sin orquestador."""
    if not REDIS_DISPONIBLE:
        return None
    try:
        cliente = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            password=REDIS_PASSWORD, ssl=REDIS_SSL,
            socket_connect_timeout=2, socket_timeout=2, decode_responses=True
        )
        cliente.ping()
        return cliente
    except Exception:
        return None

def adquirir_lock_simulacion(cliente, propietario, ttl=REDIS_LOCK_TTL):
    """Toma el lock de orquestación para evitar que dos simulaciones pesadas
    corran al mismo tiempo y dejen la app 'pegada'. El TTL libera el lock solo
    si el proceso muere sin avisar."""
    if cliente is None:
        return True  # sin Redis disponible no se bloquea la ejecución
    return bool(cliente.set(REDIS_LOCK_KEY, propietario, nx=True, ex=ttl))

def liberar_lock_simulacion(cliente, propietario):
    """Libera el lock solo si sigue siendo el dueño actual del proceso."""
    if cliente is None:
        return
    if cliente.get(REDIS_LOCK_KEY) == propietario:
        cliente.delete(REDIS_LOCK_KEY)

def actualizar_estado_orquestador(cliente, estado, progreso=0, detalle=""):
    """Publica el estado/progreso del proceso en Redis para poder consultarlo
    sin depender de que la sesión de Streamlit siga viva."""
    if cliente is None:
        return
    cliente.set(REDIS_ESTADO_KEY, json.dumps({
        "estado": estado, "progreso": progreso, "detalle": detalle, "timestamp": time.time()
    }))

def registrar_tarea_historial(cliente, id_proceso, evento, detalle=""):
    """Encola en Redis un registro de la tarea (inicio/fin) para trazabilidad
    del orquestador."""
    if cliente is None:
        return
    cliente.lpush(REDIS_HISTORIAL_KEY, json.dumps({
        "id_proceso": id_proceso, "evento": evento, "detalle": detalle, "timestamp": time.time()
    }))
    cliente.ltrim(REDIS_HISTORIAL_KEY, 0, 49)  # conserva solo las últimas 50 tareas

# Funciones auxiliares

def r2(valor):
    """Redondeo estándar del reporte: máximo 2 decimales.
    Si el valor absoluto es menor a 0.01 (se vería como 0.00 con 2 decimales),
    usa 2 cifras significativas en su lugar (ej. 0.0033445 -> 0.0033) para no
    perder información en valores muy pequeños. Vectorizado: acepta escalares,
    listas, arrays de numpy o Series/columnas de pandas.
    """
    def _r2_escalar(x):
        if x is None:
            return x
        try:
            if isinstance(x, (float, int, np.floating, np.integer)):
                if pd.isna(x):
                    return x
                x = float(x)
                if x == 0:
                    return 0.0
                if abs(x) < 0.01:
                    exponente = math.floor(math.log10(abs(x)))
                    decimales_sig = -exponente + 1  # 2 cifras significativas
                    return round(x, decimales_sig)
                return round(x, 2)
        except (TypeError, ValueError):
            pass
        return x

    if isinstance(valor, (pd.Series,)):
        return valor.apply(_r2_escalar)
    if isinstance(valor, np.ndarray):
        return np.array([_r2_escalar(v) for v in valor])
    if isinstance(valor, (list, tuple)):
        return [_r2_escalar(v) for v in valor]
    return _r2_escalar(valor)


def fmt2(valor):
    """Formatea un número como string usando la misma regla de r2(), para
    reemplazar los f-strings tipo :.2f/:.3f/:.4f/:.6f del reporte.
    Máximo 2 decimales para valores normales; para valores muy pequeños
    (< 0.01 en valor absoluto) conserva las cifras significativas que ya
    calculó r2() en vez de truncarlas a 0.00 (ej. 0.0033445 -> "0.0033")."""
    v = r2(valor)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if not isinstance(v, (float, int, np.floating, np.integer)):
        return str(v)
    v = float(v)
    if v != 0 and abs(v) < 0.01:
        texto = f"{v:.10f}".rstrip('0').rstrip('.')
        return texto if texto else "0"
    return f"{v:.2f}"


def percentiles_reporte(arr):
    """Calcula el set estándar de percentiles usados en el reporte:
    P0 (mínimo), P10, P30, P50, Esperanza (promedio), P70, P90, P100 (máximo)."""
    arr = np.asarray(arr, dtype=float)
    return {
        'p0': r2(np.min(arr)),
        'p10': r2(np.percentile(arr, 10)),
        'p30': r2(np.percentile(arr, 30)),
        'p50': r2(np.percentile(arr, 50)),
        'esperanza': r2(np.mean(arr)),
        'p70': r2(np.percentile(arr, 70)),
        'p90': r2(np.percentile(arr, 90)),
        'p100': r2(np.max(arr)),
    }


def graficar_convergencia_montecarlo(valores, titulo="Convergencia Monte Carlo",
                                      n_checkpoints=60, n_repeticiones=15):
    """
    Grafico de convergencia: muestra como se estabiliza el promedio acumulado del
    AVANCE FINAL de un frente a medida que se van "sumando" simulaciones (1, 2, 3...
    hasta N). Se repite el calculo con distintos barajados (bootstrap del orden) para
    dejar claro que la estabilizacion no depende del orden en que llegaron los datos,
    sino de la cantidad de simulaciones acumuladas.
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    checkpoints = np.unique(np.linspace(5, n, min(n_checkpoints, n)).astype(int))

    fig = go.Figure()
    for _ in range(n_repeticiones):
        muestra = np.random.permutation(valores)
        medias_acumuladas = [np.mean(muestra[:k]) for k in checkpoints]
        fig.add_trace(go.Scatter(
            x=checkpoints, y=medias_acumuladas, mode='lines',
            line=dict(width=1, color='rgba(31,119,180,0.35)'),
            showlegend=False, hoverinfo='skip'
        ))

    promedio_final = np.mean(valores)
    fig.add_hline(
        y=promedio_final, line_dash="dash", line_color="red",
        annotation_text=f"Promedio con N={n}: {promedio_final:.2f} m"
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Número de simulaciones acumuladas",
        yaxis_title="Avance Final Promedio Acumulado (m)",
        height=450
    )
    return fig


def graficar_convergencia_percentiles(valores, percentiles=(10, 50, 90),
                                       titulo="Convergencia de Percentiles",
                                       n_checkpoints=60):
    """
    Complementa el grafico de convergencia del promedio: muestra como se estabilizan
    los percentiles P10/P50/P90 (los que realmente se usan en el reporte) a medida
    que aumenta el numero de simulaciones acumuladas.
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    checkpoints = np.unique(np.linspace(10, n, min(n_checkpoints, n)).astype(int))

    fig = go.Figure()
    colores = {10: 'rgb(255,127,14)', 50: 'rgb(31,119,180)', 90: 'rgb(44,160,44)'}
    for p in percentiles:
        valores_pct = [np.percentile(valores[:k], p) for k in checkpoints]
        fig.add_trace(go.Scatter(
            x=checkpoints, y=valores_pct, mode='lines+markers',
            name=f'P{p}', line=dict(width=2, color=colores.get(p, None))
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Número de simulaciones acumuladas",
        yaxis_title="Avance Final (m)",
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig


def calcular_error_relativo_convergencia(valores):
    """
    Calcula el error estandar relativo de la media con el N total disponible.
    Sirve como criterio numerico simple para decidir si el N de simulaciones ya
    es suficiente (regla practica: < 2% se considera bien convergido).
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    media = np.mean(valores)
    error_estandar = np.std(valores, ddof=1) / np.sqrt(n)
    error_relativo_pct = (error_estandar / media) * 100 if media != 0 else 0.0
    return media, error_estandar, error_relativo_pct


def parse_tiempo(tiempo_str):
    """Parsea el string de tiempo a dict o float"""
    if isinstance(tiempo_str, (int, float)):
        return tiempo_str
    try:
        return ast.literal_eval(tiempo_str)
    except:
        return tiempo_str

def validar_hojas_y_columnas(excel_file, required_schema, nombre_archivo):
    """Valida hojas y columnas minimas antes de ejecutar la simulacion."""
    errors = []
    warnings = []
    xls = pd.ExcelFile(excel_file)
    missing_sheets = [sheet for sheet in required_schema if sheet not in xls.sheet_names]
    if missing_sheets:
        errors.append(f"{nombre_archivo}: faltan hojas {', '.join(missing_sheets)}.")
        return {}, errors, warnings

    data = {}
    for sheet, required_cols in required_schema.items():
        df = xls.parse(sheet)
        data[sheet] = df
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"{nombre_archivo}/{sheet}: faltan columnas {', '.join(missing_cols)}.")

    return data, errors, warnings

def validar_datos_operacion(df_actividades, df_recursos, df_frentes):
    errors = []
    warnings = []

    if df_actividades.empty:
        errors.append("La hoja Actividades esta vacia.")
    if df_recursos.empty:
        errors.append("La hoja Recursos esta vacia.")
    if df_frentes.empty:
        errors.append("La hoja Frentes esta vacia.")

    if 'Distribucion' in df_actividades.columns:
        invalid_dist = sorted(set(df_actividades['Distribucion'].dropna()) - VALID_DISTRIBUTIONS)
        if invalid_dist:
            warnings.append(
                "Distribuciones no reconocidas: "
                + ", ".join(map(str, invalid_dist))
                + ". Se simularan como duracion 1.0 h si no se corrigen."
            )

    for col in ['Xi', 'Yi', 'Xf', 'Yf']:
        if col in df_frentes.columns and df_frentes[col].isna().any():
            errors.append(f"La hoja Frentes tiene coordenadas vacias en la columna {col}.")

    return errors, warnings

def cargar_y_validar_inputs(archivo_main, archivo_restricciones):
    data_main, errors_main, warnings_main = validar_hojas_y_columnas(
        archivo_main, SHEETS_OPERACION, "Archivo 1 Operacion"
    )
    data_rest, errors_rest, warnings_rest = validar_hojas_y_columnas(
        archivo_restricciones, SHEETS_RESTRICCIONES, "Archivo 2 Restricciones"
    )

    errors = errors_main + errors_rest
    warnings = warnings_main + warnings_rest
    if errors:
        return None, errors, warnings

    df_actividades = data_main['Actividades'].copy()
    df_recursos = data_main['Recursos'].copy()
    df_frentes = data_main['Frentes'].copy()
    df_actividades['Tiempo'] = df_actividades['Tiempo'].apply(parse_tiempo)

    data_errors, data_warnings = validar_datos_operacion(df_actividades, df_recursos, df_frentes)
    errors.extend(data_errors)
    warnings.extend(data_warnings)

    return {
        'df_actividades': df_actividades,
        'df_recursos': df_recursos,
        'df_frentes': df_frentes,
        'df_fh': data_rest['FrenteHundimiento'].copy(),
        'df_oc': data_rest['ObrasCiviles'].copy(),
        'df_res': data_rest['Restriccion'].copy()
    }, errors, warnings

def restriction_radius(tipo, default_radius, radios_por_tipo=None):
    if radios_por_tipo is None:
        radios_por_tipo = DEFAULT_RESTRICTION_RADII
    if pd.isna(tipo):
        return float(default_radius)
    return float(radios_por_tipo.get(str(tipo), default_radius))

def columna_seccion(df_frentes):
    for candidate in ['Sección', 'Seccion', 'SecciÃ³n']:
        if candidate in df_frentes.columns:
            return candidate
    return None

# ---- Filtrado por Nivel (cacheado) ----
# frentes_nivel y actividades_nivel son las tablas base que casi todo el resto de la app
# lee (Menús 3 a 12). Sin cache, Streamlit repetía este filtrado del DataFrame completo
# en CADA rerun -aunque el Nivel seleccionado no cambiara- por cualquier interacción en
# cualquier parte de la app. Los DataFrames grandes se reciben con guion bajo
# (_df_actividades, _df_recursos, _df_frentes) para que Streamlit no intente hashearlos
# completos en cada llamada; la validez del cache depende solo de 'nivel_seleccionado' y
# de 'data_version' (que únicamente cambia cuando se cargan archivos nuevos en el Menú 1).
@st.cache_data(show_spinner=False)
def filtrar_actividades_nivel(_df_actividades, nivel_seleccionado, data_version):
    """Filtra la hoja Actividades por Nivel."""
    return _df_actividades[_df_actividades['Nivel'] == nivel_seleccionado].copy()


@st.cache_data(show_spinner=False)
def filtrar_recursos_nivel(_df_recursos, nivel_seleccionado, data_version):
    """Filtra la hoja Recursos por Nivel."""
    return _df_recursos[_df_recursos['Nivel'] == nivel_seleccionado].copy()


@st.cache_data(show_spinner=False)
def filtrar_frentes_nivel(_df_frentes, nivel_seleccionado, data_version):
    """Filtra la hoja Frentes por Nivel."""
    return _df_frentes[_df_frentes['Nivel'] == nivel_seleccionado].copy()


def distancia_minima_a_puntos(df_puntos, x, y):
    if df_puntos is None or df_puntos.empty or not {'X', 'Y'}.issubset(df_puntos.columns):
        return float('inf')
    return min(calcular_distancia(x, y, row['X'], row['Y']) for _, row in df_puntos.iterrows())

def ordenar_frentes_por_prioridad(frentes_df, frentes_seleccionados, df_fh, df_oc, ruta_critica):
    """Replica la intencion del motor original: ruta critica primero, luego cercania FH/OOCC."""
    prioridad = []
    ruta_critica = set(ruta_critica or [])
    for frente in frentes_seleccionados:
        row = frentes_df[frentes_df['Frentes'] == frente].iloc[0]
        critical_rank = 0 if frente in ruta_critica else 1
        fh_distance = distancia_minima_a_puntos(df_fh, row['Xi'], row['Yi'])
        oc_distance = distancia_minima_a_puntos(df_oc, row['Xi'], row['Yi'])
        prioridad.append((critical_rank, fh_distance, oc_distance, frente))
    return [item[-1] for item in sorted(prioridad)]

def avance_efectivo_frente(frente, metros_avance_default, frentes_info, mapa_restriccion_geologica=None):
    metros = frentes_info.get(frente, {}).get('metros_por_ciclo', metros_avance_default)
    if mapa_restriccion_geologica and frente in mapa_restriccion_geologica:
        castigo_pct = mapa_restriccion_geologica[frente].get('castigo_avance', 0)
        metros *= 1 + (castigo_pct / 100.0)
    return max(0.0, float(metros))

DIST_OBJ_SCIPY = {
    'fisk': stats.fisk, 'normal': stats.norm, 'weibull': stats.weibull_min,
    'gamma': stats.gamma, 'lognormal': stats.lognorm, 'kstwobign': stats.kstwobign,
    'rayleigh': stats.rayleigh, 'foldcauchy': stats.foldcauchy, 'foldnorm': stats.foldnorm,
    'ncx2': stats.ncx2, 'burr': stats.burr, 'loglaplace': stats.loglaplace,
    'maxwell': stats.maxwell, 'nakagami': stats.nakagami,
}


def _kwargs_scipy_desde_params(dist_params, distribucion):
    """Traduce el dict de parametros (nombres de la UI) a los kwargs que espera
    cada funcion de scipy.stats, sin volver a escribir el mapeo en cada lugar."""
    if distribucion == 'normal':
        return {'loc': dist_params['mean'], 'scale': dist_params['std']}
    elif distribucion == 'fisk':
        return {'c': dist_params['c'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'weibull':
        return {'c': dist_params['c'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'gamma':
        return {'a': dist_params['a'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'lognormal':
        return {'s': dist_params['s'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'kstwobign':
        return {'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'rayleigh':
        return {'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'foldcauchy':
        return {'c': dist_params.get('c', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'foldnorm':
        return {'c': dist_params.get('c', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'ncx2':
        return {'df': dist_params.get('df', 2), 'nc': dist_params.get('nc', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'burr':
        return {'c': dist_params['c'], 'd': dist_params['d'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'loglaplace':
        return {'c': dist_params['c'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'maxwell':
        return {'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'nakagami':
        return {'nu': dist_params['nu'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    return {}


def generar_tiempos_batch(dist_params, distribucion, n_samples):
    """Genera múltiples tiempos aleatorios de una vez.

    Si dist_params trae las claves '_trunc_lo'/'_trunc_hi' (filtro de rango aplicado
    como TRUNCAMIENTO real, no solo como indicador visual), los valores generados
    quedan garantizados dentro de [_trunc_lo, _trunc_hi]: se invierte la CDF de la
    distribucion base restringida a ese tramo (muestreo uniforme de u en
    [cdf(lo), cdf(hi)] y ppf(u)), que es exacto y no depende de rechazo/reintentos."""
    trunc_lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
    trunc_hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None

    if distribucion == 'Cte' or isinstance(dist_params, (int, float)):
        valor = float(dist_params) if isinstance(dist_params, (int, float)) else 1.0
        return np.full(n_samples, valor)

    try:
        if trunc_lo is not None and trunc_hi is not None and distribucion in DIST_OBJ_SCIPY:
            dist_obj = DIST_OBJ_SCIPY[distribucion]
            kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
            cdf_lo = dist_obj.cdf(trunc_lo, **kwargs)
            cdf_hi = dist_obj.cdf(trunc_hi, **kwargs)
            u = np.random.uniform(cdf_lo, cdf_hi, size=n_samples)
            return dist_obj.ppf(u, **kwargs)

        if distribucion == 'fisk':
            return stats.fisk.rvs(c=dist_params['c'], loc=dist_params['loc'], 
                                 scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'normal':
            return stats.norm.rvs(loc=dist_params['mean'], scale=dist_params['std'], size=n_samples)
        elif distribucion == 'weibull':
            return stats.weibull_min.rvs(c=dist_params['c'], loc=dist_params['loc'], 
                                         scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'gamma':
            return stats.gamma.rvs(a=dist_params['a'], loc=dist_params['loc'], 
                                  scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'lognormal':
            return stats.lognorm.rvs(s=dist_params['s'], loc=dist_params['loc'], 
                                    scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'kstwobign':
            return stats.kstwobign.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'rayleigh':
            return stats.rayleigh.rvs(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'foldcauchy':
            return stats.foldcauchy.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'foldnorm':
            return stats.foldnorm.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'ncx2':
            return stats.ncx2.rvs(df=dist_params.get('df', 2), nc=dist_params.get('nc', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'burr':
            return stats.burr.rvs(c=dist_params['c'], d=dist_params['d'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'loglaplace':
            return stats.loglaplace.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'maxwell':
            return stats.maxwell.rvs(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'nakagami':
            return stats.nakagami.rvs(nu=dist_params['nu'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        else:
            return np.ones(n_samples)
    except Exception as e:
        return np.ones(n_samples)

def calcular_pdf_teorica(dist_params, distribucion, x_vals):
    """Calcula la densidad de probabilidad (PDF) analitica exacta de la distribucion
    de entrada evaluada sobre x_vals. El area bajo esta curva integra a 1 por
    definicion (a diferencia de un histograma, que depende del binning).

    Si dist_params trae '_trunc_lo'/'_trunc_hi', devuelve la PDF TRUNCADA y
    RENORMALIZADA (0 fuera del rango, y dentro del rango la pdf original dividida
    por el area que queda entre esos limites, para que vuelva a integrar a 1)."""
    trunc_lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
    trunc_hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None
    try:
        if distribucion == 'Cte' or isinstance(dist_params, (int, float)):
            return np.zeros_like(x_vals)
        elif distribucion == 'fisk':
            pdf_vals = stats.fisk.pdf(x_vals, c=dist_params['c'], loc=dist_params['loc'],
                                  scale=dist_params['scale'])
        elif distribucion == 'normal':
            pdf_vals = stats.norm.pdf(x_vals, loc=dist_params['mean'], scale=dist_params['std'])
        elif distribucion == 'weibull':
            pdf_vals = stats.weibull_min.pdf(x_vals, c=dist_params['c'], loc=dist_params['loc'],
                                         scale=dist_params['scale'])
        elif distribucion == 'gamma':
            pdf_vals = stats.gamma.pdf(x_vals, a=dist_params['a'], loc=dist_params['loc'],
                                   scale=dist_params['scale'])
        elif distribucion == 'lognormal':
            pdf_vals = stats.lognorm.pdf(x_vals, s=dist_params['s'], loc=dist_params['loc'],
                                     scale=dist_params['scale'])
        elif distribucion == 'kstwobign':
            pdf_vals = stats.kstwobign.pdf(x_vals, loc=dist_params['loc'], scale=dist_params['scale'])
        elif distribucion == 'rayleigh':
            pdf_vals = stats.rayleigh.pdf(x_vals, loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'foldcauchy':
            pdf_vals = stats.foldcauchy.pdf(x_vals, c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                        scale=dist_params.get('scale', 1))
        elif distribucion == 'foldnorm':
            pdf_vals = stats.foldnorm.pdf(x_vals, c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                      scale=dist_params.get('scale', 1))
        elif distribucion == 'ncx2':
            pdf_vals = stats.ncx2.pdf(x_vals, df=dist_params.get('df', 2), nc=dist_params.get('nc', 0),
                                  loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'burr':
            pdf_vals = stats.burr.pdf(x_vals, c=dist_params['c'], d=dist_params['d'],
                                  loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'loglaplace':
            pdf_vals = stats.loglaplace.pdf(x_vals, c=dist_params['c'], loc=dist_params.get('loc', 0),
                                        scale=dist_params.get('scale', 1))
        elif distribucion == 'maxwell':
            pdf_vals = stats.maxwell.pdf(x_vals, loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'nakagami':
            pdf_vals = stats.nakagami.pdf(x_vals, nu=dist_params['nu'], loc=dist_params.get('loc', 0),
                                      scale=dist_params.get('scale', 1))
        else:
            return np.zeros_like(x_vals)

        if trunc_lo is not None and trunc_hi is not None and distribucion in DIST_OBJ_SCIPY:
            dist_obj = DIST_OBJ_SCIPY[distribucion]
            kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
            area_tramo = dist_obj.cdf(trunc_hi, **kwargs) - dist_obj.cdf(trunc_lo, **kwargs)
            fuera_rango = (x_vals < trunc_lo) | (x_vals > trunc_hi)
            pdf_vals = np.where(fuera_rango, 0.0, pdf_vals / area_tramo if area_tramo > 0 else 0.0)

        return pdf_vals
    except Exception:
        return np.zeros_like(x_vals)

def generar_velocidad_batch(dist_params, distribucion, n_samples):
    """Genera múltiples velocidades aleatorias de una vez"""
    if distribucion == 'Constante' or isinstance(dist_params, (int, float)):
        valor = float(dist_params) if isinstance(dist_params, (int, float)) else 15.0
        return np.full(n_samples, valor)

    try:
        if distribucion == 'normal':
            return np.abs(stats.norm.rvs(loc=dist_params['mean'], scale=dist_params['std'], size=n_samples))
        elif distribucion == 'weibull':
            return np.abs(stats.weibull_min.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                                 scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'gamma':
            return np.abs(stats.gamma.rvs(a=dist_params['a'], loc=dist_params.get('loc', 0),
                                          scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'lognormal':
            return np.abs(stats.lognorm.rvs(s=dist_params['s'], loc=dist_params.get('loc', 0),
                                            scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'fisk':
            return np.abs(stats.fisk.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                         scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'kstwobign':
            return np.abs(stats.kstwobign.rvs(loc=dist_params.get('loc', 0),
                                              scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'rayleigh':
            return stats.rayleigh.rvs(loc=dist_params.get('loc', 0),
                                      scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'foldcauchy':
            return stats.foldcauchy.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                        scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'foldnorm':
            return stats.foldnorm.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                      scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'ncx2':
            return np.abs(stats.ncx2.rvs(df=dist_params.get('df', 2), nc=dist_params.get('nc', 0),
                                         loc=dist_params.get('loc', 0), scale=dist_params['scale'],
                                         size=n_samples))
        elif distribucion == 'burr':
            return np.abs(stats.burr.rvs(c=dist_params['c'], d=dist_params['d'], loc=dist_params.get('loc', 0),
                                         scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'loglaplace':
            return np.abs(stats.loglaplace.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                               scale=dist_params['scale'], size=n_samples))
        elif distribucion == 'maxwell':
            return stats.maxwell.rvs(loc=dist_params.get('loc', 0), scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'nakagami':
            return np.abs(stats.nakagami.rvs(nu=dist_params['nu'], loc=dist_params.get('loc', 0),
                                             scale=dist_params['scale'], size=n_samples))
        else:
            return np.full(n_samples, 15.0)
    except Exception:
        return np.full(n_samples, 15.0)

def calcular_distancia(xi, yi, xf, yf):
    """Calcula distancia euclidiana"""
    return math.sqrt((xf - xi)**2 + (yf - yi)**2)

def calcular_posicion_avanzada(xi, yi, xf, yf, metros_avanzados, distancia_total):
    """Calcula la nueva posición después de avanzar ciertos metros en el túnel"""
    if distancia_total == 0:
        return xi, yi
    
    proporcion = metros_avanzados / distancia_total
    nuevo_x = xi + (xf - xi) * proporcion
    nuevo_y = yi + (yf - yi) * proporcion
    
    return nuevo_x, nuevo_y

def check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radius):
    """
    Verifica si el segmento de línea (túnel) [Xi, Yi] a [Xf, Yf]
    interseca la circunferencia centrada en [Px, Py] con radio 'radius'.
    """
    
    dist_centro_a_Xi = calcular_distancia(Px, Py, Xi, Yi)
    dist_centro_a_Xf = calcular_distancia(Px, Py, Xf, Yf)
    if dist_centro_a_Xi <= radius or dist_centro_a_Xf <= radius:
        return True
    
    dx = Xf - Xi
    dy = Yf - Yi
    t_den = dx*dx + dy*dy

    if t_den == 0:
        return dist_centro_a_Xi <= radius
    
    t_num = (Px - Xi) * dx + (Py - Yi) * dy
    t = t_num / t_den
    
    if t < 0.0:
        closest_x, closest_y = Xi, Yi
    elif t > 1.0:
        closest_x, closest_y = Xf, Yf
    else:
        closest_x = Xi + t * dx
        closest_y = Yi + t * dy

    distance_to_segment = calcular_distancia(Px, Py, closest_x, closest_y)
    
    return distance_to_segment <= radius

def calcular_demoras_restricciones(df_frentes, df_fh, df_oc, df_res, radio_restriccion, demora_horas, radios_por_tipo=None):
    """
    Calcula las demoras totales (FH + OC) que se aplican a cada túnel por ciclo,
    y las demoras específicas por turno (R).
    """
    
    dict_demoras_constantes = {frente: 0 for frente in df_frentes['Frentes'].unique()}
    list_demoras_por_turno = []
    
    for frente_nombre in df_frentes['Frentes'].unique():
        frente_data = df_frentes[df_frentes['Frentes'] == frente_nombre].iloc[0]
        Xi, Yi, Xf, Yf = frente_data['Xi'], frente_data['Yi'], frente_data['Xf'], frente_data['Yf']
        
        demora_acumulada_constante = 0
        
        # 1. Check FH (Frente Hundimiento)
        impacto_fh = False
        if 'X' in df_fh.columns and 'Y' in df_fh.columns:
            for _, row in df_fh.iterrows():
                Px, Py = row['X'], row['Y']
                if check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_restriccion):
                    impacto_fh = True
                    break
        if impacto_fh:
            demora_acumulada_constante += demora_horas
            
        # 2. Check OC (Obras Civiles)
        impacto_oc = False
        if 'X' in df_oc.columns and 'Y' in df_oc.columns:
            for _, row in df_oc.iterrows():
                Px, Py = row['X'], row['Y']
                if check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_restriccion):
                    impacto_oc = True
                    break
        if impacto_oc:
            demora_acumulada_constante += demora_horas

        dict_demoras_constantes[frente_nombre] = demora_acumulada_constante

    # 3. Procesar Restricción (R) - Demora Dependiente del Turno
    if 'X' in df_res.columns and 'Y' in df_res.columns and 'Turno' in df_res.columns and 'Tipo' in df_res.columns:
        for frente_nombre in df_frentes['Frentes'].unique():
            frente_data = df_frentes[df_frentes['Frentes'] == frente_nombre].iloc[0]
            Xi, Yi, Xf, Yf = frente_data['Xi'], frente_data['Yi'], frente_data['Xf'], frente_data['Yf']

            for _, row in df_res.iterrows():
                Px, Py = row['X'], row['Y']
                turno = int(row['Turno']) 
                tipo = f"Demora por Restricción ({row['Tipo']})"
                radio_turno = restriction_radius(row.get('Tipo'), radio_restriccion, radios_por_tipo)
                
                if check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_turno):
                    list_demoras_por_turno.append({
                        'Frente': frente_nombre,
                        'Turno': turno,
                        'Demora': demora_horas,
                        'Tipo': tipo,
                        'Radio': radio_turno,
                        'X': Px,
                        'Y': Py
                    })
    
    return dict_demoras_constantes, list_demoras_por_turno

def calcular_ventanas_flota(recurso, cantidad_base, plan_recurso, duracion_turno):
    """
    A partir de la cantidad base de un recurso y su lista de cambios de plan
    (turno, cantidad_nueva, demora_horas), calcula las ventanas de actividad
    [activa_desde, activa_hasta) de cada instancia (unidad de equipo/cuadrilla).

    - Un aumento de cantidad crea instancias nuevas que se activan en
      'inicio_turno + demora_horas'.
    - Una disminucion de cantidad desactiva las instancias de mayor id en
      'inicio_turno + demora_horas'.
    - Los cambios se aplican en orden cronologico de turno; si hay varios
      cambios para el mismo recurso en el mismo turno, se aplica el ultimo
      definido en el plan.
    """
    eventos = sorted(
        [c for c in plan_recurso if c['recurso'] == recurso],
        key=lambda c: c['turno']
    )
    # Si hay duplicados de turno, se queda con el ultimo definido (orden de insercion)
    por_turno = {}
    for ev in eventos:
        por_turno[ev['turno']] = ev
    eventos_ordenados = [por_turno[t] for t in sorted(por_turno.keys())]

    cantidad_actual = int(cantidad_base)
    ventanas = [{'id': i, 'activa_desde': 0.0, 'activa_hasta': float('inf')} for i in range(cantidad_actual)]
    registro_cambios = []  # para trazabilidad en resultados / Gantt

    for ev in eventos_ordenados:
        tiempo_evento = max(0, (ev['turno'] - 1)) * duracion_turno + ev.get('demora_horas', 0.0)
        cantidad_nueva = int(ev['cantidad_nueva'])

        if cantidad_nueva > cantidad_actual:
            # Agregar instancias nuevas, activas desde tiempo_evento
            for i in range(cantidad_actual, cantidad_nueva):
                ventanas.append({'id': i, 'activa_desde': tiempo_evento, 'activa_hasta': float('inf')})
            registro_cambios.append({
                'recurso': recurso, 'turno': ev['turno'], 'tipo': 'aumento',
                'cantidad_anterior': cantidad_actual, 'cantidad_nueva': cantidad_nueva,
                'demora_horas': ev.get('demora_horas', 0.0), 'tiempo_efectivo': tiempo_evento
            })
        elif cantidad_nueva < cantidad_actual:
            # Cerrar la ventana de las instancias de mayor id (las ultimas agregadas)
            ids_a_cerrar = sorted([v['id'] for v in ventanas if v['activa_hasta'] == float('inf')], reverse=True)[:cantidad_actual - cantidad_nueva]
            for v in ventanas:
                if v['id'] in ids_a_cerrar:
                    v['activa_hasta'] = tiempo_evento
            registro_cambios.append({
                'recurso': recurso, 'turno': ev['turno'], 'tipo': 'disminucion',
                'cantidad_anterior': cantidad_actual, 'cantidad_nueva': cantidad_nueva,
                'demora_horas': ev.get('demora_horas', 0.0), 'tiempo_efectivo': tiempo_evento
            })
        cantidad_actual = cantidad_nueva

    return ventanas, registro_cambios


def simular_avance_con_transporte(actividades_nivel, frentes_info, recursos_config, 
                                   tiempo_limite, metros_avance, n_simulaciones, 
                                   velocidades_config, demoras_constantes, demoras_por_turno, 
                                   sistema_turnos, restricciones_geologicas, fallas_equipos,
                                   plan_recursos_turno=None, progress_callback=None):
    """
    Simulación con gestión de eventos discretos: cada actividad libera su recurso inmediatamente.
    """
    if plan_recursos_turno is None:
        plan_recursos_turno = []
    
    # Determinar duración del turno
    if '12x2' in sistema_turnos:
        duracion_turno = 12.0
    elif '8x3' in sistema_turnos:
        duracion_turno = 8.0
    else:
        duracion_turno = 24.0

    n_frentes = len(frentes_info)
    frentes_nombres = list(frentes_info.keys())
    
    # Pre-procesar restricciones geológicas para búsqueda rápida
    mapa_restriccion_geologica = {}
    for frente in frentes_nombres:
        for res_geo in restricciones_geologicas:
            if frente in res_geo.get('frentes_aplicables', []):
                mapa_restriccion_geologica[frente] = res_geo
                break  # Aplicar solo la primera restricción encontrada

    # Pre-procesar fallas de equipos para búsqueda rápida
    mapa_fallas_equipos = {}
    for falla_config in fallas_equipos:
        for frente in falla_config.get('frentes_aplicables', []):
            mapa_fallas_equipos[frente] = falla_config

    def metros_ciclo(frente_nombre):
        return avance_efectivo_frente(frente_nombre, metros_avance, frentes_info, mapa_restriccion_geologica)

    resultados = {frente: [] for frente in frentes_nombres}
    traza_eventos = []
    # Avance REAL acumulado por turno, capturado en vivo desde avances_frentes durante la
    # simulacion (misma fuente que "resultados"). Reemplaza el calculo posterior por conteo
    # de ciclos en la traza de eventos, que podia mostrar avance donde el Avance Final real
    # quedo en 0 (ciclos truncados por tiempo_limite o por bloqueo de recurso).
    n_turnos_total = math.ceil(tiempo_limite / duracion_turno)
    avance_por_turno_resultado = {
        frente: {turno: [] for turno in range(1, n_turnos_total + 1)} for frente in frentes_nombres
    }
    
    # Pre-calcular velocidades
    num_precalc = max(n_simulaciones * n_frentes * 100, 50000)
    velocidades_precalculadas = {}
    for recurso, vel_config in velocidades_config.items():
        velocidades_km_h = generar_velocidad_batch(vel_config['params'], vel_config['dist'], num_precalc)
        velocidades_precalculadas[recurso] = velocidades_km_h

    actividades_filtradas = actividades_nivel[actividades_nivel['Actividad'].notna()].copy()
    recursos_necesarios = list(actividades_filtradas['Recurso'].dropna().unique())
    
    # Obtener secuencias ordenadas
    secuencias_ordenadas = sorted(actividades_filtradas['Secuencia'].unique())

    # Pre-calcular ventanas de actividad (activa_desde/activa_hasta) por recurso segun
    # el plan de orquestacion de flota por turno. Es igual para todas las simulaciones
    # (determinístico), asi que se calcula una sola vez fuera del loop de Monte Carlo.
    ventanas_por_recurso = {}
    registro_cambios_flota = []
    for recurso in recursos_necesarios:
        if recurso in recursos_config:
            cantidad_base = int(recursos_config[recurso]['cantidad'])
            ventanas, cambios_recurso = calcular_ventanas_flota(
                recurso, cantidad_base, plan_recursos_turno, duracion_turno
            )
            ventanas_por_recurso[recurso] = ventanas
            registro_cambios_flota.extend(cambios_recurso)
    
    # Cada cuantas simulaciones se notifica el progreso al callback. Se evita notificar en
    # cada iteracion cuando n_simulaciones es grande (ej. 1000+) para no saturar la UI con
    # actualizaciones innecesarias; con n_simulaciones chico (ej. 10) notifica cada una.
    intervalo_notificacion_progreso = max(1, n_simulaciones // 100)

    for sim in range(n_simulaciones):
        # Inicializar recursos segun sus ventanas de actividad (cantidad base +/- cambios de plan)
        instancias_recursos = {}
        for recurso in recursos_necesarios:
            if recurso in ventanas_por_recurso and ventanas_por_recurso[recurso]:
                instancias_recursos[recurso] = []
                for ventana in ventanas_por_recurso[recurso]:
                    instancias_recursos[recurso].append({
                        'id': ventana['id'], 
                        'x': frentes_info[frentes_nombres[0]]['xi'],
                        'y': frentes_info[frentes_nombres[0]]['yi'],
                        'frente_actual': frentes_nombres[0], 
                        'disponible_en': max(0.0, ventana['activa_desde']),
                        'activa_desde': ventana['activa_desde'],
                        'activa_hasta': ventana['activa_hasta']
                    })

        # Registrar en la traza los cambios de flota que ocurren dentro del horizonte simulado,
        # para que se vean en la carta Gantt / resultados junto con su demora de activacion.
        # Se registran DOS eventos por cambio:
        #  1) 'fleet_change' (como antes): trazabilidad global del cambio de flota, no asociado
        #     a un tunel (se sigue mostrando en el expander de "Linea de tiempo de cambios de flota").
        #  2) 'delay_fleet_change' (NUEVO): solo si demora_horas > 0. Este SI se asocia a cada
        #     frente que usa ese recurso (Resource_Origin_Tunnel = frente), y su 'type' esta
        #     incluido en la lista de eventos de ciclo/Gantt, para que aparezca como una demora
        #     mas ("Demora cambio de recurso {recurso}") en el Resumen de Tiempos y en la Carta
        #     Gantt de cada tunel afectado.
        for cambio in registro_cambios_flota:
            if cambio['tiempo_efectivo'] <= tiempo_limite:
                traza_eventos.append({
                    'Simulation_ID': sim + 1, 'Cycle': -1,
                    'Actividad': f"Cambio de flota: {cambio['tipo']} ({cambio['cantidad_anterior']} -> {cambio['cantidad_nueva']})",
                    'Recurso': cambio['recurso'], 'type': 'fleet_change',
                    'Start': round(cambio['tiempo_efectivo'], 2),
                    'Finish': round(cambio['tiempo_efectivo'], 2),
                    'Duration': round(cambio['demora_horas'], 2),
                    'Start_X_Front': None, 'Start_Y_Front': None,
                    'End_X_Front': None, 'End_Y_Front': None,
                    'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': 'N/A',
                    'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                    'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                    'Es_Personalizada': True,
                    'Turno_Plan': cambio['turno'], 'Demora_Aplicada_h': cambio['demora_horas']
                })

                if cambio.get('demora_horas', 0.0) > 1e-9 and cambio['recurso'] in recursos_necesarios:
                    for frente_afectado in frentes_nombres:
                        pos_x_fc, pos_y_fc = calcular_posicion_avanzada(
                            frentes_info[frente_afectado]['xi'], frentes_info[frente_afectado]['yi'],
                            frentes_info[frente_afectado]['xf'], frentes_info[frente_afectado]['yf'],
                            0, frentes_info[frente_afectado]['distancia']
                        )
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': -1,
                            'Actividad': f"Demora cambio de recurso {cambio['recurso']}",
                            'Recurso': cambio['recurso'], 'type': 'delay_fleet_change',
                            'Start': round(cambio['tiempo_efectivo'], 2),
                            'Finish': round(cambio['tiempo_efectivo'] + cambio['demora_horas'], 2),
                            'Duration': round(cambio['demora_horas'], 2),
                            'Start_X_Front': round(pos_x_fc, 2), 'Start_Y_Front': round(pos_y_fc, 2),
                            'End_X_Front': round(pos_x_fc, 2), 'End_Y_Front': round(pos_y_fc, 2),
                            'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente_afectado,
                            'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                            'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                            'Es_Personalizada': False
                        })

        # Estado de los frentes
        avances_frentes = {frente: 0 for frente in frentes_nombres}
        # Captura del avance REAL acumulado al cierre de cada turno, tomado directamente
        # de avances_frentes (la misma variable que usa "resultados"/Avance Final). Esto
        # reemplaza la reconstruccion posterior desde la traza de eventos (que contaba
        # ciclos con actividad registrada aunque estuvieran truncados por tiempo_limite o
        # por bloqueo de recurso, inflando el avance por turno respecto al Avance Final real).
        avance_por_turno_real = {frente: {} for frente in frentes_nombres}
        siguiente_turno_a_registrar = {frente: 1 for frente in frentes_nombres}
        secuencia_actual_frente = {frente: 0 for frente in frentes_nombres}  # Índice en secuencias_ordenadas
        ciclos_completados_frente = {frente: 0 for frente in frentes_nombres}
        # Frentes bloqueados permanentemente por falta absoluta de un recurso requerido
        # (cantidad 0 ahora y en todo el resto del horizonte de simulación).
        frente_bloqueado = {frente: False for frente in frentes_nombres}
        
        # Cola de eventos: (tiempo, tipo, datos)
        event_queue = []
        event_counter = 0  # Para desempatar eventos con mismo tiempo
        
        # Índices para velocidades
        idx_velocidades = {recurso: sim * 100 for recurso in velocidades_config.keys()}
        
        global_cycle_count = 0
        
        # Inicializar: agregar eventos para iniciar primera secuencia de cada frente
        for frente in frentes_nombres:
            heapq.heappush(event_queue, (0, event_counter, 'start_sequence', {
                'frente': frente,
                'secuencia_idx': 0
            }))
            event_counter += 1
        
        # Procesar eventos
        while event_queue:
            current_time, _, event_type, event_data = heapq.heappop(event_queue)
            
            if current_time >= tiempo_limite:
                break

            # Registrar avance REAL acumulado (avances_frentes) en cada turno cuyo limite
            # de tiempo ya quedo atras respecto al reloj de la simulacion. Se hace para
            # todos los frentes (no solo el del evento actual) para que un frente sin mas
            # eventos propios (bloqueado, terminado, sin recurso) siga registrando su
            # avance congelado en los turnos que van pasando.
            for frente_reg in frentes_nombres:
                turno_pendiente = siguiente_turno_a_registrar[frente_reg]
                while turno_pendiente * duracion_turno <= current_time:
                    avance_por_turno_real[frente_reg][turno_pendiente] = min(
                        avances_frentes[frente_reg], frentes_info[frente_reg]['distancia']
                    )
                    turno_pendiente += 1
                siguiente_turno_a_registrar[frente_reg] = turno_pendiente

            frente = event_data['frente']
            
            # Verificar si el frente ya completó su distancia
            if avances_frentes[frente] >= frentes_info[frente]['distancia']:
                continue

            # Verificar si el frente quedó bloqueado permanentemente por falta de un recurso
            if frente_bloqueado[frente]:
                continue
            
            if event_type == 'start_sequence':
                secuencia_idx = event_data['secuencia_idx']
                
                # Verificar si todavía hay secuencias por ejecutar
                if secuencia_idx >= len(secuencias_ordenadas):
                    # Ciclo completo, avanzar y reiniciar secuencias
                    
                    metros_avance_ciclo = metros_ciclo(frente)

                    avances_frentes[frente] += metros_avance_ciclo
                    ciclos_completados_frente[frente] += 1
                    
                    # Si no completó la distancia, iniciar nuevo ciclo
                    if avances_frentes[frente] < frentes_info[frente]['distancia']:
                        heapq.heappush(event_queue, (current_time, event_counter, 'start_sequence', {
                            'frente': frente,
                            'secuencia_idx': 0
                        }))
                        event_counter += 1
                    continue
                
                secuencia_num = secuencias_ordenadas[secuencia_idx]
                acts_secuencia_base = actividades_filtradas[actividades_filtradas['Secuencia'] == secuencia_num].copy()
                
                # --- MODIFICACION: Filtrar actividades por frente (Personalizadas vs. Originales) ---
                def is_applicable(row, frente_actual):
                    if row.get('Es_Personalizada', False):
                        # Solo aplicable si el frente está en la lista de frentes aplicables
                        return frente_actual in row.get('Frentes_Aplicables', [])
                    else:
                        # Las actividades no personalizadas se aplican a todos los frentes por defecto
                        return True

                acts_secuencia = acts_secuencia_base[
                    acts_secuencia_base.apply(lambda row: is_applicable(row, frente), axis=1)
                ].copy()

                if acts_secuencia.empty:
                    # Si no hay actividades para esta secuencia/frente, pasar a la siguiente secuencia inmediatamente.
                    heapq.heappush(event_queue, (current_time, event_counter, 'start_sequence', {
                        'frente': frente,
                        'secuencia_idx': secuencia_idx + 1
                    }))
                    event_counter += 1
                    continue
                # --- FIN MODIFICACION ---

                # Aplicar demoras al inicio de cada ciclo (solo en secuencia 0)
                tiempo_inicio_secuencia = current_time
                if secuencia_idx == 0:
                    global_cycle_count += 1
                    
                    # Demoras constantes (FH/OC)
                    demora_fh_oc = demoras_constantes.get(frente, 0)
                    if demora_fh_oc > 0:
                        pos_x, pos_y = calcular_posicion_avanzada(
                            frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                            frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                            avances_frentes[frente], frentes_info[frente]['distancia']
                        )
                        
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                            'Actividad': 'Demora por Frente Hundimiento, Restricción por Obras Civiles',
                            'Recurso': 'N/A', 'type': 'delay_fh_oc',
                            'Start': round(tiempo_inicio_secuencia, 2),
                            'Finish': round(tiempo_inicio_secuencia + demora_fh_oc, 2),
                            'Duration': round(demora_fh_oc, 2),
                            'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                            'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                            'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                            'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                            'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                            'Es_Personalizada': False # NO es personalizada
                        })
                        tiempo_inicio_secuencia += demora_fh_oc
                    
                    # Demoras por turno (R) — turno 1-based: turno 1 = [0, duracion_turno)
                    turno_inicio = int(tiempo_inicio_secuencia / duracion_turno) + 1
                    for res in demoras_por_turno:
                        if res['Frente'] == frente and res['Turno'] == turno_inicio:
                            demora_r = res['Demora']
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                                'Actividad': res['Tipo'], 'Recurso': 'N/A', 'type': 'delay_res',
                                'Start': round(tiempo_inicio_secuencia, 2),
                                'Finish': round(tiempo_inicio_secuencia + demora_r, 2),
                                'Duration': round(demora_r, 2),
                                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                                'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                'Es_Personalizada': False # NO es personalizada
                            })
                            tiempo_inicio_secuencia += demora_r
                    
                    # --- NUEVO: Demoras por Falla de Equipo ---
                    if frente in mapa_fallas_equipos:
                        falla_rule = mapa_fallas_equipos[frente]
                        demora_total_falla = 0
                        recursos_fallados = []

                        for recurso, params in falla_rule['fallas_por_recurso'].items():
                            prob_falla = params.get('probabilidad', 0)
                            demora_falla = params.get('demora', 0)

                            if prob_falla > 0 and np.random.rand() < (prob_falla / 100.0):
                                demora_total_falla += demora_falla
                                recursos_fallados.append(recurso)

                        if demora_total_falla > 0:
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )
                            
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                                'Actividad': 'Demora por Falla de Equipos',
                                'Recurso': ", ".join(recursos_fallados),
                                'type': 'delay_equipment_failure',
                                'Start': round(tiempo_inicio_secuencia, 2),
                                'Finish': round(tiempo_inicio_secuencia + demora_total_falla, 2),
                                'Duration': round(demora_total_falla, 2),
                                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                                'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                'Es_Personalizada': False
                            })
                            tiempo_inicio_secuencia += demora_total_falla
                
                # Procesar cada actividad de la secuencia
                tiempo_max_secuencia = tiempo_inicio_secuencia
                recursos_asignados_secuencia = []  # Permite mas de una actividad con el mismo tipo de recurso.
                recursos_reservados_secuencia = set()
                bloqueo_permanente = None  # Si se detecta falta absoluta de un recurso requerido
                
                # PASO 1: Asignar recursos a actividades (sin registrar viajes aún)
                for _, act in acts_secuencia.iterrows():
                    recurso = act['Recurso']
                    
                    # Si no requiere recurso, ejecutar directamente
                    if pd.isna(recurso):
                        duracion_act = generar_tiempos_batch(act['Tiempo'], act['Distribucion'], 1)[0]
                        
                        start_act = tiempo_inicio_secuencia
                        finish_act = start_act + duracion_act
                        
                        if finish_act > tiempo_limite:
                            finish_act = tiempo_limite
                            duracion_act = finish_act - start_act
                        
                        if duracion_act > 1e-6:
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )
                            
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                                'Actividad': act['Actividad'], 'Recurso': 'N/A', 'type': 'activity',
                                'Start': round(start_act, 2), 'Finish': round(finish_act, 2),
                                'Duration': round(duracion_act, 2),
                                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                'Cycle_Advance_Length_m': metros_ciclo(frente), 'Resource_Origin_Tunnel': frente,
                                'Resource_Destination_Tunnel': frente, 'Travel_Time_Actual': 0,
                                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                'Es_Personalizada': act.get('Es_Personalizada', False) # <--- PROPAGAR FLAG
                            })
                        
                        tiempo_max_secuencia = max(tiempo_max_secuencia, finish_act)
                        continue
                    
                    # Buscar recurso disponible. Si el recurso ni siquiera existe en la
                    # configuracion (cantidad 0 y nunca lo aumenta el plan de flota), esta
                    # actividad JAMAS podra ejecutarse: el frente queda bloqueado.
                    if recurso not in instancias_recursos:
                        bloqueo_permanente = recurso
                        break
                    
                    mejor_inst = None
                    menor_tiempo_llegada = float('inf')
                    info_viaje = None
                    
                    pos_x_frente, pos_y_frente = calcular_posicion_avanzada(
                        frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                        frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                        avances_frentes[frente], frentes_info[frente]['distancia']
                    )

                    for inst in instancias_recursos[recurso]:
                        if (recurso, inst['id']) in recursos_reservados_secuencia:
                            continue
                        # Una instancia ya retirada definitivamente para este instante nunca
                        # podra atender esta actividad.
                        if tiempo_inicio_secuencia >= inst.get('activa_hasta', float('inf')):
                            continue
                        tiempo_viaje = 0
                        distancia_viaje = 0
                        velocidad_km_h = 15.0
                        necesita_viaje = inst['frente_actual'] != frente
                        
                        if necesita_viaje:
                            # --- NUEVA LÓGICA DE VIAJE EN U ---
                            frente_origen = inst['frente_actual']
                            
                            # 1. Salida del túnel actual (desde posición actual hasta entrada del túnel origen)
                            entrada_origen_x = frentes_info[frente_origen]['xi']
                            entrada_origen_y = frentes_info[frente_origen]['yi']
                            distancia_salida = calcular_distancia(inst['x'], inst['y'], entrada_origen_x, entrada_origen_y)
                            
                            # 2. Viaje entre entradas de túneles (entrada origen a entrada destino)
                            entrada_destino_x = frentes_info[frente]['xi']
                            entrada_destino_y = frentes_info[frente]['yi']
                            distancia_entre_tuneles = calcular_distancia(entrada_origen_x, entrada_origen_y, entrada_destino_x, entrada_destino_y)
                            
                            # 3. Entrada al túnel destino (desde entrada hasta posición del frente)
                            distancia_entrada = calcular_distancia(entrada_destino_x, entrada_destino_y, pos_x_frente, pos_y_frente)
                            
                            # Distancia total del viaje en U
                            distancia_viaje = distancia_salida + distancia_entre_tuneles + distancia_entrada
                            
                            # Solo calcular viaje si la distancia es significativa
                            if distancia_viaje > 0.1:  # Más de 10cm
                                idx_v = idx_velocidades.get(recurso, 0)
                                if recurso in velocidades_precalculadas and idx_v < len(velocidades_precalculadas[recurso]):
                                    velocidad_km_h = velocidades_precalculadas[recurso][idx_v]
                                else:
                                    velocidad_km_h = np.mean(velocidades_precalculadas.get(recurso, [15.0]))
                                
                                distancia_km = distancia_viaje / 1000.0
                                tiempo_viaje = distancia_km / velocidad_km_h if velocidad_km_h > 0 else 0
                            else:
                                necesita_viaje = False  # Distancia insignificante
                            # --- FIN NUEVA LÓGICA ---
                        
                        # Instante mas temprano en que esta instancia podria efectivamente
                        # atender la actividad: lo maximo entre (a) cuando queda libre de su
                        # tarea anterior, (b) el inicio de esta secuencia, y (c) el instante en
                        # que la instancia entra en vigencia segun el plan de flota (si aun no
                        # se ha incorporado, hay que ESPERAR a que se incorpore, no descartarla).
                        tiempo_disponible_inst = max(inst['disponible_en'], tiempo_inicio_secuencia, inst.get('activa_desde', 0.0))
                        # Si para cuando la instancia estaria libre y vigente ya fue retirada, no sirve.
                        if tiempo_disponible_inst >= inst.get('activa_hasta', float('inf')):
                            continue

                        tiempo_llegada = tiempo_disponible_inst + tiempo_viaje
                        if tiempo_llegada >= inst.get('activa_hasta', float('inf')):
                            continue
                        
                        if tiempo_llegada < menor_tiempo_llegada:
                            menor_tiempo_llegada = tiempo_llegada
                            mejor_inst = inst
                            info_viaje = {
                                'necesita_viaje': necesita_viaje,
                                'distancia': distancia_viaje,
                                'tiempo': tiempo_viaje,
                                'velocidad': velocidad_km_h,
                                'idx_velocidad': idx_velocidades.get(recurso, 0)
                            }

                    if mejor_inst is None:
                        # Ninguna instancia de este recurso podra atender la actividad, ni ahora
                        # ni en el resto del horizonte simulado (todas retiradas / inexistentes
                        # para siempre desde este instante). El frente queda bloqueado: no puede
                        # completar el ciclo sin este recurso.
                        bloqueo_permanente = recurso
                        break
                    
                    # Guardar la asignación para procesarla después
                    recursos_reservados_secuencia.add((recurso, mejor_inst['id']))
                    recursos_asignados_secuencia.append({
                        'recurso': recurso,
                        'instancia': mejor_inst,
                        'info_viaje': info_viaje,
                        'actividad': act,
                        'pos_frente': (pos_x_frente, pos_y_frente)
                    })

                # Si se detectó falta absoluta de un recurso requerido, el frente no puede
                # completar este ciclo: se bloquea su avance de forma permanente a partir de
                # este instante (no se procesan las asignaciones parciales ni se agenda la
                # siguiente secuencia).
                if bloqueo_permanente is not None:
                    frente_bloqueado[frente] = True
                    pos_x_b, pos_y_b = calcular_posicion_avanzada(
                        frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                        frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                        avances_frentes[frente], frentes_info[frente]['distancia']
                    )
                    traza_eventos.append({
                        'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                        'Actividad': f"BLOQUEO: sin '{bloqueo_permanente}' disponible para completar el ciclo",
                        'Recurso': bloqueo_permanente, 'type': 'blocked_no_resource',
                        'Start': round(tiempo_inicio_secuencia, 2), 'Finish': round(tiempo_limite, 2),
                        'Duration': round(tiempo_limite - tiempo_inicio_secuencia, 2),
                        'Start_X_Front': round(pos_x_b, 2), 'Start_Y_Front': round(pos_y_b, 2),
                        'End_X_Front': round(pos_x_b, 2), 'End_Y_Front': round(pos_y_b, 2),
                        'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                        'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                        'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                        'Es_Personalizada': False
                    })
                    continue
                
                # PASO 2: Procesar recursos asignados (registrar viajes y actividades)
                for asignacion in recursos_asignados_secuencia:
                    recurso = asignacion['recurso']
                    mejor_inst = asignacion['instancia']
                    info_viaje = asignacion['info_viaje']
                    act = asignacion['actividad']
                    pos_x_frente, pos_y_frente = asignacion['pos_frente']
                    
                    # Tiempo inicial del recurso
                    tiempo_disponible_recurso = max(mejor_inst['disponible_en'], tiempo_inicio_secuencia)
                    
                    # Registrar viaje si es necesario
                    if info_viaje['necesita_viaje'] and info_viaje['tiempo'] > 1e-3:
                        start_viaje = tiempo_disponible_recurso
                        finish_viaje = start_viaje + info_viaje['tiempo']
                        
                        if finish_viaje > tiempo_limite:
                            finish_viaje = tiempo_limite
                        
                        if finish_viaje > start_viaje + 1e-6:
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                                'Actividad': act['Actividad'], 
                                'Recurso': f"{recurso}_{mejor_inst['id']}", 
                                'type': 'travel',
                                'Start': round(start_viaje, 2), 'Finish': round(finish_viaje, 2),
                                'Duration': round(finish_viaje - start_viaje, 4),
                                'Start_X_Front': round(mejor_inst['x'], 2),
                                'Start_Y_Front': round(mejor_inst['y'], 2),
                                'End_X_Front': round(pos_x_frente, 2),
                                'End_Y_Front': round(pos_y_frente, 2),
                                'Cycle_Advance_Length_m': 0,
                                'Resource_Origin_Tunnel': mejor_inst['frente_actual'],
                                'Resource_Destination_Tunnel': frente,
                                'Travel_Time_Actual': round(info_viaje['tiempo'], 4),
                                'Travel_Speed_Used_m_h': round(info_viaje['velocidad'] * 1000, 2),
                                'Travel_Distance_m': round(info_viaje['distancia'], 2),
                                'Es_Personalizada': False # Los viajes no son una actividad personalizada para el Gantt
                            })
                            
                            # Actualizar posición después del viaje
                            mejor_inst['x'] = pos_x_frente
                            mejor_inst['y'] = pos_y_frente
                            mejor_inst['frente_actual'] = frente
                            mejor_inst['disponible_en'] = finish_viaje
                            tiempo_disponible_recurso = finish_viaje
                            
                            idx_velocidades[recurso] = info_viaje['idx_velocidad'] + 1
                    
                    # Ejecutar actividad
                    duracion_act = generar_tiempos_batch(act['Tiempo'], act['Distribucion'], 1)[0]

                    # --- MODIFICACION: Aplicar ponderador de recurso por restricción geológica ---
                    if frente in mapa_restriccion_geologica:
                        res_geo = mapa_restriccion_geologica[frente]
                        ponderadores = res_geo.get('ponderadores_recursos', {})
                        if recurso in ponderadores:
                            ponderador_pct = ponderadores[recurso]
                            factor_ajuste = 1 + (ponderador_pct / 100.0)
                            duracion_act *= factor_ajuste
                    # --- FIN MODIFICACION ---

                    start_act = tiempo_disponible_recurso
                    finish_act = start_act + duracion_act
                    
                    if finish_act > tiempo_limite:
                        finish_act = tiempo_limite
                        duracion_act = finish_act - start_act
                    
                    if duracion_act > 1e-6:
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': global_cycle_count,
                            'Actividad': act['Actividad'],
                            'Recurso': f"{recurso}_{mejor_inst['id']}", 
                            'type': 'activity',
                            'Start': round(start_act, 2), 'Finish': round(finish_act, 2),
                            'Duration': round(duracion_act, 2),
                            'Start_X_Front': round(pos_x_frente, 2),
                            'Start_Y_Front': round(pos_y_frente, 2),
                            'End_X_Front': round(pos_x_frente, 2),
                            'End_Y_Front': round(pos_y_frente, 2),
                            'Cycle_Advance_Length_m': metros_ciclo(frente),
                            'Resource_Origin_Tunnel': frente,
                            'Resource_Destination_Tunnel': frente,
                            'Travel_Time_Actual': 0, 'Travel_Speed_Used_m_h': 0,
                            'Travel_Distance_m': 0,
                            'Es_Personalizada': act.get('Es_Personalizada', False) # <--- PROPAGAR FLAG
                        })
                    
                    # ✅ LIBERACIÓN INMEDIATA del recurso después de la actividad
                    mejor_inst['disponible_en'] = finish_act
                    mejor_inst['x'] = pos_x_frente
                    mejor_inst['y'] = pos_y_frente
                    mejor_inst['frente_actual'] = frente
                    
                    tiempo_max_secuencia = max(tiempo_max_secuencia, finish_act)
                
                # Agregar evento para siguiente secuencia
                secuencia_siguiente = secuencia_idx + 1
                if secuencia_siguiente >= len(secuencias_ordenadas) and tiempo_max_secuencia >= tiempo_limite:
                    # Esta era la ultima secuencia del ciclo y ya no queda tiempo para que el
                    # evento 'start_sequence' que cerraria el ciclo llegue a procesarse (el
                    # bucle principal corta con current_time >= tiempo_limite). Sin este ajuste,
                    # las actividades del ciclo ya quedaban en la traza (avance_actividad) pero
                    # avances_frentes nunca sumaba esos metros: se cerraba el ciclo aqui mismo,
                    # en el mismo instante en que se sabe que no quedan mas secuencias, en vez
                    # de depender de un evento futuro que nunca se alcanza a procesar.
                    metros_avance_ciclo = metros_ciclo(frente)
                    avances_frentes[frente] += metros_avance_ciclo
                    ciclos_completados_frente[frente] += 1
                else:
                    heapq.heappush(event_queue, (tiempo_max_secuencia, event_counter, 'start_sequence', {
                        'frente': frente,
                        'secuencia_idx': secuencia_siguiente
                    }))
                    event_counter += 1
        
        # Guardar resultados
        for frente in frentes_nombres:
            avance_final_frente = min(avances_frentes[frente], frentes_info[frente]['distancia'])
            resultados[frente].append(avance_final_frente)

            # Completar los turnos que quedaron sin registrar (la simulacion termino antes
            # de tiempo_limite, o el event_queue se vacio) con el avance final ya conocido,
            # para que la serie por turno tenga el mismo largo para todos los frentes.
            n_turnos_frente = math.ceil(tiempo_limite / duracion_turno)
            turno_pendiente = siguiente_turno_a_registrar[frente]
            while turno_pendiente <= n_turnos_frente:
                avance_por_turno_real[frente][turno_pendiente] = avance_final_frente
                turno_pendiente += 1

            # Volcar al dict que se retorna/expone (lista ordenada 1..N, alineada 1 a 1
            # con la simulacion "sim+1", igual que resultados[frente]).
            for turno_idx in range(1, n_turnos_frente + 1):
                avance_por_turno_resultado[frente][turno_idx].append(
                    avance_por_turno_real[frente].get(turno_idx, avance_final_frente)
                )

        # Notificar progreso (simulaciones completadas / total) para que la UI pueda
        # mostrar un contador tipo "3/10" y mover la barra en vivo en vez de quedar
        # estatica hasta el final de todo el loop.
        if progress_callback and ((sim + 1) % intervalo_notificacion_progreso == 0 or (sim + 1) == n_simulaciones):
            progress_callback(sim + 1, n_simulaciones)
    
    # Calcular estadísticas de recursos
    estadisticas_recursos = {}
    if traza_eventos:
        df_traza = pd.DataFrame(traza_eventos)
            
        for recurso_tipo in recursos_config.keys():
            if recursos_config[recurso_tipo]['cantidad'] > 0:
                utilizaciones = []
                tiempos_trabajo = []
                tiempos_viaje = []

                # Tiempo disponible real integrando las ventanas de actividad del plan de
                # flota por turno (si no hay cambios de plan, equivale a cantidad_base * tiempo_limite)
                ventanas_recurso = ventanas_por_recurso.get(recurso_tipo, [])
                if ventanas_recurso:
                    tiempo_total_disponible = sum(
                        max(0.0, min(v['activa_hasta'], tiempo_limite) - min(v['activa_desde'], tiempo_limite))
                        for v in ventanas_recurso
                    )
                else:
                    tiempo_total_disponible = tiempo_limite * recursos_config[recurso_tipo]['cantidad']

                for sim_id in range(1, n_simulaciones + 1):
                    df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
                    
                    tiempo_trabajo_sim = df_sim[(df_sim['Recurso'].str.startswith(recurso_tipo)) & (df_sim['type'] == 'activity')]['Duration'].sum()
                    tiempo_viaje_sim = df_sim[(df_sim['Recurso'].str.startswith(recurso_tipo)) & (df_sim['type'] == 'travel')]['Duration'].sum()
                    
                    tiempo_total_ocupado = tiempo_trabajo_sim + tiempo_viaje_sim
                    
                    utilizacion = (tiempo_total_ocupado / tiempo_total_disponible) * 100 if tiempo_total_disponible > 0 else 0
                    
                    utilizaciones.append(min(100.0, utilizacion))
                    tiempos_trabajo.append(tiempo_trabajo_sim)
                    tiempos_viaje.append(tiempo_viaje_sim)

                estadisticas_recursos[recurso_tipo] = {
                    'utilizacion': utilizaciones, 
                    'tiempo_trabajando': tiempos_trabajo, 
                    'tiempo_viaje': tiempos_viaje
                }

    return resultados, estadisticas_recursos, traza_eventos, ventanas_por_recurso, registro_cambios_flota, avance_por_turno_resultado

# Funciones Auxiliares para el Cálculo por Turnos
@st.cache_data(show_spinner=False)
def calcular_avance_por_turno(_run_id, df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance):
    """
    Cacheada con @st.cache_data: agrupa/ordena la traza completa por ciclo y turno, algo
    costoso con cientos de miles de filas. Sin cache, Streamlit repetia este groupby en
    cada rerun (ej. al cambiar el tunel seleccionado en otra seccion), aunque la traza no
    hubiera cambiado. El parametro _run_id (con guion bajo, para que Streamlit no intente
    hashear el DataFrame completo) identifica de forma unica cada corrida de simulacion:
    mientras no cambie, se reutiliza el resultado ya calculado.
    """
    return _calcular_avance_por_turno_impl(df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance)


def _calcular_avance_por_turno_impl(df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance):
    """Calcula la distancia avanzada por turno para cada simulación y frente."""
    
    if df_traza_completa.empty:
        return {}

    if '12x2' in sistema_turnos:
        duracion_turno = 12.0
    elif '8x3' in sistema_turnos:
        duracion_turno = 8.0
    else:
        duracion_turno = 24.0

    n_turnos = math.ceil(tiempo_limite / duracion_turno)
    frentes_nombres = list(frentes_info.keys())
    
    avance_por_turno = {frente: {turno: [] for turno in range(1, n_turnos + 1)} for frente in frentes_nombres}
    
    # Se ordena para que la posición N-1 de cada lista corresponda siempre a la
    # simulación N (permite luego seleccionar una simulación específica por índice).
    sim_ids = sorted(df_traza_completa['Simulation_ID'].unique())

    for sim_id in sim_ids:
        df_sim = df_traza_completa[df_traza_completa['Simulation_ID'] == sim_id].copy()
        
        for frente in frentes_nombres:
            # Contar ciclos completados basado en el número de veces que aparece la última secuencia
            df_frente_sim = df_sim[
                (df_sim['Resource_Origin_Tunnel'] == frente) &
                (df_sim['type'] == 'activity')
            ].copy()
            
            if df_frente_sim.empty:
                for turno in range(1, n_turnos + 1):
                    avance_por_turno[frente][turno].append(0)
                continue
            
            df_frente_sim = df_frente_sim.sort_values(by='Finish')
            
            # Agrupar por ciclo y obtener el tiempo de finalización de cada ciclo
            ciclos_finish_times = df_frente_sim.groupby('Cycle')['Finish'].max().sort_values()
            
            for turno in range(1, n_turnos + 1):
                tiempo_fin_turno = turno * duracion_turno
                
                # Contar ciclos completados hasta el fin del turno
                ciclos_completados = (ciclos_finish_times <= tiempo_fin_turno).sum()
                
                avance_ciclo = frentes_info.get(frente, {}).get('metros_por_ciclo', metros_avance)
                if 'Cycle_Advance_Length_m' in df_frente_sim.columns:
                    avances_registrados = df_frente_sim['Cycle_Advance_Length_m'].dropna()
                    avances_registrados = avances_registrados[avances_registrados > 0]
                    if not avances_registrados.empty:
                        avance_ciclo = float(avances_registrados.max())

                avance_acumulado_sim = min(
                    ciclos_completados * avance_ciclo,
                    frentes_info[frente]['distancia']
                )
                
                avance_por_turno[frente][turno].append(avance_acumulado_sim)

    return avance_por_turno


@st.cache_data(show_spinner=False)
def calcular_resumen_bloqueos(_df_traza, run_id):
    """Agrupa los eventos de bloqueo por falta de recurso. Cacheada porque este groupby
    se repetia en cada rerun de la Seccion 13, aunque la traza no cambiara."""
    if _df_traza.empty or 'type' not in _df_traza.columns:
        return None
    df_bloqueos = _df_traza[_df_traza['type'] == 'blocked_no_resource']
    if df_bloqueos.empty:
        return None
    resumen = df_bloqueos.groupby(['Resource_Origin_Tunnel', 'Recurso']).agg(
        Simulaciones_Afectadas=('Simulation_ID', 'nunique'),
        Hora_Bloqueo_Promedio=('Start', 'mean')
    ).reset_index().rename(columns={
        'Resource_Origin_Tunnel': 'Túnel', 'Recurso': 'Recurso Faltante'
    })
    resumen['Hora_Bloqueo_Promedio'] = r2(resumen['Hora_Bloqueo_Promedio'])
    return resumen


@st.cache_data(show_spinner=False)
def calcular_tiempos_agregados_actividad(_df_tunel, run_id, tunel, sim_id):
    """Agrupa tiempos de actividad/demora por Simulation_ID y Actividad para un tunel.
    Cacheada: es el groupby que alimenta tanto el resumen de tiempos como la Carta Gantt,
    y se recalculaba en cada rerun al cambiar de tunel/simulacion."""
    df_ciclo_events = _df_tunel[_df_tunel['type'].isin(
        ['activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_equipment_failure', 'delay_fleet_change', 'blocked_no_resource']
    )].copy()
    if df_ciclo_events.empty:
        return pd.DataFrame(columns=['Simulation_ID', 'Actividad', 'Duration'])
    return df_ciclo_events.groupby(['Simulation_ID', 'Actividad'])['Duration'].sum().reset_index()


@st.cache_data(show_spinner=False)
def calcular_viajes_agregados(_df_tunel, run_id, tunel, sim_id):
    """Agrupa distancia/tiempo de viaje por Simulation_ID y tipo de recurso. Cacheada por
    la misma razon: groupby costoso repetido en cada rerun de la Seccion 14."""
    df_viajes = _df_tunel[_df_tunel['type'] == 'travel'].copy()
    if df_viajes.empty:
        return pd.DataFrame(columns=['Simulation_ID', 'Recurso_Tipo', 'Distancia_Total', 'Tiempo_Total', 'Num_Viajes'])
    df_viajes['Recurso_Tipo'] = df_viajes['Recurso'].apply(lambda x: x.split('_')[0])
    return df_viajes.groupby(['Simulation_ID', 'Recurso_Tipo']).agg(
        Distancia_Total=('Travel_Distance_m', 'sum'),
        Tiempo_Total=('Travel_Time_Actual', 'sum'),
        Num_Viajes=('Cycle', 'count')
    ).reset_index()


TIPOS_EVENTO_CICLO_GANTT = [
    'activity', 'delay_fh_oc', 'delay_res', 'delay_equipment_failure',
    'delay_fleet_change', 'blocked_no_resource'
]

# Mapeo de la etiqueta visible al percentil elegido -> nombre de columna que arma
# calcular_datos_gantt (ver PERCENTILES_DISPONIBLES_GANTT mas abajo, usado en la UI).
PERCENTILES_DISPONIBLES_GANTT = {
    'P0 (Mínimo)': 'P0', 'P10': 'P10', 'P30': 'P30', 'P50 (Mediana)': 'P50',
    'Esperanza (Promedio)': 'Esperanza', 'P70': 'P70', 'P90': 'P90', 'P100 (Máximo)': 'P100'
}


@st.cache_data(show_spinner=False)
def calcular_ciclos_completados_por_simulacion(_df_tunel, run_id, tunel, sim_id):
    """Cuenta, para cada Simulation_ID, cuantos ciclos distintos se completaron en este
    tunel (numero de valores unicos de 'Cycle' >= 0 entre los eventos de tipo 'activity').
    Se usa como divisor para el modo 'Duracion promedio por ocurrencia' de la Carta Gantt:
    cada actividad ocurre una vez por ciclo, asi que tiempo_total_actividad / n_ciclos =
    duracion tipica de UNA ocurrencia de esa actividad."""
    df_act = _df_tunel[(_df_tunel['type'] == 'activity') & (_df_tunel['Cycle'] >= 0)]
    if df_act.empty:
        return {}
    conteo = df_act.groupby('Simulation_ID')['Cycle'].nunique()
    return conteo.to_dict()


@st.cache_data(show_spinner=False)
def calcular_datos_gantt(_df_tunel, run_id, tunel, sim_id, modo='acumulado', percentil_barra='P50'):
    """Arma la lista de datos para la Carta Gantt (percentiles P0..P100/Esperanza por actividad).
    Es el calculo mas pesado de la Seccion 14 (percentiles sobre cientos de simulaciones
    por actividad); cachearlo evita recalcularlo en cada rerun cuando el tunel/simulacion/
    modo/percentil seleccionados no cambiaron.

    modo:
        - 'acumulado': tiempo TOTAL de esa actividad sumado sobre todos los ciclos de la
          simulacion (comportamiento original: si el Jumbo perfora 1h y lo hace 4 veces,
          esto acumula 4h).
        - 'promedio_ciclo': tiempo total de la simulacion / N de ciclos completados en esa
          simulacion. Osea, la duracion TIPICA de UNA sola ocurrencia de la actividad
          (si el Jumbo perfora 1h y lo hace 4 veces, esto muestra ~1h).
    percentil_barra: que percentil de la distribucion (entre simulaciones) usar como largo
        de la barra principal y como eje de acumulacion en el tiempo (Start de la siguiente
        actividad = fin de la anterior, segun este mismo percentil). Una de las claves de
        PERCENTILES_DISPONIBLES_GANTT ('P0','P10','P30','P50','Esperanza','P70','P90','P100').
    """
    df_ciclo_events = _df_tunel[_df_tunel['type'].isin(TIPOS_EVENTO_CICLO_GANTT)].copy()
    if df_ciclo_events.empty:
        return []

    tiempo_por_actividad = df_ciclo_events.groupby(
        ['Simulation_ID', 'Actividad', 'Es_Personalizada']
    )['Duration'].sum().reset_index()

    if sim_id is not None:
        tiempo_por_actividad = tiempo_por_actividad[tiempo_por_actividad['Simulation_ID'] == sim_id]

    # Modo "promedio por ocurrencia": dividir el tiempo acumulado de cada simulacion por el
    # numero de VECES que ESA MISMA actividad ocurrio en esa simulacion (no por un conteo
    # global de "ciclos" tomado de otra actividad). Antes se usaba N_Ciclos global
    # (calcular_ciclos_completados_por_simulacion), pero esa cuenta se basaba en valores
    # unicos de 'Cycle' sobre TODOS los eventos de tipo 'activity' del tunel; si otra
    # actividad distinta a la que se esta promediando aporta mas granularidad de Cycle
    # de la que le corresponde a esta actividad, el N_Ciclos queda inflado y el promedio
    # por ocurrencia sale artificialmente bajo (esto es lo que se veia: PS con P50 real
    # ~0.94h saliendo como ~0.33h en el modo "promedio por ocurrencia", porque se estaba
    # dividiendo por un N_Ciclos ~3x mayor al numero real de veces que PS ocurrio).
    if modo == 'promedio_ciclo':
        conteo_ocurrencias = df_ciclo_events[df_ciclo_events['type'] == 'activity'].groupby(
            ['Simulation_ID', 'Actividad']
        ).size().rename('N_Ocurrencias').reset_index()
        # Las demoras/bloqueos no ocurren necesariamente 1 vez por ciclo; para esas se
        # mantiene el divisor global de ciclos completados (representa "contribucion
        # promedio por ciclo" de una demora intermitente), tal como antes.
        ciclos_por_sim = calcular_ciclos_completados_por_simulacion(_df_tunel, run_id, tunel, sim_id)
        if not ciclos_por_sim:
            return []
        tiempo_por_actividad = tiempo_por_actividad.merge(
            conteo_ocurrencias, on=['Simulation_ID', 'Actividad'], how='left'
        )
        tiempo_por_actividad['N_Ciclos_Global'] = tiempo_por_actividad['Simulation_ID'].map(ciclos_por_sim)
        # Para actividades reales (type == 'activity'), divide por su propio conteo de
        # ocurrencias; para demoras/bloqueos (sin match en conteo_ocurrencias, N_Ocurrencias
        # queda NaN), cae al N_Ciclos_Global como antes.
        tiempo_por_actividad['Divisor'] = tiempo_por_actividad['N_Ocurrencias'].fillna(
            tiempo_por_actividad['N_Ciclos_Global']
        )
        tiempo_por_actividad = tiempo_por_actividad[tiempo_por_actividad['Divisor'] > 0]
        if tiempo_por_actividad.empty:
            return []
        tiempo_por_actividad['Duration'] = (
            tiempo_por_actividad['Duration'] / tiempo_por_actividad['Divisor']
        )

    actividades_gantt = sorted(
        tiempo_por_actividad['Actividad'].unique(),
        key=lambda x: 0 if (x.startswith('Demora') or x.startswith('BLOQUEO')) else 1
    )

    gantt_data = []
    tiempo_acumulado = 0
    for actividad in actividades_gantt:
        act_data = tiempo_por_actividad[tiempo_por_actividad['Actividad'] == actividad]
        tiempos_actividad = act_data['Duration'].values
        if len(tiempos_actividad) < 1:
            continue

        pct_act = percentiles_reporte(tiempos_actividad)
        p10 = np.percentile(tiempos_actividad, 10)
        p50 = np.percentile(tiempos_actividad, 50)
        p90 = np.percentile(tiempos_actividad, 90)
        es_personalizada = act_data['Es_Personalizada'].iloc[0]

        if actividad.startswith('BLOQUEO'):
            color = 'black'
        elif actividad.startswith('Demora'):
            color = 'red'
        elif es_personalizada:
            color = 'green'
        else:
            color = 'rgb(31, 119, 180)'

        fila = {
            'Actividad': actividad,
            'P0': pct_act['p0'],
            'P10': p10,
            'P30': pct_act['p30'],
            'P50': p50,
            'Esperanza': pct_act['esperanza'],
            'P70': pct_act['p70'],
            'P90': p90,
            'P100': pct_act['p100'],
            'Color': color
        }
        # El largo de la barra y el "Start" acumulado usan el percentil elegido por el
        # usuario (por defecto P50, igual que el comportamiento original).
        valor_barra = fila.get(percentil_barra, fila['P50'])
        fila['Start_Barra'] = tiempo_acumulado
        fila['Valor_Barra'] = valor_barra
        gantt_data.append(fila)
        tiempo_acumulado += valor_barra

    return gantt_data


def graficar_geometria(df_frentes, df_fh, df_oc, df_res, radio_restriccion, radios_por_tipo=None):
    """Grafica los túneles y las restricciones, incluyendo círculos de impacto."""
    
    data_g = []
    
    # 1. TÚNELES (Frentes)
    for _, row in df_frentes.iterrows():
        data_g.append(
            go.Scatter(
                x=[row['Xi'], row['Xf']], y=[row['Yi'], row['Yf']],
                mode='lines+markers',
                name=f"{row['Frentes']} (Túnel)",
                line=dict(width=3, color='green'),
                marker=dict(size=6, symbol='circle', color='green')
            )
        )
        
    # 2. Restricciones y sus radios
    restricciones = [
        (df_fh, 'FH (Frente Hundimiento)', 'red', 'circle'),
        (df_oc, 'OC (Obras Civiles)', 'blue', 'square'),
        (df_res, 'R (Restricción por Turno)', 'darkorange', 'star')
    ]
    
    legend_added = set()

    for df, name, color, symbol in restricciones:
        if 'X' in df.columns and 'Y' in df.columns and not df.empty:
            
            # Graficar los centros de restricción
            data_g.append(
                go.Scatter(
                    x=df['X'], y=df['Y'],
                    mode='markers',
                    name=name,
                    marker=dict(symbol=symbol, size=10, color=color),
                    text=df.apply(lambda r: f"{name}<br>X: {r['X']}<br>Y: {r['Y']}", axis=1),
                    hoverinfo='text'
                )
            )

            # Graficar el círculo de impacto para cada restricción
            if radio_restriccion > 0:
                for i, row in df.iterrows():
                    Px, Py = row['X'], row['Y']
                    radio_evento = radio_restriccion
                    if 'Tipo' in row.index:
                        radio_evento = restriction_radius(row.get('Tipo'), radio_restriccion, radios_por_tipo)
                    t = np.linspace(0, 2 * np.pi, 100)
                    circle_x = Px + radio_evento * np.cos(t)
                    circle_y = Py + radio_evento * np.sin(t)
                    
                    show_legend_for_radius = False
                    if name not in legend_added:
                        show_legend_for_radius = True
                        legend_added.add(name)

                    data_g.append(
                        go.Scatter(
                            x=circle_x, y=circle_y,
                            mode='lines',
                            line=dict(width=1, color=color, dash='dash'),
                            name=f'Radio de Impacto ({name.split()[0]})',
                            showlegend=show_legend_for_radius
                        )
                    )


    fig = go.Figure(data=data_g)
    fig.update_layout(
        title='Geometría del Nivel: Túneles y Restricciones',
        xaxis_title='Coordenada X',
        yaxis_title='Coordenada Y',
        legend_title="Elementos",
        hovermode="closest",
        width=None, height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig


# ============================================================
# GENERACIÓN DE OUTPUTS FORMATO DAVID MONTENEGRO
# ============================================================

def _build_actividad_to_secuencia(df_actividades):
    """Mapea nombre de actividad → número de secuencia."""
    mapping = {}
    for _, row in df_actividades.dropna(subset=['Actividad']).iterrows():
        mapping[str(row['Actividad'])] = int(row['Secuencia'])
    return mapping


def generar_advances_total_scenary(resultados, frentes_info, nivel_seleccionado):
    """
    Formato: David Montenegro Advances_total_scenary.xlsx
    Una hoja Sheet1, filas = frentes, cols = scenary 1..N
    """
    rows = []
    for i, (frente, avances_lista) in enumerate(resultados.items()):
        row = {'Index': i + 1, 'Frentes': frente, 'Nivel': nivel_seleccionado}
        for j, avance in enumerate(avances_lista):
            row[f'scenary {j + 1}'] = round(float(avance), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def generar_simulation_front_actual(traza_eventos, frentes_info, df_frentes_nivel, n_simulaciones):
    """
    Formato: David Montenegro Simulation_Front_Actual.xlsx
    Una hoja por escenario ('scenary 1'...'scenary N').
    Columnas: Nivel, Index, Frentes, Sección, Xi, Yi, Xf, Yf, Dirección, LastShift, State, shift_State, FH_Distance, OOCC_Distance
    Xi/Yi = posición final alcanzada en ese escenario.
    """
    if not traza_eventos:
        return {}
    df_traza = pd.DataFrame(traza_eventos)
    sheets = {}
    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows = []
        for i, (frente, info) in enumerate(frentes_info.items()):
            # Buscar última posición registrada para este frente en este escenario
            df_f = df_sim[df_sim['Resource_Origin_Tunnel'] == frente]
            if not df_f.empty:
                last_row = df_f.sort_values('Finish').iloc[-1]
                xi_actual = last_row['End_X_Front']
                yi_actual = last_row['End_Y_Front']
                last_shift = last_row['Finish']
            else:
                xi_actual = info['xi']
                yi_actual = info['yi']
                last_shift = 0

            avance_total = min(
                calcular_distancia(info['xi'], info['yi'], xi_actual, yi_actual),
                info['distancia']
            )
            state = 'False' if avance_total >= info['distancia'] * 0.999 else 'True'

            # Datos de la fila original de frentes
            fila_original = df_frentes_nivel[df_frentes_nivel['Frentes'] == frente]
            seccion = str(fila_original[columna_seccion(df_frentes_nivel)].iloc[0]) if (columna_seccion(df_frentes_nivel) and not fila_original.empty) else ''
            direccion = float(fila_original['Dirección'].iloc[0]) if ('Dirección' in fila_original.columns and not fila_original.empty) else 0.0

            rows.append({
                'Nivel': info.get('nivel', ''),
                'Index': i + 1,
                'Frentes': frente,
                'Sección': seccion,
                'Xi': round(xi_actual, 4),
                'Yi': round(yi_actual, 4),
                'Xf': round(info['xf'], 4),
                'Yf': round(info['yf'], 4),
                'Dirección': direccion,
                'LastShift': round(last_shift, 2),
                'State': state,
                'shift_State': 'True',
                'FH_Distance': '',
                'OOCC_Distance': ''
            })
        sheets[f'scenary {sim_id}'] = pd.DataFrame(rows)
    return sheets


def generar_simulation_program_por_turno(traza_eventos, df_actividades, frentes_info,
                                          numero_turnos, duracion_turno, n_simulaciones):
    """
    Genera 3 estructuras con formato de David Montenegro (una hoja por escenario):
    - Program_Sequences: {scenary N: DataFrame} — secuencias ejecutadas por turno por frente
    - Program_Activities: {scenary N: DataFrame} — actividades ejecutadas por turno por frente
    - Program_Advances: {scenary N: DataFrame} — avances registrados por turno por frente
    """
    if not traza_eventos:
        return {}, {}, {}

    df_traza = pd.DataFrame(traza_eventos)
    act_to_seq = _build_actividad_to_secuencia(df_actividades)

    sheets_seq, sheets_act, sheets_adv = {}, {}, {}
    turnos_cols = [f'T{t}' for t in range(1, numero_turnos + 1)]
    frentes_nombres = list(frentes_info.keys())

    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows_seq, rows_act, rows_adv = [], [], []

        for i, frente in enumerate(frentes_nombres):
            df_f = df_sim[
                (df_sim['Resource_Origin_Tunnel'] == frente) &
                (df_sim['type'] == 'activity')
            ].sort_values('Start')

            # Detectar fin de ciclos para avances
            ciclos_por_fin = {}
            if 'Cycle' in df_f.columns:
                for cycle_id, grp in df_f.groupby('Cycle'):
                    t_fin = grp['Finish'].max()
                    adv_m = grp['Cycle_Advance_Length_m'].max() if 'Cycle_Advance_Length_m' in grp.columns else 0
                    last = grp.sort_values('Finish').iloc[-1]
                    ciclos_por_fin[cycle_id] = {
                        'finish': t_fin,
                        'metros': adv_m,
                        'x': last['End_X_Front'],
                        'y': last['End_Y_Front']
                    }

            row_seq = {'Index': i + 1, 'Frentes': frente}
            row_act = {'Index': i + 1, 'Frentes': frente}
            row_adv = {'Index': i + 1, 'Frentes': frente}

            for t in range(1, numero_turnos + 1):
                t_start = (t - 1) * duracion_turno
                t_end = t * duracion_turno
                col = f'T{t}'

                df_shift = df_f[
                    (df_f['Start'] >= t_start) & (df_f['Start'] < t_end)
                ]

                if df_shift.empty:
                    row_seq[col] = ''
                    row_act[col] = ''
                else:
                    acts = df_shift['Actividad'].dropna().tolist()
                    row_act[col] = '-'.join(str(a) for a in acts)
                    seqs = [str(act_to_seq.get(str(a), '?')) for a in acts]
                    row_seq[col] = '-'.join(seqs)

                # Avances: ciclos que terminaron en este turno
                adv_parts = []
                for cid, cinfo in ciclos_por_fin.items():
                    if t_start < cinfo['finish'] <= t_end and cinfo['metros'] > 0:
                        adv_parts.append(
                            f"meters: {round(cinfo['metros'], 2)} "
                            f"X: {round(cinfo['x'], 2)} "
                            f"Y: {round(cinfo['y'], 2)}"
                        )
                row_adv[col] = ' | '.join(adv_parts)

            rows_seq.append(row_seq)
            rows_act.append(row_act)
            rows_adv.append(row_adv)

        key = f'scenary {sim_id}'
        sheets_seq[key] = pd.DataFrame(rows_seq)
        sheets_act[key] = pd.DataFrame(rows_act)
        sheets_adv[key] = pd.DataFrame(rows_adv)

    return sheets_seq, sheets_act, sheets_adv


def generar_simulation_time_program(traza_eventos, recursos_config, numero_turnos,
                                     duracion_turno, n_simulaciones, ventanas_por_recurso=None):
    """
    Formato: David Montenegro Simulation_Time_Program.xlsx
    Una hoja por escenario. Filas = recursos, cols = T1..TN.
    Valor = tiempo disponible restante en ese turno (capacidad del turno - tiempo_usado).
    Si hay un plan de orquestacion de flota por turno, la capacidad de cada turno
    refleja la cantidad de unidades realmente activas en ese turno (no la cantidad base fija).
    """
    if not traza_eventos:
        return {}
    df_traza = pd.DataFrame(traza_eventos)
    sheets = {}
    recursos_lista = [r for r, cfg in recursos_config.items() if cfg['cantidad'] > 0]
    ventanas_por_recurso = ventanas_por_recurso or {}

    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows = []
        for recurso in recursos_lista:
            df_rec = df_sim[
                df_sim['Recurso'].astype(str).str.startswith(recurso) &
                (df_sim['type'] == 'activity')
            ]
            row = {'Recurso': recurso}
            ventanas_recurso = ventanas_por_recurso.get(recurso)
            for t in range(1, numero_turnos + 1):
                t_start = (t - 1) * duracion_turno
                t_end = t * duracion_turno
                df_shift = df_rec[
                    (df_rec['Start'] >= t_start) & (df_rec['Start'] < t_end)
                ]
                tiempo_usado = df_shift['Duration'].sum()
                if ventanas_recurso:
                    unidades_activas_turno = sum(
                        1 for v in ventanas_recurso
                        if v['activa_desde'] < t_end and v['activa_hasta'] > t_start
                    )
                    capacidad_total = duracion_turno * unidades_activas_turno
                else:
                    capacidad_total = duracion_turno * recursos_config[recurso]['cantidad']
                restante = max(0.0, capacidad_total - tiempo_usado)
                row[f'T{t}'] = round(restante, 6)
            rows.append(row)
        sheets[f'scenary {sim_id}'] = pd.DataFrame(rows)
    return sheets


def escribir_excel_multihoja(filepath, sheets_dict):
    """Escribe un dict {sheet_name: DataFrame} como Excel multi-hoja."""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


def _excel_multihoja_bytes(sheets_dict):
    """Igual que escribir_excel_multihoja, pero devuelve los bytes del .xlsx
    en memoria (BytesIO) en vez de escribir en disco."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def generar_outputs_zip_en_memoria(traza_eventos, resultados, frentes_info,
                                    df_frentes_nivel, df_actividades, recursos_config,
                                    numero_turnos, duracion_turno, n_simulaciones,
                                    nivel_seleccionado):
    """
    Genera los 6 archivos en formato David Montenegro completamente en memoria
    y los empaqueta en un .zip (BytesIO), listo para st.download_button.

    Streamlit Cloud no puede escribir en el disco del usuario (la app corre en
    un servidor remoto y efímero), así que en vez de guardar en una ruta local
    como hacía la versión de escritorio, se arma todo en RAM y se ofrece como
    descarga desde el navegador.
    """
    archivos = {}

    # 1. Advances_total_scenary.xlsx
    df_advances = generar_advances_total_scenary(resultados, frentes_info, nivel_seleccionado)
    buf1 = BytesIO()
    df_advances.to_excel(buf1, index=False)
    buf1.seek(0)
    archivos['Advances_total_scenary.xlsx'] = buf1.getvalue()

    # 2. Simulation_Front_Actual.xlsx
    sheets_front = generar_simulation_front_actual(
        traza_eventos, frentes_info, df_frentes_nivel, n_simulaciones)
    archivos['Simulation_Front_Actual.xlsx'] = _excel_multihoja_bytes(sheets_front)

    # 3-5. Sequences, Activities, Advances (por turno)
    sheets_seq, sheets_act, sheets_adv = generar_simulation_program_por_turno(
        traza_eventos, df_actividades, frentes_info,
        numero_turnos, duracion_turno, n_simulaciones
    )
    archivos['Simulation_Program_Sequences.xlsx'] = _excel_multihoja_bytes(sheets_seq)
    archivos['Simulation_Program_Activities.xlsx'] = _excel_multihoja_bytes(sheets_act)
    archivos['Simulation_Program_Advances.xlsx'] = _excel_multihoja_bytes(sheets_adv)

    # 6. Simulation_Time_Program.xlsx
    sheets_time = generar_simulation_time_program(
        traza_eventos, recursos_config, numero_turnos, duracion_turno, n_simulaciones)
    archivos['Simulation_Time_Program.xlsx'] = _excel_multihoja_bytes(sheets_time)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for nombre, data in archivos.items():
            zf.writestr(nombre, data)
    zip_buf.seek(0)
    return zip_buf, list(archivos.keys())


def validar_marina(archivo_marina):
    """Valida y carga Input_Marina.xlsx (hoja Vaciaderos con sitios de depósito)."""
    errors, warnings_list = [], []
    try:
        xls = pd.ExcelFile(archivo_marina)
        if 'Vaciaderos' not in xls.sheet_names:
            # Intentar leer la primera hoja si no hay 'Vaciaderos'
            df = xls.parse(xls.sheet_names[0])
            warnings_list.append("Input_Marina: no se encontró hoja 'Vaciaderos', se usó la primera hoja.")
        else:
            df = xls.parse('Vaciaderos')
        req_cols = ['Nivel', 'Capacidad', 'Maximo', 'Tipo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            warnings_list.append(f"Input_Marina/Vaciaderos: columnas opcionales no encontradas: {', '.join(missing)}")
        return df, errors, warnings_list
    except Exception as e:
        errors.append(f"Error al leer Input_Marina: {e}")
        return pd.DataFrame(), errors, warnings_list


def validar_muckpile(archivo_muckpile):
    """Valida y carga Muckpile.xlsx (estado actual del stockpile por nivel)."""
    errors, warnings_list = [], []
    try:
        xls = pd.ExcelFile(archivo_muckpile)
        sheet = xls.sheet_names[0]
        df = xls.parse(sheet)
        req_cols = ['Nivel', 'Capacidad Ocupada', 'Maximo', 'Tipo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            warnings_list.append(f"Muckpile/{sheet}: columnas opcionales no encontradas: {', '.join(missing)}")
        return df, errors, warnings_list
    except Exception as e:
        errors.append(f"Error al leer Muckpile: {e}")
        return pd.DataFrame(), errors, warnings_list


def calcular_capacidad_disponible_muckpile(df_muckpile, nivel_seleccionado):
    """Retorna capacidad libre total en el nivel dado (Máximo - Ocupada)."""
    if df_muckpile is None or df_muckpile.empty:
        return None
    if 'Nivel' not in df_muckpile.columns:
        return None
    df_nivel = df_muckpile[df_muckpile['Nivel'] == nivel_seleccionado]
    if df_nivel.empty:
        return None
    total_max = df_nivel['Maximo'].sum() if 'Maximo' in df_nivel.columns else 0
    total_ocupado = df_nivel['Capacidad Ocupada'].sum() if 'Capacidad Ocupada' in df_nivel.columns else 0
    return max(0.0, float(total_max) - float(total_ocupado))


# Inicializar session state
if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = False
if 'data_version' not in st.session_state:
    # Se incrementa cada vez que se cargan archivos nuevos y validos (Menu 1). Junto con
    # nivel_seleccionado, es la clave de cache de filtrar_actividades_nivel/recursos_nivel/
    # frentes_nivel: mientras no cambie, se reutiliza el filtrado ya calculado.
    st.session_state.data_version = 0
if 'actividades_modificadas' not in st.session_state:
    st.session_state.actividades_modificadas = {}
if 'actividades_originales' not in st.session_state:
    st.session_state.actividades_originales = pd.DataFrame() # Inicializar como DataFrame vacío
if 'actividades_adicionales' not in st.session_state:
    st.session_state.actividades_adicionales = []
if 'restricciones_geologicas' not in st.session_state:
    st.session_state.restricciones_geologicas = []
if 'fallas_equipos' not in st.session_state:
    st.session_state.fallas_equipos = []
if 'df_marina' not in st.session_state:
    st.session_state.df_marina = pd.DataFrame()
if 'df_muckpile' not in st.session_state:
    st.session_state.df_muckpile = pd.DataFrame()


# SECCION DE INPUTS
st.header("Parámetros de Entrada")

# ----------------------------------------------------
# 1. Cargar Datos 
# ----------------------------------------------------
st.subheader("1. Cargar Datos")
st.info(
    """
    **Archivos de entrada requeridos (2 obligatorios + 2 opcionales):**

    - **Archivo 1 – Operación** *(obligatorio)*: `Input_Actividades_Recursos_Frentes.xlsx`
      Hojas: `Actividades` (secuencias y distribuciones de tiempo), `Recursos` (equipos disponibles con posibles cambios por turno),
      `Frentes` (frentes de trabajo con coordenadas Xi/Yi → Xf/Yf, sección de excavación y dirección).

    - **Archivo 2 – Restricciones** *(obligatorio)*: `Input_Restricciones.xlsx`
      Hojas: `FrenteHundimiento` (límites de hundimiento), `ObrasCiviles` (zonas protegidas), `Restriccion` (restricciones por turno como Polvorazo/PA FH).

    - **Archivo 3 – Vaciaderos** *(opcional)*: `Input_Marina.xlsx`
      Hoja: `Vaciaderos` — sitios de depósito de material (ubicación, nivel, capacidad máxima, tipo). Define dónde se puede acumular el material extraído.

    - **Archivo 4 – Muckpile** *(opcional)*: `Muckpile.xlsx`
      Estado actual del stockpile por nivel (capacidad ocupada vs. máxima). Permite verificar restricciones de almacenamiento antes de la simulación.
    """
)

col_file1, col_file2 = st.columns(2)

with col_file1:
    st.markdown("**Archivo 1 (Operación):** `Input_Actividades_Recursos_Frentes.xlsx`")
    archivo_main = st.file_uploader("Cargar archivo de Operación", type=['xlsx', 'xls'], key="file_main")

with col_file2:
    st.markdown("**Archivo 2 (Restricciones):** `Input_Restricciones.xlsx`")
    archivo_restricciones = st.file_uploader("Cargar archivo de Restricciones", type=['xlsx', 'xls'], key="file_restricciones")

col_file3, col_file4 = st.columns(2)
with col_file3:
    st.markdown("**Archivo 3 – Vaciaderos** *(opcional)*: `Input_Marina.xlsx`")
    archivo_marina = st.file_uploader("Cargar Input_Marina (Vaciaderos)", type=['xlsx', 'xls'], key="file_marina")

with col_file4:
    st.markdown("**Archivo 4 – Muckpile** *(opcional)*: `Muckpile.xlsx`")
    archivo_muckpile_file = st.file_uploader("Cargar Muckpile (Stockpile actual)", type=['xlsx', 'xls'], key="file_muckpile")

# Cargar archivos opcionales
if archivo_marina:
    df_marina_loaded, errs_marina, warns_marina = validar_marina(archivo_marina)
    if errs_marina:
        for e in errs_marina:
            st.error(e)
    else:
        st.session_state.df_marina = df_marina_loaded
        for w in warns_marina:
            st.warning(w)
        st.success(f"Input_Marina cargado: {len(df_marina_loaded)} vaciaderos encontrados.")

if archivo_muckpile_file:
    df_muckpile_loaded, errs_mp, warns_mp = validar_muckpile(archivo_muckpile_file)
    if errs_mp:
        for e in errs_mp:
            st.error(e)
    else:
        st.session_state.df_muckpile = df_muckpile_loaded
        for w in warns_mp:
            st.warning(w)
        st.success(f"Muckpile cargado: {len(df_muckpile_loaded)} registros de stockpile encontrados.")

if archivo_main and archivo_restricciones:
    try:
        datos_validados, errores_input, warnings_input = cargar_y_validar_inputs(archivo_main, archivo_restricciones)

        if errores_input:
            st.session_state.datos_cargados = False
            for error in errores_input:
                st.error(error)
        else:
            st.session_state.df_actividades = datos_validados['df_actividades']
            st.session_state.df_recursos = datos_validados['df_recursos']
            st.session_state.df_frentes = datos_validados['df_frentes']
            st.session_state.df_fh = datos_validados['df_fh']
            st.session_state.df_oc = datos_validados['df_oc']
            st.session_state.df_res = datos_validados['df_res']
            st.session_state.actividades_originales = st.session_state.df_actividades.copy()
            st.session_state.datos_cargados = True
            st.session_state.data_version += 1
            st.success("Datos cargados y validados correctamente.")
            for warning in warnings_input:
                st.warning(warning)

    except Exception as e:
        st.session_state.datos_cargados = False
        st.error(f"Error al cargar datos. Revise hojas, columnas y formatos. Detalle: {e}")

# ----------------------------------------------------
# 1a. Vista de Transparencia - Bases de Datos Cargadas (los 4 archivos, hoja por hoja)
# ----------------------------------------------------
_archivos_subidos = [
    ("Archivo 1 – Operación", archivo_main),
    ("Archivo 2 – Restricciones", archivo_restricciones),
    ("Archivo 3 – Vaciaderos (Input_Marina)", archivo_marina),
    ("Archivo 4 – Muckpile (Stockpile actual)", archivo_muckpile_file),
]
_archivos_disponibles = [(nombre, arch) for nombre, arch in _archivos_subidos if arch is not None]

if _archivos_disponibles:
    st.subheader("1a. Vista de Transparencia — Bases de Datos Cargadas")
    st.caption(
        "Revisión completa de los archivos Excel tal como fueron subidos, hoja por hoja, "
        "antes de cualquier validación o transformación del programa."
    )

    tabs_archivos = st.tabs([nombre for nombre, _ in _archivos_disponibles])

    for tab, (nombre, archivo) in zip(tabs_archivos, _archivos_disponibles):
        with tab:
            try:
                archivo.seek(0)
                excel_file_obj = pd.ExcelFile(archivo)
                nombres_hojas = excel_file_obj.sheet_names

                if not nombres_hojas:
                    st.info("Este archivo no contiene hojas legibles.")
                else:
                    st.caption(f"📄 `{archivo.name}` — {len(nombres_hojas)} hoja(s): {', '.join(nombres_hojas)}")
                    sub_tabs_hojas = st.tabs(nombres_hojas)
                    for sub_tab, hoja in zip(sub_tabs_hojas, nombres_hojas):
                        with sub_tab:
                            try:
                                df_hoja = excel_file_obj.parse(hoja)
                                st.dataframe(df_hoja, use_container_width=True, hide_index=True)
                                st.caption(f"{df_hoja.shape[0]} filas × {df_hoja.shape[1]} columnas.")
                            except Exception as e_hoja:
                                st.warning(f"No se pudo leer la hoja '{hoja}': {e_hoja}")
            except Exception as e_archivo:
                st.error(f"No se pudo leer el archivo '{archivo.name}': {e_archivo}")
            finally:
                try:
                    archivo.seek(0)
                except Exception:
                    pass

# ----------------------------------------------------
# 1b. Vista de Vaciaderos y Muckpile (si se cargaron)
# ----------------------------------------------------
if not st.session_state.df_marina.empty or not st.session_state.df_muckpile.empty:
    st.subheader("1b. Vaciaderos y Estado del Stockpile (Muckpile)")
    col_mar, col_mp = st.columns(2)

    with col_mar:
        if not st.session_state.df_marina.empty:
            st.markdown("**Vaciaderos (Input_Marina)**")
            if 'Capacidad' in st.session_state.df_marina.columns and 'Maximo' in st.session_state.df_marina.columns:
                total_cap = st.session_state.df_marina['Capacidad'].sum()
                total_max = st.session_state.df_marina['Maximo'].sum()
                st.metric("Capacidad Total", f"{total_cap:.0f}", delta=f"Máximo: {total_max:.0f}")
        else:
            st.info("No se cargó Input_Marina.")

    with col_mp:
        if not st.session_state.df_muckpile.empty:
            st.markdown("**Stockpile Actual (Muckpile)**")
            st.caption("Estado actual del material almacenado por nivel/ubicación.")
            if 'Capacidad Ocupada' in st.session_state.df_muckpile.columns and 'Maximo' in st.session_state.df_muckpile.columns:
                ocupado = st.session_state.df_muckpile['Capacidad Ocupada'].sum()
                maximo = st.session_state.df_muckpile['Maximo'].sum()
                pct = (ocupado / maximo * 100) if maximo > 0 else 0
                st.metric("Ocupación Total", f"{ocupado:.0f} / {maximo:.0f}", delta=f"{pct:.1f}% lleno")
                if pct > 80:
                    st.warning("⚠️ Stockpile sobre 80% de capacidad. Puede restringir operaciones de extracción.")
        else:
            st.info("No se cargó Muckpile.")

# ----------------------------------------------------
# 2. Seleccionar Nivel
# ----------------------------------------------------
if st.session_state.datos_cargados:
    st.subheader("2. Seleccionar Nivel")
    niveles_disponibles = st.session_state.df_actividades['Nivel'].unique()
    nivel_seleccionado = st.selectbox("Nivel", niveles_disponibles)

    if nivel_seleccionado:
        actividades_nivel = filtrar_actividades_nivel(
            st.session_state.df_actividades, nivel_seleccionado, st.session_state.data_version
        )
        recursos_nivel = filtrar_recursos_nivel(
            st.session_state.df_recursos, nivel_seleccionado, st.session_state.data_version
        )
        frentes_nivel = filtrar_frentes_nivel(
            st.session_state.df_frentes, nivel_seleccionado, st.session_state.data_version
        )

        # Aplicar modificaciones guardadas
        for idx, modificacion in st.session_state.actividades_modificadas.items():
            if idx in actividades_nivel.index:
                actividades_nivel.at[idx, 'Distribucion'] = modificacion['Distribucion']
                actividades_nivel.at[idx, 'Tiempo'] = modificacion['Tiempo']

        # --- INSERCIÓN: Agregar Actividades Personalizadas ---
        if 'Es_Personalizada' not in actividades_nivel.columns:
            actividades_nivel['Es_Personalizada'] = False
        if 'Frentes_Aplicables' not in actividades_nivel.columns:
            actividades_nivel['Frentes_Aplicables'] = None # Se asume aplicable a todos si es None/False
        
        if st.session_state.actividades_adicionales:
            df_nuevas_act = pd.DataFrame(st.session_state.actividades_adicionales)
            
            # Asegurar que las columnas existan en las nuevas actividades para el concat
            for col in actividades_nivel.columns:
                if col not in df_nuevas_act.columns:
                    df_nuevas_act[col] = np.nan
            
            # Unir actividades: las nuevas actividades tienen Secuencia 99
            actividades_nivel = pd.concat([actividades_nivel, df_nuevas_act], ignore_index=True)

        # --- FIN INSERCIÓN ---

        # ----------------------------------------------------
        # 3. Parametros Generales
        # ----------------------------------------------------
        # Toda la sección vive dentro de un @st.fragment: así, cambiar el sistema de
        # turnos, sincronizar turnos/horas, o presionar "Aplicar parámetros" solo
        # vuelve a ejecutar ESTE bloque (rápido) en vez de re-correr todo el script
        # (~4600 líneas, con llamadas a Redis, pandas y Plotly de las secciones de
        # abajo) en cada interacción. Eso es lo que antes hacía "recargar toda la
        # app" y se veía como si secciones como la 7 se vaciaran/reiniciaran.
        # Los valores finales se dejan en st.session_state para que el resto del
        # script (fuera del fragment) los siga leyendo con normalidad.
        @st.fragment
        def _render_parametros_generales():
            st.subheader("3. Parametros Generales")

            def _duracion_turno_actual():
                return 12.0 if '12x2' in st.session_state.get('sistema_turnos', '12x2 (2 turnos de 12h)') else 8.0

            def _sync_horas_desde_turnos():
                """Al cambiar 'Numero de Turnos del Programa', recalcula 'Tiempo Limite (horas)' = turnos x duracion del turno."""
                dur = _duracion_turno_actual()
                num = st.session_state.get('numero_turnos_input', 1)
                st.session_state['tiempo_limite_input'] = float(num * dur)

            def _sync_turnos_desde_horas():
                """Al cambiar 'Tiempo Limite (horas)', recalcula 'Numero de Turnos del Programa' = horas / duracion del turno."""
                dur = _duracion_turno_actual()
                tiempo = st.session_state.get('tiempo_limite_input', dur)
                st.session_state['numero_turnos_input'] = max(1, round(tiempo / dur))

            sistema_turnos = st.radio(
                "Sistema de Turnos", 
                ['12x2 (2 turnos de 12h)', '8x3 (3 turnos de 8h)'], 
                key="sistema_turnos", 
                horizontal=True,
                on_change=_sync_horas_desde_turnos  # al cambiar el sistema, se recalculan las horas manteniendo el N° de turnos
            )
            if '12x2' in sistema_turnos:
                turnos_info = "2 turnos de 12 horas."
                duracion_turno_base = 12.0
            else:
                turnos_info = "3 turnos de 8 horas."
                duracion_turno_base = 8.0
            st.info(turnos_info)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                numero_turnos = st.number_input(
                    "Numero de Turnos del Programa", min_value=1, value=62, step=1,
                    key="numero_turnos_input", on_change=_sync_horas_desde_turnos,
                    help="Al modificarlo, el 'Tiempo Limite (horas)' se recalcula automáticamente (N° turnos × duración del turno)."
                )
            with col_p2:
                tiempo_limite = st.number_input(
                    "Tiempo Limite (horas)",
                    min_value=1.0,
                    value=float(numero_turnos * duracion_turno_base),
                    step=1.0,
                    key="tiempo_limite_input",
                    on_change=_sync_turnos_desde_horas,
                    help="En el motor original equivale a numero de turnos x horas por turno. Al modificarlo, el "
                         "'Numero de Turnos del Programa' se recalcula automáticamente (horas ÷ duración del turno), "
                         "y viceversa: ambos campos quedan sincronizados."
                )

            # Estos 3 parámetros van en un form: no disparan rerun en cada tecla/click,
            # solo al presionar "Aplicar parámetros" — y al estar el form DENTRO del
            # fragment, ese submit tampoco dispara un rerun de toda la app, solo de
            # este fragment. Turnos/Tiempo Límite se dejan FUERA del form porque
            # dependen de on_change para sincronizarse en vivo entre sí (st.form no
            # permite ejecutar on_change hasta el submit).
            with st.form("form_parametros_generales"):
                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    metros_avance = st.number_input("Metros por Ciclo por Defecto", min_value=0.1, value=3.5, step=0.1, key="metros_avance_input")
                with col_p4:
                    n_simulaciones = st.number_input("Número de Simulaciones Monte Carlo", min_value=1, value=1000, step=100, key="n_simulaciones_input")

                ruta_critica = st.multiselect(
                    "Ruta critica / frentes prioritarios",
                    sorted(frentes_nivel['Frentes'].dropna().unique()),
                    default=[],
                    help="Replica la entrada R_critica del main.py original: estos frentes se priorizan antes de ordenar por cercania a FH/OOCC."
                )
                st.form_submit_button("Aplicar parámetros", type="secondary")

            seccion_col = columna_seccion(frentes_nivel)
            metros_por_seccion = {}
            if seccion_col:
                secciones = sorted(frentes_nivel[seccion_col].dropna().astype(str).unique())
                metros_por_seccion = {
                    seccion: float(metros_avance)
                    for seccion in secciones
                }

            st.session_state.tiempo_limite = tiempo_limite
            st.session_state.metros_avance = metros_avance
            st.session_state.n_simulaciones = n_simulaciones
            st.session_state.numero_turnos = numero_turnos
            st.session_state.ruta_critica = ruta_critica
            st.session_state.metros_por_seccion = metros_por_seccion
            st.session_state.duracion_turno_base = duracion_turno_base

        _render_parametros_generales()

        # ----------------------------------------------------
        # 4. Configuración de Recursos
        # ----------------------------------------------------
        st.subheader("4. Configuración de Recursos")
        recursos_config = {}
        velocidades_config = {}

        for _, row in recursos_nivel.iterrows():
            recurso = row['Recurso']
            col_r1, col_r2 = st.columns([1, 2])
            cantidad_base = float(row['Cantidad']) if pd.notna(row['Cantidad']) else 0.0
            cantidad_default = math.ceil(cantidad_base) if cantidad_base > 0 else 0
            
            with col_r1:
                st.markdown(f"**{recurso}**")
                cantidad = st.number_input(
                    f"Cantidad de {recurso}", 
                    min_value=0,
                    value=int(cantidad_default),
                    step=1,
                    key=f"recurso_qty_{recurso}",
                    help="Si el archivo trae cantidad fraccional, se redondea hacia arriba para esta simulacion discreta de equipos."
                )
                recursos_config[recurso] = {'cantidad': cantidad}
            
            with col_r2:
                st.markdown(f"**Velocidad de Viaje (km/h)**")
                dist_velocidad = st.selectbox(
                    f"Distribución de Velocidad para {recurso}",
                    ['Constante', 'normal', 'weibull', 'gamma', 'lognormal',
                     'fisk', 'kstwobign', 'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                     'burr', 'loglaplace', 'maxwell', 'nakagami'],
                    key=f"vel_dist_{recurso}"
                )

                if dist_velocidad == 'Constante':
                    valor = st.number_input(
                        f"Valor Constante (km/h) para {recurso}",
                        min_value=0.1, value=15.0, step=0.5, key=f"vel_cte_{recurso}"
                    )
                    velocidades_config[recurso] = {'dist': 'Constante', 'params': valor}
                elif dist_velocidad == 'normal':
                    col_vn1, col_vn2 = st.columns(2)
                    with col_vn1:
                        mean_val = st.number_input("mean", min_value=0.1, value=15.0, step=0.5, key=f"vel_mean_{recurso}")
                    with col_vn2:
                        std_val = st.number_input("std", min_value=0.01, value=3.0, step=0.1, key=f"vel_std_{recurso}")
                    velocidades_config[recurso] = {'dist': 'normal', 'params': {'mean': mean_val, 'std': std_val}}
                elif dist_velocidad == 'weibull':
                    col_vw1, col_vw2 = st.columns(2)
                    with col_vw1:
                        c_val = st.number_input("c", min_value=0.01, value=2.0, step=0.1, key=f"vel_weib_c_{recurso}")
                    with col_vw2:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_weib_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'weibull', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'gamma':
                    col_vg1, col_vg2 = st.columns(2)
                    with col_vg1:
                        a_val = st.number_input("a", min_value=0.01, value=2.0, step=0.1, key=f"vel_gamma_a_{recurso}")
                    with col_vg2:
                        scale_val = st.number_input("scale", min_value=0.01, value=7.5, step=0.5, key=f"vel_gamma_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'gamma', 'params': {'a': a_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'lognormal':
                    col_vl1, col_vl2 = st.columns(2)
                    with col_vl1:
                        s_val = st.number_input("s (sigma)", min_value=0.01, value=0.3, step=0.01, key=f"vel_logn_s_{recurso}")
                    with col_vl2:
                        scale_val = st.number_input("scale", min_value=0.01, value=14.0, step=0.5, key=f"vel_logn_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'lognormal', 'params': {'s': s_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'fisk':
                    col_vf1, col_vf2, col_vf3 = st.columns(3)
                    with col_vf1:
                        c_val = st.number_input("c", min_value=0.01, value=6.26, step=0.01, key=f"vel_fisk_c_{recurso}")
                    with col_vf2:
                        loc_val = st.number_input("loc", value=0.0, step=0.1, key=f"vel_fisk_loc_{recurso}")
                    with col_vf3:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_fisk_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'fisk', 'params': {'c': c_val, 'loc': loc_val, 'scale': scale_val}}
                elif dist_velocidad == 'kstwobign':
                    col_vk1, col_vk2 = st.columns(2)
                    with col_vk1:
                        loc_val = st.number_input("loc", value=0.0, step=0.1, key=f"vel_ks_loc_{recurso}")
                    with col_vk2:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_ks_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'kstwobign', 'params': {'loc': loc_val, 'scale': scale_val}}
                elif dist_velocidad == 'rayleigh':
                    col_vr1, col_vr2 = st.columns(2)
                    with col_vr1:
                        loc_val = st.number_input("loc", value=0.0, step=0.1, key=f"vel_ray_loc_{recurso}")
                    with col_vr2:
                        scale_val = st.number_input("scale", min_value=0.01, value=12.0, step=0.5, key=f"vel_ray_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'rayleigh', 'params': {'loc': loc_val, 'scale': scale_val}}
                elif dist_velocidad == 'foldcauchy':
                    col_vc1, col_vc2 = st.columns(2)
                    with col_vc1:
                        c_val = st.number_input("c", min_value=0.0, value=0.0, step=0.01, key=f"vel_fc_c_{recurso}")
                    with col_vc2:
                        scale_val = st.number_input("scale", min_value=0.01, value=10.0, step=0.5, key=f"vel_fc_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'foldcauchy', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'foldnorm':
                    col_vfn1, col_vfn2 = st.columns(2)
                    with col_vfn1:
                        c_val = st.number_input("c", min_value=0.0, value=0.0, step=0.01, key=f"vel_fn_c_{recurso}")
                    with col_vfn2:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_fn_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'foldnorm', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'ncx2':
                    col_vx1, col_vx2, col_vx3 = st.columns(3)
                    with col_vx1:
                        df_val = st.number_input("df (grados libertad)", min_value=1.0, value=2.0, step=1.0, key=f"vel_ncx2_df_{recurso}")
                    with col_vx2:
                        nc_val = st.number_input("nc (no-centralidad)", min_value=0.0, value=0.0, step=0.1, key=f"vel_ncx2_nc_{recurso}")
                    with col_vx3:
                        scale_val = st.number_input("scale", min_value=0.01, value=5.0, step=0.5, key=f"vel_ncx2_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'ncx2', 'params': {'df': df_val, 'nc': nc_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'burr':
                    col_vb1, col_vb2, col_vb3 = st.columns(3)
                    with col_vb1:
                        c_val = st.number_input("c", min_value=0.01, value=4.0, step=0.1, key=f"vel_burr_c_{recurso}")
                    with col_vb2:
                        d_val = st.number_input("d", min_value=0.01, value=2.0, step=0.1, key=f"vel_burr_d_{recurso}")
                    with col_vb3:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_burr_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'burr', 'params': {'c': c_val, 'd': d_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'loglaplace':
                    col_vll1, col_vll2 = st.columns(2)
                    with col_vll1:
                        c_val = st.number_input("c", min_value=0.01, value=2.0, step=0.1, key=f"vel_logl_c_{recurso}")
                    with col_vll2:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_logl_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'loglaplace', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'maxwell':
                    scale_val = st.number_input("scale", min_value=0.01, value=10.0, step=0.5, key=f"vel_maxwell_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'maxwell', 'params': {'loc': 0, 'scale': scale_val}}
                elif dist_velocidad == 'nakagami':
                    col_vnk1, col_vnk2 = st.columns(2)
                    with col_vnk1:
                        nu_val = st.number_input("nu", min_value=0.01, value=1.0, step=0.1, key=f"vel_naka_nu_{recurso}")
                    with col_vnk2:
                        scale_val = st.number_input("scale", min_value=0.01, value=15.0, step=0.5, key=f"vel_naka_scale_{recurso}")
                    velocidades_config[recurso] = {'dist': 'nakagami', 'params': {'nu': nu_val, 'loc': 0, 'scale': scale_val}}
        
        # Precarga de cambios de recursos por turno definidos en el archivo (columnas
        # Cambio / Turno_Cambio / Cantidad_Cambio), como punto de partida editable del
        # plan de orquestacion de flota por turno.
        if 'plan_recursos_turno' not in st.session_state:
            st.session_state.plan_recursos_turno = []
            for _, row in recursos_nivel.iterrows():
                if pd.notna(row.get('Cambio')) and str(row.get('Cambio')).strip().upper() in ('TRUE', '1', 'SI', 'SÍ', 'YES'):
                    if pd.notna(row.get('Turno_Cambio')) and pd.notna(row.get('Cantidad_Cambio')):
                        try:
                            st.session_state.plan_recursos_turno.append({
                                'recurso': row['Recurso'],
                                'turno': int(row['Turno_Cambio']),
                                'cantidad_nueva': int(row['Cantidad_Cambio']),
                                'demora_horas': 0.0
                            })
                        except (ValueError, TypeError):
                            pass

        st.session_state.recursos_config = recursos_config
        st.session_state.velocidades_config = velocidades_config

        # ----------------------------------------------------
        # 4b. Orquestacion de Flota por Turno
        # ----------------------------------------------------
        # Todo el Menu 4b vive dentro de un @st.fragment, igual que los Menus 3 y 12:
        # el form ya evitaba rerun tecla por tecla, pero al presionar "Añadir cambio al
        # plan" o los botones de eliminar/limpiar, disparaba un rerun de la app COMPLETA
        # (Menus 1 a 16). Encerrado en el fragment, esos botones solo vuelven a ejecutar
        # este bloque.
        @st.fragment
        def _menu_4b_orquestacion_flota(recursos_nivel):
            st.subheader("4b. Orquestación de Flota por Turno")
            st.caption(
                "Defina, recurso por recurso, cómo cambia su cantidad disponible a lo largo del programa de turnos "
                "(ej. bajar de 3 a 2 cuadrillas de carguío en el turno 50, subir a 9 en el turno 53). "
                "Estos cambios se aplican durante la simulación y quedan sujetos a las demás restricciones "
                "operacionales (fallas de equipos, restricciones geológicas, etc.): si en algún tramo un recurso "
                "queda en 0 unidades disponibles, no habrá avance en los frentes que dependan de él."
            )

            def _cantidad_vigente_antes_de_turno(recurso, turno, cantidad_base):
                """Cantidad que tendría el recurso justo antes de 'turno', segun los cambios ya
                configurados en el plan. Si no hay cambios previos, devuelve la cantidad base
                definida en el Menu 4. Se usa para saber a que cantidad "revertir" cuando el
                usuario define un cambio con turno de termino."""
                anteriores = [
                    c for c in st.session_state.plan_recursos_turno
                    if c['recurso'] == recurso and c['turno'] < turno
                ]
                if not anteriores:
                    return int(cantidad_base)
                ultimo = max(anteriores, key=lambda c: c['turno'])
                return int(ultimo['cantidad_nueva'])

            with st.expander("Añadir cambio de cantidad de un recurso en un turno"):
                # Los 2 checkboxes viven FUERA del form (a diferencia de "Aplicar demora al
                # cambio", que antes estaba adentro): dentro de un st.form, Streamlit no
                # re-ejecuta el script al tildar un checkbox (solo al presionar el boton de
                # submit), asi que el campo de horas de demora / turno de termino que dependen
                # de estos checkboxes se quedaban "congelados" en su estado anterior y no se
                # podian habilitar. Al vivir afuera, tildarlos S actualiza de inmediato (rerun
                # de este fragment) el resto del formulario que se arma justo debajo.
                col_chk1, col_chk2 = st.columns(2)
                with col_chk1:
                    cambio_temporal = st.checkbox(
                        "Cambio temporal (revertir automáticamente tras un turno de término)",
                        key="plan_cambio_temporal",
                        help="Ej: subir a 3 cuadrillas del turno 1 al turno 3 inclusive, y que desde "
                             "el turno 4 vuelva a la cantidad que tenía antes."
                    )
                with col_chk2:
                    con_demora = st.checkbox("Aplicar demora al cambio", key="plan_con_demora")

                with st.form("form_plan_recurso_turno", clear_on_submit=True):
                    col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
                    with col_pr1:
                        recurso_plan = st.selectbox(
                            "Recurso",
                            sorted(recursos_nivel['Recurso'].unique()),
                            key="plan_recurso_select"
                        )
                    with col_pr2:
                        turno_plan = st.number_input(
                            "Turno de inicio del cambio",
                            min_value=1,
                            value=1,
                            step=1,
                            key="plan_turno_input",
                            help="Numeración de turnos igual a la usada en 'Número de Turnos del Programa' (1-based)."
                        )
                    with col_pr3:
                        cantidad_plan = st.number_input(
                            "Cantidad nueva a partir de ese turno",
                            min_value=0,
                            value=1,
                            step=1,
                            key="plan_cantidad_input"
                        )
                    with col_pr4:
                        demora_plan = st.number_input(
                            "Horas de demora en hacer efectivo el cambio",
                            min_value=0.0,
                            value=0.0,
                            step=0.5,
                            key="plan_demora_input",
                            disabled=not con_demora,
                            help="Ej: si sube la cantidad de Jumbos, el/los equipo(s) nuevo(s) solo estarán disponibles "
                                 "'demora_horas' después del inicio del turno indicado. Si baja la cantidad, el/los "
                                 "equipo(s) retirado(s) dejan de estar disponibles 'demora_horas' después del inicio del turno."
                        )

                    turno_fin_plan = None
                    if cambio_temporal:
                        turno_fin_plan = st.number_input(
                            "Turno de término (inclusive) — desde el turno siguiente, el recurso vuelve "
                            "a la cantidad que tenía antes de este cambio",
                            min_value=int(turno_plan),
                            value=int(turno_plan),
                            step=1,
                            key="plan_turno_fin_input"
                        )

                    submitted_plan = st.form_submit_button("Añadir cambio al plan", type="primary")
                    if submitted_plan:
                        if cambio_temporal and turno_fin_plan < turno_plan:
                            st.error("El turno de término no puede ser anterior al turno de inicio del cambio.")
                        else:
                            cantidad_previa = _cantidad_vigente_antes_de_turno(
                                recurso_plan, int(turno_plan),
                                st.session_state.recursos_config.get(recurso_plan, {}).get('cantidad', 0)
                            )
                            st.session_state.plan_recursos_turno.append({
                                'recurso': recurso_plan,
                                'turno': int(turno_plan),
                                'cantidad_nueva': int(cantidad_plan),
                                'demora_horas': float(demora_plan) if con_demora else 0.0
                            })
                            mensaje = f"Cambio añadido: **{recurso_plan}** → {int(cantidad_plan)} unidades desde el turno {int(turno_plan)}"
                            if cambio_temporal:
                                st.session_state.plan_recursos_turno.append({
                                    'recurso': recurso_plan,
                                    'turno': int(turno_fin_plan) + 1,
                                    'cantidad_nueva': cantidad_previa,
                                    'demora_horas': 0.0,
                                    'auto_revertido': True
                                })
                                mensaje += (f" hasta el turno {int(turno_fin_plan)} (inclusive); "
                                            f"desde el turno {int(turno_fin_plan) + 1} vuelve a {cantidad_previa} unidades.")
                            else:
                                mensaje += "."
                            st.success(mensaje)

            if st.session_state.plan_recursos_turno:
                st.markdown("##### Plan de cambios configurado")
                df_plan = pd.DataFrame(st.session_state.plan_recursos_turno).sort_values(['recurso', 'turno']).reset_index(drop=True)
                df_plan_display = df_plan.copy()
                if 'auto_revertido' in df_plan_display.columns:
                    df_plan_display['Origen'] = df_plan_display['auto_revertido'].fillna(False).map(
                        {True: 'Automático (fin de rango)', False: 'Manual'}
                    )
                    df_plan_display = df_plan_display.drop(columns=['auto_revertido'])
                else:
                    df_plan_display['Origen'] = 'Manual'
                df_plan_display = df_plan_display.rename(columns={
                    'recurso': 'Recurso', 'turno': 'Turno de Cambio',
                    'cantidad_nueva': 'Cantidad Nueva', 'demora_horas': 'Demora Aplicada (h)'
                })
                st.dataframe(df_plan_display, use_container_width=True, hide_index=True)

                col_plan_del1, col_plan_del2 = st.columns([3, 1])
                with col_plan_del1:
                    idx_a_borrar = st.multiselect(
                        "Seleccionar fila(s) del plan para eliminar (por índice mostrado arriba, empezando en 0)",
                        list(range(len(df_plan))),
                        key="plan_idx_borrar"
                    )
                with col_plan_del2:
                    st.write("")
                    st.write("")
                    if st.button("Eliminar seleccionados", key="plan_borrar_btn") and idx_a_borrar:
                        filas_restantes = df_plan.drop(index=idx_a_borrar).to_dict('records')
                        st.session_state.plan_recursos_turno = filas_restantes
                        st.rerun()

                if st.button("Limpiar todo el plan de orquestación", key="plan_limpiar_btn"):
                    st.session_state.plan_recursos_turno = []
                    st.rerun()
            else:
                st.caption("No hay cambios de flota por turno configurados. Los recursos mantienen su cantidad base durante toda la simulación.")

            st.session_state.cambios_recursos = st.session_state.plan_recursos_turno

            # ----------------------------------------------------
            # Gráfico: cantidad disponible de cada recurso por turno
            # ----------------------------------------------------
            # Se arma desde el inicio con las cantidades base del Menu 4 (una línea plana
            # por recurso) y se actualiza solo, en vivo, con cada cambio que se añada/elimine
            # del plan de orquestación de arriba. El eje X va de 1 al 'Numero de Turnos del
            # Programa' definido en el Menu 3.
            st.markdown("##### Cantidad de cada recurso disponible, turno a turno")
            n_turnos_grafico = int(st.session_state.get('numero_turnos', 1))
            recursos_lista_grafico = sorted(recursos_nivel['Recurso'].unique())

            if n_turnos_grafico >= 1 and recursos_lista_grafico:
                turnos_eje_x = list(range(1, n_turnos_grafico + 1))
                fig_flota_turno = go.Figure()
                for recurso in recursos_lista_grafico:
                    cantidad_base_recurso = st.session_state.recursos_config.get(recurso, {}).get('cantidad', 0)
                    eventos_recurso = sorted(
                        [c for c in st.session_state.plan_recursos_turno if c['recurso'] == recurso],
                        key=lambda c: c['turno']
                    )
                    cantidad_por_turno = {}
                    for ev in eventos_recurso:
                        cantidad_por_turno[ev['turno']] = int(ev['cantidad_nueva'])

                    serie_y = []
                    cantidad_actual = int(cantidad_base_recurso)
                    for turno in turnos_eje_x:
                        if turno in cantidad_por_turno:
                            cantidad_actual = cantidad_por_turno[turno]
                        serie_y.append(cantidad_actual)

                    fig_flota_turno.add_trace(go.Scatter(
                        x=turnos_eje_x,
                        y=serie_y,
                        mode='lines',
                        name=recurso,
                        line=dict(shape='hv', width=2.5),
                        hovertemplate=f'<b>{recurso}</b><br>Turno %{{x}}<br>Cantidad: %{{y}}<extra></extra>'
                    ))

                fig_flota_turno.update_layout(
                    xaxis_title="Turno del Programa",
                    yaxis_title="Cantidad Disponible",
                    height=400,
                    hovermode='x unified',
                    legend_title_text="Recurso",
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                fig_flota_turno.update_yaxes(rangemode='tozero')
                st.plotly_chart(fig_flota_turno, use_container_width=True)
                st.caption(
                    "El gráfico ubica cada cambio en su turno de inicio (a nivel de turno completo); "
                    "una demora en horas dentro del mismo turno no se refleja en este escalón."
                )
            else:
                st.info("Defina el 'Número de Turnos del Programa' (Menú 3) para ver este gráfico.")

        _menu_4b_orquestacion_flota(recursos_nivel)

        # ----------------------------------------------------
        # 5. Modificar Distribuciones de Actividades
        # ----------------------------------------------------
        st.subheader("5. Modificar Distribuciones de Actividades")
        with st.expander("Distribuciones disponibles y sus parámetros (scipy.stats)"):
            usos_referencia = {
                'Cte': 'Valor constante', 'fisk': 'Perforación', 'kstwobign': 'Transporte/Acarreo',
                'rayleigh': 'Empotramiento', 'foldcauchy': 'Sostenimiento, Shotcrete',
                'foldnorm': 'Perforación Fin', 'ncx2': 'Instalación',
                'burr': 'colas pesadas, uso general', 'loglaplace': 'picos con colas asimétricas',
                'maxwell': 'velocidades, siempre positiva', 'nakagami': 'variabilidad de señal/desempeño',
                'normal': 'estándar', 'weibull': 'estándar', 'gamma': 'estándar', 'lognormal': 'estándar',
            }
            for nombre_d, info_d in DISTRIBUTION_PARAM_INFO.items():
                params_str = ", ".join(p['label'] for p in info_d['parametros'])
                uso = usos_referencia.get(nombre_d, '')
                st.markdown(f"- **{nombre_d}** ({uso}): {params_str}")

        actividades_lista = actividades_nivel[actividades_nivel['Actividad'].notna() & (actividades_nivel['Es_Personalizada'] == False)].copy()

        if 'actividades_seleccionadas' not in st.session_state:
            st.session_state.actividades_seleccionadas = []

        # ---- Filtros de tramo (para optimizacion: "solo este trazo",
        # "de esta secuencia en adelante", "hasta tal parte", "solo tal distribucion", etc.) ----
        st.markdown("##### Filtrar Actividades (por tramo de Secuencia y/o Distribución):")
        secuencias_disponibles = sorted(actividades_lista['Secuencia'].dropna().unique().tolist())
        distribuciones_presentes = sorted(actividades_lista['Distribucion'].dropna().unique().tolist())

        col_f1, col_f2, col_f3 = st.columns([1.2, 1.5, 1.5])
        with col_f1:
            modo_filtro_seq = st.selectbox(
                "Condición sobre la Secuencia",
                ["Sin filtro", "Igual a", "Mayor a (>)", "Mayor o igual (>=)",
                 "Menor a (<)", "Menor o igual (<=)", "Entre (rango)"],
                key="modo_filtro_secuencia"
            )
        with col_f2:
            if modo_filtro_seq == "Entre (rango)" and secuencias_disponibles:
                seq_min_sel, seq_max_sel = st.select_slider(
                    "Rango de Secuencia",
                    options=secuencias_disponibles,
                    value=(secuencias_disponibles[0], secuencias_disponibles[-1]),
                    key="rango_secuencia_filtro"
                )
            elif modo_filtro_seq != "Sin filtro" and secuencias_disponibles:
                seq_valor_sel = st.selectbox(
                    "Valor de Secuencia",
                    secuencias_disponibles,
                    key="valor_secuencia_filtro"
                )
            else:
                st.caption("Sin filtro de secuencia aplicado.")
        with col_f3:
            dist_filtro_sel = st.multiselect(
                "Filtrar por Distribución (vacío = todas)",
                distribuciones_presentes,
                default=[],
                key="dist_filtro_multiselect"
            )

        actividades_filtradas = actividades_lista.copy()
        if modo_filtro_seq == "Igual a" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Secuencia'] == seq_valor_sel]
        elif modo_filtro_seq == "Mayor a (>)" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Secuencia'] > seq_valor_sel]
        elif modo_filtro_seq == "Mayor o igual (>=)" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Secuencia'] >= seq_valor_sel]
        elif modo_filtro_seq == "Menor a (<)" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Secuencia'] < seq_valor_sel]
        elif modo_filtro_seq == "Menor o igual (<=)" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Secuencia'] <= seq_valor_sel]
        elif modo_filtro_seq == "Entre (rango)" and secuencias_disponibles:
            actividades_filtradas = actividades_filtradas[
                (actividades_filtradas['Secuencia'] >= seq_min_sel) & (actividades_filtradas['Secuencia'] <= seq_max_sel)
            ]
        if dist_filtro_sel:
            actividades_filtradas = actividades_filtradas[actividades_filtradas['Distribucion'].isin(dist_filtro_sel)]

        hay_filtro_activo = (modo_filtro_seq != "Sin filtro") or bool(dist_filtro_sel)

        # Botones de selección masiva: sobre TODAS las actividades, o solo sobre el tramo filtrado
        col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 1, 1.3, 2.7])
        with col_sel1:
            if st.button("Seleccionar Todo", use_container_width=True):
                st.session_state.actividades_seleccionadas = actividades_lista.index.tolist()
                st.rerun()
        with col_sel2:
            if st.button("Deseleccionar Todo", use_container_width=True):
                st.session_state.actividades_seleccionadas = []
                st.rerun()
        with col_sel3:
            if st.button("Seleccionar Filtrados", use_container_width=True, disabled=not hay_filtro_activo):
                st.session_state.actividades_seleccionadas = actividades_filtradas.index.tolist()
                st.rerun()

        tabla_a_mostrar = actividades_filtradas if hay_filtro_activo else actividades_lista
        actividades_para_mostrar = tabla_a_mostrar.copy()
        actividades_para_mostrar['Seleccionado'] = actividades_para_mostrar.index.isin(st.session_state.actividades_seleccionadas)

        if hay_filtro_activo:
            st.markdown(f"**Actividades Filtradas ({len(tabla_a_mostrar)} de {len(actividades_lista)} originales):**")
        else:
            st.markdown(f"**Actividades Originales ({len(actividades_lista)}):**")
        df_display = actividades_para_mostrar[['Actividad', 'Recurso', 'Secuencia', 'Distribucion', 'Tiempo', 'Seleccionado']]

        edited_df = st.data_editor(
            df_display,
            column_config={
                "Seleccionado": st.column_config.CheckboxColumn("Seleccionado", help="Seleccionar para modificar", default=False)
            },
            hide_index=True,
            use_container_width=True,
            key="actividades_mod_table"
        )

        # Las filas visibles (posiblemente filtradas) actualizan su estado de selección;
        # las filas fuera del filtro actual conservan la selección que ya tenían.
        seleccion_previa = set(st.session_state.actividades_seleccionadas)
        seleccion_en_vista = set(edited_df[edited_df['Seleccionado']].index.tolist())
        indices_fuera_de_vista = seleccion_previa - set(df_display.index.tolist())
        st.session_state.actividades_seleccionadas = list(indices_fuera_de_vista | seleccion_en_vista)

        if st.session_state.actividades_seleccionadas:
            st.markdown("##### Configuración de la Modificación:")
            
            actividad_ejemplo_idx = st.session_state.actividades_seleccionadas[0]
            actividad_ejemplo = actividades_lista.loc[actividad_ejemplo_idx]['Actividad']
            
            st.info(f"Configurando modificación para {len(st.session_state.actividades_seleccionadas)} actividades (Ej. **{actividad_ejemplo}**).")
            
            col_mod1, col_mod2 = st.columns(2)
            with col_mod1:
                nueva_dist = st.selectbox(
                    "Nueva Distribución de Tiempo",
                    ['Cte', 'normal', 'weibull', 'gamma', 'lognormal', 'fisk', 'kstwobign',
                     'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                     'burr', 'loglaplace', 'maxwell', 'nakagami'],
                    key="nueva_dist"
                )

            with col_mod2:
                # Los inputs de cada parametro se generan automaticamente a partir de
                # DISTRIBUTION_PARAM_INFO, usando los nombres reales de scipy.stats
                # (ej. gamma usa 'a' y 'scale', no "parametro 1 y 2").
                info_dist = DISTRIBUTION_PARAM_INFO[nueva_dist]
                valores_ingresados = {}
                for p in info_dist['parametros']:
                    valores_ingresados[p['key']] = st.number_input(
                        p['label'],
                        min_value=p.get('min'),
                        value=p['default'],
                        step=0.01 if p.get('min') == 0.0001 else 0.1,
                        key=f"param_{nueva_dist}_{p['key']}"
                    )

                if nueva_dist == 'Cte':
                    nuevos_params = valores_ingresados['valor']
                else:
                    nuevos_params = valores_ingresados

            # ---- Vista previa grafica de la PDF (punto 2: validar visualmente el cambio) ----
            st.markdown("###### Vista Previa de la Distribución (PDF):")
            if nueva_dist == 'Cte':
                st.info(f"Distribución constante: siempre entrega **{nuevos_params:.3f} h**. No aplica curva de densidad.")
            else:
                try:
                    # Solo se arma el kwargs de la distribucion ACTUALMENTE seleccionada.
                    # (Antes se armaba un diccionario con las 14 distribuciones de una vez,
                    # lo que evaluaba nuevos_params['mean'] aunque la distribucion elegida
                    # fuera 'burr', reventando con KeyError: 'mean'.)
                    dist_kwargs_por_nombre = {
                        'normal': lambda: {'loc': nuevos_params['mean'], 'scale': nuevos_params['std']},
                        'weibull': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'gamma': lambda: {'a': nuevos_params['a'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'lognormal': lambda: {'s': nuevos_params['s'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'fisk': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'kstwobign': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'rayleigh': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'foldcauchy': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'foldnorm': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'ncx2': lambda: {'df': nuevos_params['df'], 'nc': nuevos_params['nc'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'burr': lambda: {'c': nuevos_params['c'], 'd': nuevos_params['d'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'loglaplace': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'maxwell': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        'nakagami': lambda: {'nu': nuevos_params['nu'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                    }
                    dist_obj_por_nombre = {
                        'normal': stats.norm, 'weibull': stats.weibull_min, 'gamma': stats.gamma,
                        'lognormal': stats.lognorm, 'fisk': stats.fisk, 'kstwobign': stats.kstwobign,
                        'rayleigh': stats.rayleigh, 'foldcauchy': stats.foldcauchy, 'foldnorm': stats.foldnorm,
                        'ncx2': stats.ncx2, 'burr': stats.burr, 'loglaplace': stats.loglaplace,
                        'maxwell': stats.maxwell, 'nakagami': stats.nakagami,
                    }
                    dist_obj = dist_obj_por_nombre[nueva_dist]
                    dist_kwargs = dist_kwargs_por_nombre[nueva_dist]()

                    # Rango de x centrado en la forma de la distribucion elegida, usando
                    # los percentiles 0.1% y 99.9% de la propia distribucion (ppf) para
                    # no adivinar limites a mano.
                    x_lo = dist_obj.ppf(0.001, **dist_kwargs)
                    x_hi = dist_obj.ppf(0.999, **dist_kwargs)
                    if not np.isfinite(x_lo):
                        x_lo = 0.0
                    if not np.isfinite(x_hi):
                        x_hi = x_lo + 10.0
                    if x_hi <= x_lo:
                        x_hi = x_lo + 1.0
                    x_vals_preview = np.linspace(x_lo, x_hi, 400)
                    y_vals_preview = calcular_pdf_teorica(nuevos_params, nueva_dist, x_vals_preview)

                    fig_preview = go.Figure()
                    fig_preview.add_trace(go.Scatter(
                        x=x_vals_preview, y=y_vals_preview,
                        mode='lines', fill='tozeroy',
                        line=dict(color='#1f77b4', width=2),
                        name=nueva_dist
                    ))

                    # ---- Filtro de rango en X sobre la propia curva (punto 3: "que pasa
                    # si solo miro los tiempos mayores/menores/entre tales valores", para
                    # fines de optimizacion). Sombrea el area correspondiente y calcula su
                    # probabilidad exacta con la CDF, sin aproximar con el histograma. ----
                    col_area1, col_area2 = st.columns([1, 2])
                    with col_area1:
                        modo_area = st.selectbox(
                            "Área bajo la curva a resaltar",
                            ["Ninguna", "Mayor a (>)", "Menor a (<)", "Entre (rango)"],
                            key=f"modo_area_{nueva_dist}"
                        )
                    with col_area2:
                        x_area_lo, x_area_hi = None, None
                        if modo_area == "Mayor a (>)":
                            x_area_lo = st.slider("Valor de corte (h)", float(x_lo), float(x_hi),
                                                   value=float((x_lo + x_hi) / 2), key=f"corte_mayor_{nueva_dist}")
                            x_area_hi = x_hi
                        elif modo_area == "Menor a (<)":
                            x_area_hi = st.slider("Valor de corte (h)", float(x_lo), float(x_hi),
                                                   value=float((x_lo + x_hi) / 2), key=f"corte_menor_{nueva_dist}")
                            x_area_lo = x_lo
                        elif modo_area == "Entre (rango)":
                            x_area_lo, x_area_hi = st.slider(
                                "Rango (h)", float(x_lo), float(x_hi),
                                value=(float(x_lo), float(x_hi)), key=f"corte_rango_{nueva_dist}"
                            )

                    if modo_area != "Ninguna" and x_area_lo is not None:
                        mascara_area = (x_vals_preview >= x_area_lo) & (x_vals_preview <= x_area_hi)
                        fig_preview.add_trace(go.Scatter(
                            x=x_vals_preview[mascara_area], y=y_vals_preview[mascara_area],
                            mode='lines', fill='tozeroy',
                            line=dict(width=0), fillcolor='rgba(255,99,71,0.55)',
                            name='Área seleccionada'
                        ))
                        prob_area = dist_obj.cdf(x_area_hi, **dist_kwargs) - dist_obj.cdf(x_area_lo, **dist_kwargs)
                        st.caption(f"Probabilidad de que el tiempo caiga en [{x_area_lo:.3f}, {x_area_hi:.3f}] h: **{prob_area*100:.2f}%** del área total.")

                    fig_preview.update_layout(
                        height=280,
                        margin=dict(l=40, r=20, t=20, b=40),
                        xaxis_title="Tiempo (h)",
                        yaxis_title="Densidad de probabilidad",
                        showlegend=False
                    )
                    st.plotly_chart(fig_preview, use_container_width=True, key="preview_pdf_nueva_dist")
                except Exception as e:
                    st.warning(f"No fue posible graficar la vista previa con los parámetros actuales: {e}")

            if st.button("Aplicar Modificación de Distribución", type="secondary"):
                if nuevos_params is not None:
                    # Si el usuario dejo seleccionada un area (Mayor a / Menor a / Entre)
                    # en la vista previa, ese rango se guarda como truncamiento real de la
                    # distribucion: '_trunc_lo'/'_trunc_hi' dentro de los propios parametros.
                    # generar_tiempos_batch() y calcular_pdf_teorica() ya saben leer estas
                    # claves e invertir la CDF restringida a ese tramo, asi que los tiempos
                    # que salgan de la simulacion quedan garantizados dentro del rango
                    # (nada de valores en 0, 1 o fuera de [2, 3] si se filtro "Entre 2 y 3").
                    # 'Cte' no es un dict (es un float) y tampoco tiene sentido truncarla:
                    # una constante ya es un unico valor fijo, sin distribucion que recortar.
                    if isinstance(nuevos_params, dict):
                        modo_area_actual = st.session_state.get(f"modo_area_{nueva_dist}", "Ninguna")
                        # x_lo/x_hi solo existen como variable local si la vista previa se
                        # armo sin excepciones; si algo fallo arriba, se cae a los propios
                        # sliders (que ya traen su rango real guardado por Streamlit).
                        x_lo_local = locals().get('x_lo')
                        x_hi_local = locals().get('x_hi')
                        if modo_area_actual == "Mayor a (>)":
                            corte_mayor = st.session_state.get(f"corte_mayor_{nueva_dist}")
                            if corte_mayor is not None and x_hi_local is not None:
                                nuevos_params['_trunc_lo'] = float(corte_mayor)
                                nuevos_params['_trunc_hi'] = float(x_hi_local)
                        elif modo_area_actual == "Menor a (<)":
                            corte_menor = st.session_state.get(f"corte_menor_{nueva_dist}")
                            if corte_menor is not None and x_lo_local is not None:
                                nuevos_params['_trunc_lo'] = float(x_lo_local)
                                nuevos_params['_trunc_hi'] = float(corte_menor)
                        elif modo_area_actual == "Entre (rango)":
                            rango_actual = st.session_state.get(f"corte_rango_{nueva_dist}")
                            if rango_actual is not None:
                                nuevos_params['_trunc_lo'] = float(rango_actual[0])
                                nuevos_params['_trunc_hi'] = float(rango_actual[1])
                        else:
                            nuevos_params.pop('_trunc_lo', None)
                            nuevos_params.pop('_trunc_hi', None)

                    for idx in st.session_state.actividades_seleccionadas:
                        st.session_state.actividades_modificadas[idx] = {
                            'Distribucion': nueva_dist,
                            'Tiempo': nuevos_params
                        }
                    # El cambio se aplica de inmediato: se fuerza un rerun para que el
                    # bloque "2. Seleccionar Nivel" (que lee actividades_modificadas en
                    # cada ejecucion) refresque la tabla ya con la nueva distribucion,
                    # sin que el usuario tenga que volver a tocar el selector de Nivel.
                    if isinstance(nuevos_params, dict) and '_trunc_lo' in nuevos_params:
                        st.success(
                            f"Distribución '{nueva_dist}' aplicada a {len(st.session_state.actividades_seleccionadas)} "
                            f"actividades, truncada al rango [{nuevos_params['_trunc_lo']:.3f}, {nuevos_params['_trunc_hi']:.3f}] h."
                        )
                    else:
                        st.success(f"Distribución '{nueva_dist}' aplicada a {len(st.session_state.actividades_seleccionadas)} actividades.")
                    st.rerun()
                else:
                    st.error("Error: Los parámetros de distribución no son válidos.")

        # frentes_disponibles se calcula una sola vez aqui y se reutiliza en los Menus
        # 6, 7 y 8 (antes se recalculaba dentro del Menu 6 y se leia "prestado" desde ahi).
        frentes_disponibles = frentes_nivel['Frentes'].unique()

        # ----------------------------------------------------
        # 6. Agregar Actividades Personalizadas (Sección 99)
        # ----------------------------------------------------
        # Todo el Menu 6 vive dentro de un @st.fragment, igual que los Menus 3, 4b y 12:
        # el form ya evitaba rerun tecla por tecla, pero "Añadir Actividad" o "Limpiar
        # Actividades Personalizadas" disparaban un rerun de la app COMPLETA. Encerrado en
        # el fragment, esos botones solo vuelven a ejecutar este bloque.
        #
        # Nota: se dejo de usar st.form aqui. Un form congela sus widgets hasta el submit,
        # por lo que la vista previa de la PDF y el filtro de area (que necesitan
        # reaccionar en cada cambio de parametro, igual que en el Menu 3) no podian
        # actualizarse en vivo dentro de uno. Al vivir sueltos dentro del fragment, cada
        # widget dispara solo un rerun de este bloque (no de la app completa) y la
        # vista previa queda sincronizada con lo que el usuario esta tocando.
        @st.fragment
        def _menu_6_actividades_personalizadas(recursos_nivel, frentes_disponibles, nivel_seleccionado):
            st.subheader("6. Agregar Actividades Personalizadas (Secuencia 99)")

            distribuciones_disponibles = ['Cte', 'normal', 'weibull', 'gamma', 'lognormal', 'fisk', 'kstwobign',
                                          'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                                          'burr', 'loglaplace', 'maxwell', 'nakagami']

            recursos_disponibles_act = ['N/A'] + list(recursos_nivel['Recurso'].unique())

            # El campo de nombre se limpia aqui, ANTES de instanciar el widget.
            # Streamlit no permite modificar st.session_state[key] una vez que
            # el widget con esa key ya fue creado en el run actual, por eso el
            # boton "Añadir Actividad" solo deja una bandera (_reset_nueva_act_nombre)
            # y es este bloque, ejecutado antes del st.text_input, el que la aplica.
            if st.session_state.get("_reset_nueva_act_nombre", False):
                st.session_state["nueva_act_nombre"] = ""
                st.session_state["_reset_nueva_act_nombre"] = False

            with st.expander("Añadir Nueva Actividad Personalizada"):
                col_n1, col_n2, col_n3 = st.columns(3)
                with col_n1:
                    nombre_act = st.text_input("Nombre de la Actividad", key="nueva_act_nombre")
                    recurso_act = st.selectbox(
                        "Recurso Requerido (N/A = Sin Recurso)",
                        recursos_disponibles_act,
                        key="nueva_act_recurso"
                    )
                with col_n2:
                    distribucion_act = st.selectbox(
                        "Tipo de Distribución",
                        distribuciones_disponibles,
                        key="nueva_act_distribucion"
                    )

                    frentes_seleccionados_act = st.multiselect(
                        "Túneles (Frentes) Aplicables",
                        frentes_disponibles,
                        default=frentes_disponibles,
                        key="nueva_act_frentes"
                    )

                with col_n3:
                    st.markdown("##### Parámetros de Tiempo (Horas)")
                    # Misma fuente unica que el Menu 3 (DISTRIBUTION_PARAM_INFO): asi los
                    # inputs de cada distribucion se generan con los nombres reales de
                    # scipy.stats en vez de reescribir 14 ramas "a mano".
                    info_dist_act = DISTRIBUTION_PARAM_INFO[distribucion_act]
                    valores_ingresados_act = {}
                    for p in info_dist_act['parametros']:
                        valores_ingresados_act[p['key']] = st.number_input(
                            p['label'],
                            min_value=p.get('min'),
                            value=p['default'],
                            step=0.01 if p.get('min') == 0.0001 else 0.1,
                            key=f"param_add_{distribucion_act}_{p['key']}"
                        )

                    if distribucion_act == 'Cte':
                        params_act = valores_ingresados_act['valor']
                    else:
                        params_act = valores_ingresados_act

                # ---- Vista previa de la PDF + filtro de area, igual que en el Menu 3 ----
                x_lo_act, x_hi_act = None, None
                if distribucion_act == 'Cte':
                    st.info(f"Distribución constante: siempre entrega **{params_act:.3f} h**. No aplica curva de densidad.")
                else:
                    try:
                        dist_obj_act = DIST_OBJ_SCIPY[distribucion_act]
                        dist_kwargs_act = _kwargs_scipy_desde_params(params_act, distribucion_act)

                        x_lo_act = dist_obj_act.ppf(0.001, **dist_kwargs_act)
                        x_hi_act = dist_obj_act.ppf(0.999, **dist_kwargs_act)
                        if not np.isfinite(x_lo_act):
                            x_lo_act = 0.0
                        if not np.isfinite(x_hi_act):
                            x_hi_act = x_lo_act + 10.0
                        if x_hi_act <= x_lo_act:
                            x_hi_act = x_lo_act + 1.0
                        x_vals_act = np.linspace(x_lo_act, x_hi_act, 400)
                        y_vals_act = calcular_pdf_teorica(params_act, distribucion_act, x_vals_act)

                        fig_preview_act = go.Figure()
                        fig_preview_act.add_trace(go.Scatter(
                            x=x_vals_act, y=y_vals_act,
                            mode='lines', fill='tozeroy',
                            line=dict(color='#1f77b4', width=2),
                            name=distribucion_act
                        ))

                        st.markdown("###### Vista Previa de la Distribución (PDF):")
                        col_area_act1, col_area_act2 = st.columns([1, 2])
                        with col_area_act1:
                            modo_area_act = st.selectbox(
                                "Área bajo la curva a resaltar",
                                ["Ninguna", "Mayor a (>)", "Menor a (<)", "Entre (rango)"],
                                key=f"modo_area_add_{distribucion_act}"
                            )
                        with col_area_act2:
                            x_area_lo_act, x_area_hi_act = None, None
                            if modo_area_act == "Mayor a (>)":
                                x_area_lo_act = st.slider("Valor de corte (h)", float(x_lo_act), float(x_hi_act),
                                                           value=float((x_lo_act + x_hi_act) / 2), key=f"corte_mayor_add_{distribucion_act}")
                                x_area_hi_act = x_hi_act
                            elif modo_area_act == "Menor a (<)":
                                x_area_hi_act = st.slider("Valor de corte (h)", float(x_lo_act), float(x_hi_act),
                                                           value=float((x_lo_act + x_hi_act) / 2), key=f"corte_menor_add_{distribucion_act}")
                                x_area_lo_act = x_lo_act
                            elif modo_area_act == "Entre (rango)":
                                x_area_lo_act, x_area_hi_act = st.slider(
                                    "Rango (h)", float(x_lo_act), float(x_hi_act),
                                    value=(float(x_lo_act), float(x_hi_act)), key=f"corte_rango_add_{distribucion_act}"
                                )

                        if modo_area_act != "Ninguna" and x_area_lo_act is not None:
                            mascara_area_act = (x_vals_act >= x_area_lo_act) & (x_vals_act <= x_area_hi_act)
                            fig_preview_act.add_trace(go.Scatter(
                                x=x_vals_act[mascara_area_act], y=y_vals_act[mascara_area_act],
                                mode='lines', fill='tozeroy',
                                line=dict(width=0), fillcolor='rgba(255,99,71,0.55)',
                                name='Área seleccionada'
                            ))
                            prob_area_act = dist_obj_act.cdf(x_area_hi_act, **dist_kwargs_act) - dist_obj_act.cdf(x_area_lo_act, **dist_kwargs_act)
                            st.caption(f"Probabilidad de que el tiempo caiga en [{x_area_lo_act:.3f}, {x_area_hi_act:.3f}] h: **{prob_area_act*100:.2f}%** del área total.")

                        fig_preview_act.update_layout(
                            height=280,
                            margin=dict(l=40, r=20, t=20, b=40),
                            xaxis_title="Tiempo (h)",
                            yaxis_title="Densidad de probabilidad",
                            showlegend=False
                        )
                        st.plotly_chart(fig_preview_act, use_container_width=True, key="preview_pdf_nueva_actividad")
                    except Exception as e:
                        st.warning(f"No fue posible graficar la vista previa con los parámetros actuales: {e}")

                if st.button("Añadir Actividad", type="primary", key="btn_add_actividad"):
                    if nombre_act and frentes_seleccionados_act and params_act is not None:
                        # Igual que en el Menu 3: si se dejo seleccionada un area, ese
                        # rango se guarda como truncamiento real ('_trunc_lo'/'_trunc_hi')
                        # dentro de los propios parametros, no solo como indicador visual.
                        if isinstance(params_act, dict):
                            modo_area_actual_act = st.session_state.get(f"modo_area_add_{distribucion_act}", "Ninguna")
                            if modo_area_actual_act == "Mayor a (>)":
                                corte_mayor_act = st.session_state.get(f"corte_mayor_add_{distribucion_act}")
                                if corte_mayor_act is not None and x_hi_act is not None:
                                    params_act['_trunc_lo'] = float(corte_mayor_act)
                                    params_act['_trunc_hi'] = float(x_hi_act)
                            elif modo_area_actual_act == "Menor a (<)":
                                corte_menor_act = st.session_state.get(f"corte_menor_add_{distribucion_act}")
                                if corte_menor_act is not None and x_lo_act is not None:
                                    params_act['_trunc_lo'] = float(x_lo_act)
                                    params_act['_trunc_hi'] = float(corte_menor_act)
                            elif modo_area_actual_act == "Entre (rango)":
                                rango_actual_act = st.session_state.get(f"corte_rango_add_{distribucion_act}")
                                if rango_actual_act is not None:
                                    params_act['_trunc_lo'] = float(rango_actual_act[0])
                                    params_act['_trunc_hi'] = float(rango_actual_act[1])
                            else:
                                params_act.pop('_trunc_lo', None)
                                params_act.pop('_trunc_hi', None)

                        nueva_actividad_data = {
                            'Nivel': nivel_seleccionado,
                            'Secuencia': 99,
                            'Actividad': nombre_act.strip(),
                            'Recurso': recurso_act if recurso_act != 'N/A' else np.nan,
                            'Tiempo': params_act,
                            'Distribucion': distribucion_act,
                            'Frentes_Aplicables': frentes_seleccionados_act,
                            'Es_Personalizada': True
                        }

                        found = False
                        for i, act in enumerate(st.session_state.actividades_adicionales):
                            if act['Actividad'] == nombre_act.strip():
                                st.session_state.actividades_adicionales[i] = nueva_actividad_data
                                found = True
                                break

                        if not found:
                            st.session_state.actividades_adicionales.append(nueva_actividad_data)

                        # El cambio se aplica de inmediato (igual que en el Menu 3): se
                        # fuerza un rerun para que el resto de la app quede sincronizado
                        # sin que el usuario tenga que volver a tocar el Paso 2 a mano.
                        # Sin st.form ya no hay clear_on_submit automatico: se limpia el
                        # nombre a mano para que el campo quede listo para la siguiente
                        # actividad (los demas widgets no necesitan limpiarse: recurso y
                        # distribucion tiene sentido que mantengan el ultimo valor usado).
                        st.session_state["_reset_nueva_act_nombre"] = True
                        st.success(f"Actividad '{nombre_act}' añadida/actualizada correctamente.")
                        st.rerun()
                    else:
                        st.error("Por favor, ingrese el nombre, los parámetros de tiempo y seleccione al menos un túnel aplicable.")

            if st.session_state.actividades_adicionales:
                st.markdown("##### Actividades Personalizadas Agregadas:")
                df_adicionales = pd.DataFrame(st.session_state.actividades_adicionales)
                df_adicionales_display = df_adicionales[['Actividad', 'Recurso', 'Distribucion', 'Tiempo', 'Frentes_Aplicables']].copy()
                st.dataframe(df_adicionales_display, use_container_width=True, hide_index=True)

                col_del1, col_del2 = st.columns([2, 1])
                with col_del1:
                    nombres_actividades_act = [act['Actividad'] for act in st.session_state.actividades_adicionales]
                    actividad_a_eliminar = st.selectbox(
                        "Seleccionar actividad a eliminar",
                        nombres_actividades_act,
                        key="select_actividad_eliminar"
                    )
                    if st.button("Eliminar actividad seleccionada", key="eliminar_una_adicional"):
                        st.session_state.actividades_adicionales = [
                            act for act in st.session_state.actividades_adicionales
                            if act['Actividad'] != actividad_a_eliminar
                        ]
                        st.success(f"Actividad '{actividad_a_eliminar}' eliminada correctamente.")
                        st.rerun()
                with col_del2:
                    st.markdown("&nbsp;")
                    if st.button("Limpiar Actividades Personalizadas", key="limpiar_adicionales"):
                        st.session_state.actividades_adicionales = []
                        st.rerun()

        _menu_6_actividades_personalizadas(recursos_nivel, frentes_disponibles, nivel_seleccionado)

        # ----------------------------------------------------
        # 7. Configuración de Restricciones Geológicas
        # ----------------------------------------------------
        # Todo el Menu 7 vive dentro de un @st.fragment, igual que los Menus 3, 4b, 6 y 12:
        # "Guardar Restricción Geológica" y "Limpiar Todas las Restricciones Geológicas"
        # disparaban un rerun de la app COMPLETA. Encerrado en el fragment, esos botones
        # solo vuelven a ejecutar este bloque.
        @st.fragment
        def _menu_7_restricciones_geologicas(recursos_nivel, frentes_disponibles):
            st.subheader("7. Configuración de Restricciones Geológicas")

            with st.expander("Añadir/Editar Restricción Geológica"):
                with st.form("form_res_geo", clear_on_submit=True):
                    st.markdown("##### Definir Restricción")
                    
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        res_geo_nombre = st.text_input("Nombre de la Restricción (único)", key="res_geo_nombre")
                        res_geo_frentes = st.multiselect(
                            "Túneles (Frentes) Afectados",
                            frentes_disponibles,
                            key="res_geo_frentes"
                        )
                    with col_g2:
                        res_geo_avance = st.number_input(
                            "Castigo/Bonus a Metros por Ciclo (%)",
                            help="Ej: -50 para un 50% menos de avance. +10 para un 10% más.",
                            value=0,
                            step=5,
                            key="res_geo_avance"
                        )

                    st.markdown("##### Ponderadores de Tiempo de Actividad por Recurso (%)")
                    st.caption("Afecta la duración de las actividades. Ej: +30 para que la actividad dure un 30% más, -20 para que dure un 20% menos.")

                    ponderadores_recursos = {}
                    recursos_list_geo = list(recursos_nivel['Recurso'].unique())
                    num_cols_geo = 3
                    cols_geo = st.columns(num_cols_geo)
                    for i, recurso in enumerate(recursos_list_geo):
                        with cols_geo[i % num_cols_geo]:
                            ponderador = st.number_input(
                                f"{recurso} (%)",
                                value=0,
                                step=5,
                                key=f"ponderador_{recurso}"
                            )
                            if ponderador != 0:
                                ponderadores_recursos[recurso] = ponderador

                    submitted_geo = st.form_submit_button("Guardar Restricción Geológica", type="primary")

                    if submitted_geo:
                        if res_geo_nombre and res_geo_frentes:
                            nueva_res_geo = {
                                'nombre': res_geo_nombre.strip(),
                                'frentes_aplicables': res_geo_frentes,
                                'castigo_avance': res_geo_avance,
                                'ponderadores_recursos': ponderadores_recursos
                            }

                            # Si ya existe una con ese nombre, la actualiza. Si no, la añade.
                            found = False
                            for i, res in enumerate(st.session_state.restricciones_geologicas):
                                if res['nombre'] == res_geo_nombre.strip():
                                    st.session_state.restricciones_geologicas[i] = nueva_res_geo
                                    found = True
                                    break
                            if not found:
                                st.session_state.restricciones_geologicas.append(nueva_res_geo)
                            
                            st.success(f"Restricción **'{res_geo_nombre}'** guardada.")

                        else:
                            st.error("Por favor, ingrese el nombre y seleccione al menos un túnel afectado.")

            if st.session_state.restricciones_geologicas:
                st.markdown("##### Restricciones Geológicas Configuradas:")
                
                display_data = []
                for res in st.session_state.restricciones_geologicas:
                    ponderadores_str = ", ".join([f"{k}: {v}%" for k, v in res['ponderadores_recursos'].items()])
                    if not ponderadores_str:
                        ponderadores_str = "Ninguno"
                    
                    display_data.append({
                        "Nombre": res['nombre'],
                        "Túneles Afectados": ", ".join(res['frentes_aplicables']),
                        "Ajuste Avance": f"{res['castigo_avance']}%",
                        "Ponderadores Recursos": ponderadores_str
                    })
                
                st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

                if st.button("Limpiar Todas las Restricciones Geológicas", key="limpiar_res_geo"):
                    st.session_state.restricciones_geologicas = []
                    st.rerun()

        _menu_7_restricciones_geologicas(recursos_nivel, frentes_disponibles)

        # ----------------------------------------------------
        # 8. Configuración de Fallas de Equipos
        # ----------------------------------------------------
        # Todo el Menu 8 vive dentro de un @st.fragment, igual que los Menus 3, 4b, 6 y 7:
        # "Guardar Escenario de Falla" y "Limpiar Todos los Escenarios de Falla" disparaban
        # un rerun de la app COMPLETA. Encerrado en el fragment, esos botones solo vuelven
        # a ejecutar este bloque.
        @st.fragment
        def _menu_8_fallas_equipos(recursos_nivel, frentes_disponibles):
            st.subheader("8. Configuración de Fallas de Equipos")

            with st.expander("Añadir/Editar Escenario de Falla"):
                with st.form("form_falla_equipo", clear_on_submit=True):
                    st.markdown("##### Definir Escenario de Falla")
                    
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        falla_nombre = st.text_input("Nombre del Escenario de Falla (único)", key="falla_nombre")
                    with col_f2:
                        falla_frentes = st.multiselect(
                            "Túneles (Frentes) Afectados",
                            frentes_disponibles,
                            key="falla_frentes"
                        )

                    st.markdown("##### Probabilidad de Falla por Ciclo y Demora Asociada")
                    st.caption("Para cada recurso, defina la probabilidad (%) de que falle en un ciclo y las horas de demora si ocurre la falla.")

                    fallas_por_recurso = {}
                    recursos_list_falla = list(recursos_nivel['Recurso'].unique())
                    
                    for recurso in recursos_list_falla:
                        st.markdown(f"**{recurso}**")
                        col_prob, col_demora = st.columns(2)
                        with col_prob:
                            prob = st.number_input(
                                f"Probabilidad de Falla (%) para {recurso}",
                                min_value=0, max_value=100, value=0, step=1,
                                key=f"falla_prob_{recurso}"
                            )
                        with col_demora:
                            demora = st.number_input(
                                f"Horas de Demora por Falla para {recurso}",
                                min_value=0.0, value=0.0, step=0.5,
                                key=f"falla_demora_{recurso}"
                            )
                        if prob > 0 and demora > 0:
                            fallas_por_recurso[recurso] = {'probabilidad': prob, 'demora': demora}

                    submitted_falla = st.form_submit_button("Guardar Escenario de Falla", type="primary")

                    if submitted_falla:
                        if falla_nombre and falla_frentes:
                            nuevo_escenario_falla = {
                                'nombre': falla_nombre.strip(),
                                'frentes_aplicables': falla_frentes,
                                'fallas_por_recurso': fallas_por_recurso
                            }

                            found = False
                            for i, falla in enumerate(st.session_state.fallas_equipos):
                                if falla['nombre'] == falla_nombre.strip():
                                    st.session_state.fallas_equipos[i] = nuevo_escenario_falla
                                    found = True
                                    break
                            if not found:
                                st.session_state.fallas_equipos.append(nuevo_escenario_falla)
                            
                            st.success(f"Escenario de Falla **'{falla_nombre}'** guardado.")
                        else:
                            st.error("Por favor, ingrese el nombre y seleccione al menos un túnel afectado.")

            if st.session_state.fallas_equipos:
                st.markdown("##### Escenarios de Falla Configurados:")
                
                display_data_falla = []
                for falla in st.session_state.fallas_equipos:
                    detalles_falla = ", ".join([f"{k} ({v['probabilidad']}% prob, {v['demora']}h demora)" for k, v in falla['fallas_por_recurso'].items()])
                    if not detalles_falla:
                        detalles_falla = "Ninguno"
                    
                    display_data_falla.append({
                        "Nombre Escenario": falla['nombre'],
                        "Túneles Afectados": ", ".join(falla['frentes_aplicables']),
                        "Detalles de Falla": detalles_falla
                    })
                
                st.dataframe(pd.DataFrame(display_data_falla), use_container_width=True, hide_index=True)

                if st.button("Limpiar Todos los Escenarios de Falla", key="limpiar_fallas"):
                    st.session_state.fallas_equipos = []
                    st.rerun()

        _menu_8_fallas_equipos(recursos_nivel, frentes_disponibles)

        # ----------------------------------------------------
        # 9. Selección de Frentes
        # ----------------------------------------------------
        st.subheader("9. Selección de Frentes a Simular")
        frentes_disponibles_sorted = sorted(frentes_nivel['Frentes'].unique())
        frentes_seleccionados = st.multiselect(
            "Seleccione los frentes a simular",
            frentes_disponibles_sorted,
            default=frentes_disponibles_sorted,
            key="frentes_seleccionados"
        )

        # ----------------------------------------------------
        # 10. Configuración de Restricciones
        # ----------------------------------------------------
        st.subheader("10. Configuración de Restricciones Físicas (FH, OC, R)")

        col_rest_1, col_rest_2 = st.columns(2)
        with col_rest_1:
            radio_restriccion = st.number_input(
                "Radio de Impacto FH/OC (metros)",
                min_value=0.1, value=5.0, step=0.5, key="radio_restriccion_input"
            )
        with col_rest_2:
            demora_horas = st.number_input(
                "Demora Aplicada si hay Impacto (horas)",
                min_value=0.0, value=2.0, step=0.1, key="demora_horas_input"
            )
        st.markdown("##### Radios por tipo de restriccion de turno")
        st.caption("En el motor original: Polvorazo = 80 m y PA FH = 40 m. Estos radios se usan para la hoja Restriccion.")
        tipos_restriccion = sorted(set(st.session_state.df_res['Tipo'].dropna().astype(str)) | set(DEFAULT_RESTRICTION_RADII.keys()))
        df_radios = pd.DataFrame({
            'Tipo': tipos_restriccion,
            'Radio (m)': [float(DEFAULT_RESTRICTION_RADII.get(tipo, radio_restriccion)) for tipo in tipos_restriccion]
        })
        df_radios_editado = st.data_editor(
            df_radios,
            hide_index=True,
            use_container_width=True,
            key="radios_restriccion_editor"
        )
        radios_por_tipo = {
            str(row['Tipo']): float(row['Radio (m)'])
            for _, row in df_radios_editado.iterrows()
            if pd.notna(row['Tipo']) and pd.notna(row['Radio (m)'])
        }
        st.session_state.radio_restriccion = radio_restriccion
        st.session_state.demora_horas = demora_horas
        st.session_state.radios_por_tipo = radios_por_tipo

        # ----------------------------------------------------
        # 11. Visualización Geometría
        # ----------------------------------------------------
        st.subheader(f"11. Geometría: Túneles y Restricciones Físicas")
        frentes_completos = st.session_state.df_frentes[
            st.session_state.df_frentes['Nivel'] == nivel_seleccionado
        ].copy()
        fig_geo = graficar_geometria(
            frentes_completos, 
            st.session_state.df_fh, 
            st.session_state.df_oc, 
            st.session_state.df_res, 
            radio_restriccion,
            radios_por_tipo
        )
        st.plotly_chart(fig_geo, use_container_width=True)

        # ----------------------------------------------------
        # 12. Ruta de Salida y Ejecutar Simulación
        # ----------------------------------------------------
        # Todo el Menu 12 vive dentro de un @st.fragment: asi, tipear en el campo
        # de ruta de salida o presionar "SIMULAR" solo vuelve a ejecutar este
        # fragmento (y muestra el progreso en vivo dentro de el), en vez de
        # disparar un rerun de la app completa que re-renderiza y "apaga"
        # visualmente los Menus 1 a 11 mientras la simulacion corre.
        @st.fragment
        def _menu_12_ejecutar_simulacion(nivel_seleccionado, frentes_nivel, frentes_seleccionados,
                                          radio_restriccion, demora_horas, actividades_nivel):
            st.subheader("12. Ejecutar Simulacion")

            st.info(
                "Esta app corre en Streamlit Cloud: no puede escribir archivos en el disco "
                "de tu computador, así que los resultados (6 archivos formato David Montenegro "
                "+ reporte combinado) se generan en memoria y quedan disponibles para "
                "**descargar directamente desde el navegador** en la Sección 16, al terminar la simulación."
            )

            if st.button("SIMULAR", type="primary", use_container_width=True):
                # ---- Orquestador de tareas (Redis): evita choques entre simulaciones ----
                redis_client = get_redis_client()
                id_proceso_sim = str(uuid.uuid4())
                lock_adquirido = adquirir_lock_simulacion(redis_client, id_proceso_sim)

                frentes_simular = ordenar_frentes_por_prioridad(
                    frentes_nivel,
                    frentes_seleccionados,
                    st.session_state.df_fh,
                    st.session_state.df_oc,
                    st.session_state.ruta_critica
                )
                if not frentes_simular:
                    st.error("Seleccione al menos un frente")
                elif not lock_adquirido:
                    st.warning("El orquestador (Redis) indica que ya hay una simulación en curso. "
                               "Espere a que termine antes de lanzar otra para que los procesos no se traben.")
                else:
                    tiempo_limite_sim = st.session_state.tiempo_limite
                    metros_avance_sim = st.session_state.metros_avance
                    n_simulaciones_sim = int(st.session_state.n_simulaciones_input)

                    with st.spinner('Ejecutando simulación...'):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("Calculando demoras por restricciones...")
                        registrar_tarea_historial(redis_client, id_proceso_sim, "inicio", f"{n_simulaciones_sim} escenarios")
                        actualizar_estado_orquestador(redis_client, "en_progreso", 10, "Calculando demoras por restricciones...")
                        frentes_completos_sim = st.session_state.df_frentes[
                            st.session_state.df_frentes['Nivel'] == nivel_seleccionado
                        ].copy()
                        demoras_constantes, demoras_por_turno = calcular_demoras_restricciones(
                            frentes_completos_sim, st.session_state.df_fh, st.session_state.df_oc, st.session_state.df_res, 
                            radio_restriccion, demora_horas, st.session_state.radios_por_tipo
                        )
                    
                        demoras_constantes_sim = { 
                            k: v for k, v in demoras_constantes.items() if k in frentes_simular 
                        }
                        demoras_por_turno_sim = [ 
                            res for res in demoras_por_turno if res['Frente'] in frentes_simular 
                        ]

                        st.session_state.demoras_constantes = demoras_constantes_sim
                        st.session_state.demoras_por_turno = demoras_por_turno_sim

                        progress_bar.progress(10)

                        frentes_info = {}
                        seccion_col_sim = columna_seccion(frentes_nivel)
                        for frente in frentes_simular:
                            fila_frente = frentes_nivel[frentes_nivel['Frentes'] == frente].iloc[0]
                            distancia = calcular_distancia(
                                fila_frente['Xi'], fila_frente['Yi'], fila_frente['Xf'], fila_frente['Yf']
                            )
                            avance_planificado_val = fila_frente.get(
                                'Avance Planificado (m/h)', fila_frente.get('Avance Planificado', 0.0)
                            )
                            seccion_frente = str(fila_frente[seccion_col_sim]) if seccion_col_sim else ''
                            metros_por_ciclo_frente = st.session_state.metros_por_seccion.get(
                                seccion_frente,
                                metros_avance_sim
                            )
                            frentes_info[frente] = {
                                'distancia': distancia,
                                'xi': fila_frente['Xi'], 'yi': fila_frente['Yi'],
                                'xf': fila_frente['Xf'], 'yf': fila_frente['Yf'],
                                'avance_planificado': avance_planificado_val,
                                'seccion': seccion_frente,
                                'metros_por_ciclo': metros_por_ciclo_frente,
                                'nivel': nivel_seleccionado
                            }
                        st.session_state.frentes_info = frentes_info
                        st.caption("Orden de prioridad usado: " + " -> ".join(frentes_simular))

                        status_text.text(f"Simulando 0/{n_simulaciones_sim} escenarios...")
                        actualizar_estado_orquestador(redis_client, "en_progreso", 50, f"Simulando {n_simulaciones_sim} escenarios...")

                        plan_flota_activo = st.session_state.get('plan_recursos_turno', [])

                        # Rango de la barra reservado para el loop de Monte Carlo: arranca en 10%
                        # (justo despues del calculo de demoras por restricciones) y llega hasta 90%,
                        # dejando 90-100% para la simulacion de linea base (si aplica) y el cierre.
                        PROGRESO_INICIO_SIM, PROGRESO_FIN_SIM = 10, 90

                        def _actualizar_progreso_simulacion(completadas, total):
                            fraccion = completadas / total
                            progreso_pct = PROGRESO_INICIO_SIM + int(fraccion * (PROGRESO_FIN_SIM - PROGRESO_INICIO_SIM))
                            progress_bar.progress(min(progreso_pct, PROGRESO_FIN_SIM))
                            status_text.text(f"Simulando {completadas}/{total} escenarios...")

                        # Llamada a la función de simulación actualizada
                        resultados, estadisticas_recursos, traza_eventos, ventanas_flota, registro_cambios_flota, avance_por_turno_real_sim = simular_avance_con_transporte(
                            actividades_nivel, 
                            frentes_info, 
                            st.session_state.recursos_config, 
                            tiempo_limite_sim, 
                            metros_avance_sim, 
                            n_simulaciones_sim, 
                            st.session_state.velocidades_config,
                            demoras_constantes_sim,
                            demoras_por_turno_sim,
                            st.session_state.sistema_turnos,
                            st.session_state.restricciones_geologicas,
                            st.session_state.fallas_equipos,
                            plan_flota_activo, # <-- Plan de orquestacion de flota por turno
                            progress_callback=_actualizar_progreso_simulacion
                        )

                        # Si hay un plan de flota activo, correr tambien una linea base (sin cambios de
                        # flota) para poder validar en resultados que efectivamente estos cambios
                        # modificaron el avance obtenido.
                        resultados_base = None
                        if plan_flota_activo:
                            def _actualizar_progreso_linea_base(completadas, total):
                                fraccion = completadas / total
                                progress_bar.progress(90 + int(fraccion * 10))
                                status_text.text(f"Simulando línea base {completadas}/{total} (sin cambios de flota)...")

                            resultados_base, _, _, _, _, _ = simular_avance_con_transporte(
                                actividades_nivel,
                                frentes_info,
                                st.session_state.recursos_config,
                                tiempo_limite_sim,
                                metros_avance_sim,
                                n_simulaciones_sim,
                                st.session_state.velocidades_config,
                                demoras_constantes_sim,
                                demoras_por_turno_sim,
                                st.session_state.sistema_turnos,
                                st.session_state.restricciones_geologicas,
                                st.session_state.fallas_equipos,
                                [], # sin plan de cambios de flota: cantidad base fija
                                progress_callback=_actualizar_progreso_linea_base
                            )
                    
                        st.session_state.resultados = resultados
                        st.session_state.resultados_base_sin_cambios_flota = resultados_base
                        # Avance por turno REAL (misma fuente que "resultados"/Avance Final), para
                        # que la seccion "Avance Acumulado por Turno" deje de reconstruirlo por
                        # conteo de ciclos en la traza (fuente de la inconsistencia reportada:
                        # mostraba avance en turnos donde el Avance Final real habia quedado en 0).
                        st.session_state.avance_por_turno_real = avance_por_turno_real_sim
                        st.session_state.estadisticas_recursos = estadisticas_recursos
                        st.session_state.traza_eventos = traza_eventos
                        # Se construye el DataFrame UNA sola vez aqui, al terminar de simular, en vez
                        # de hacer pd.DataFrame(traza_eventos) cada vez que la UI hace un rerun (esto
                        # era el principal cuello de botella de "carga progresiva": con cientos de
                        # miles de filas, reconstruirlo 5 veces por rerun es lento y evitable).
                        st.session_state.df_traza_eventos_cache = pd.DataFrame(traza_eventos)
                        # run_id unico por corrida: se usa como parte de la clave de cache en las
                        # funciones @st.cache_data que procesan la traza (groupby, percentiles, Gantt),
                        # para que Streamlit sepa que puede reusar resultados si la traza no cambio.
                        st.session_state.run_id = str(uuid.uuid4())
                        st.session_state.ventanas_flota = ventanas_flota
                        st.session_state.registro_cambios_flota = registro_cambios_flota
                        st.session_state.nivel_simulado = nivel_seleccionado
                        st.session_state.duracion_turno_simulada = st.session_state.get('duracion_turno_base', 8.0)
                        st.session_state.numero_turnos_simulado = st.session_state.get('numero_turnos', 62)
                        st.session_state.n_simulaciones_ejecutadas = n_simulaciones_sim
                        st.session_state.df_actividades_simuladas = actividades_nivel.copy()
                        st.session_state.df_frentes_nivel_simuladas = frentes_nivel.copy()
                        progress_bar.progress(100)
                        status_text.success("Simulación completada con éxito!")

                        # ---- Orquestador de tareas (Redis): cierre de la tarea ----
                        actualizar_estado_orquestador(redis_client, "completado", 100, "Simulación completada con éxito")
                        registrar_tarea_historial(redis_client, id_proceso_sim, "completado", f"{n_simulaciones_sim} escenarios")
                        liberar_lock_simulacion(redis_client, id_proceso_sim)

                        # Forzar un unico rerun completo al terminar: asi los Menus 13-16
                        # (fuera de este fragmento) recogen los resultados nuevos de inmediato,
                        # sin que el usuario tenga que interactuar con algo mas para verlos.
                        st.rerun()

        _menu_12_ejecutar_simulacion(nivel_seleccionado, frentes_nivel, frentes_seleccionados,
                                      radio_restriccion, demora_horas, actividades_nivel)

        
        # ----------------------------------------------------
        # 13. Resultados
        # ----------------------------------------------------
        if 'resultados' in st.session_state:
            st.markdown("---")
            st.header("Resultados de la Simulación")

            # ----------------------------------------------------
            # 13.0 Modo de visualización: Promedio vs Simulación específica
            # ----------------------------------------------------
            # Mostrar solo promedios/percentiles puede ser engañoso (oculta la variabilidad
            # real y mezcla escenarios distintos). Se ofrece la alternativa de ver los
            # resultados de UNA simulación puntual, de forma completamente determinística.
            n_sims_totales_vis = st.session_state.get('n_simulaciones_ejecutadas', 1)
            col_modo_vis, col_sim_vis = st.columns([1.3, 1])
            with col_modo_vis:
                modo_visualizacion = st.radio(
                    "Modo de visualización de resultados",
                    ["Promedio (todas las simulaciones)", "Simulación específica"],
                    horizontal=True,
                    key="modo_visualizacion_resultados"
                )
            sim_seleccionada_global = None
            if modo_visualizacion == "Simulación específica":
                with col_sim_vis:
                    sim_seleccionada_global = st.selectbox(
                        "N° de simulación a visualizar",
                        list(range(1, n_sims_totales_vis + 1)),
                        key="sim_seleccionada_global"
                    )
                st.caption(
                    f"📌 Mostrando resultados de la **simulación N° {sim_seleccionada_global}** de forma "
                    "individual/determinística (no promedios ni percentiles). Esto aplica al Avance Final, "
                    "Avance por Turno, Utilización de Recursos y la Carta Gantt (que en este modo deja de "
                    "ser probabilística)."
                )

            # Aviso de bloqueos por falta absoluta de recurso: si alguna simulación detectó que
            # un túnel se quedó sin ningún recurso requerido (ahora y para siempre), se muestra
            # aquí de forma explícita antes de cualquier otro resultado.
            df_traza_check = st.session_state.df_traza_eventos_cache
            resumen_bloqueos = calcular_resumen_bloqueos(df_traza_check, st.session_state.run_id)
            if resumen_bloqueos is not None:
                st.error(
                    "🚫 **Uno o más túneles quedaron bloqueados durante la simulación por falta absoluta "
                    "de un recurso requerido** (cantidad 0, sin que el plan de flota lo reincorpore). "
                    "El avance de esos túneles se detuvo en el punto indicado y no continuó."
                )
                st.dataframe(resumen_bloqueos, use_container_width=True, hide_index=True)
            
            # Avance Final
            if sim_seleccionada_global is None:
                st.subheader("Avance Final Acumulado por Túnel")
            else:
                st.subheader(f"Avance Final Acumulado por Túnel — Simulación N° {sim_seleccionada_global}")
            datos_tabla = []
            for frente, avances in st.session_state.resultados.items():
                if avances:
                    distancia_total = st.session_state.frentes_info[frente]['distancia']

                    if sim_seleccionada_global is None:
                        arr_avances = np.array(avances)
                        pct_avance = percentiles_reporte(arr_avances)

                        datos_tabla.append({
                            'Túnel': frente,
                            'Distancia Total (m)': r2(distancia_total),
                            'P0 - Mínimo (m)': pct_avance['p0'],
                            'P10 (m)': pct_avance['p10'],
                            'P30 (m)': pct_avance['p30'],
                            'P50 (m)': pct_avance['p50'],
                            'Esperanza (m)': pct_avance['esperanza'],
                            'P70 (m)': pct_avance['p70'],
                            'P90 (m)': pct_avance['p90'],
                            'P100 - Máximo (m)': pct_avance['p100'],
                        })
                    else:
                        idx_sim = min(sim_seleccionada_global - 1, len(avances) - 1)
                        avance_sim = avances[idx_sim]

                        datos_tabla.append({
                            'Túnel': frente,
                            'Distancia Total (m)': r2(distancia_total),
                            f'Avance Sim. {sim_seleccionada_global} (m)': r2(avance_sim),
                            'Porcentaje Completado (%)': r2((avance_sim / distancia_total) * 100) if distancia_total else 0.0
                        })
            
            df_resultados = pd.DataFrame(datos_tabla)
            st.dataframe(df_resultados, use_container_width=True, hide_index=True)

            # ----------------------------------------------------
            # 13a. Análisis de Convergencia Monte Carlo (¿alcanzan mis N simulaciones?)
            # ----------------------------------------------------
            with st.expander("📈 Análisis de Convergencia Monte Carlo (¿son suficientes mis N° de simulaciones?)", expanded=False):
                st.caption(
                    "Se analiza el **Avance Final por Túnel** (una simulación = un valor), que es la "
                    "métrica que realmente reportas (P10/P50/P90/Esperanza). Analizar la convergencia de "
                    "actividades individuales no sirve aquí: cada actividad se sortea miles de veces por "
                    "simulación (una vez por ciclo), así que su promedio converge casi de inmediato y no "
                    "dice nada sobre si el N° de *simulaciones completas* es suficiente."
                )
                frente_convergencia = st.selectbox(
                    "Túnel a analizar",
                    list(st.session_state.resultados.keys()),
                    key="frente_convergencia_sel"
                )
                valores_convergencia = st.session_state.resultados.get(frente_convergencia, [])

                if len(valores_convergencia) < 10:
                    st.info("Se necesitan al menos 10 simulaciones para un análisis de convergencia significativo.")
                else:
                    media_c, error_c, error_rel_c = calcular_error_relativo_convergencia(valores_convergencia)
                    col_conv1, col_conv2, col_conv3 = st.columns(3)
                    col_conv1.metric("Promedio (N total)", f"{media_c:.2f} m")
                    col_conv2.metric("Error Estándar", f"{error_c:.3f} m")
                    col_conv3.metric("Error Relativo", f"{error_rel_c:.2f} %")

                    if error_rel_c < 2:
                        st.success(
                            f"✅ Con N={len(valores_convergencia)} simulaciones, el error relativo del promedio "
                            f"es {error_rel_c:.2f}% (< 2%). El número de simulaciones parece suficiente."
                        )
                    else:
                        st.warning(
                            f"⚠️ Con N={len(valores_convergencia)} simulaciones, el error relativo del promedio "
                            f"es {error_rel_c:.2f}% (≥ 2%). Podría convenir aumentar el número de simulaciones."
                        )

                    fig_convergencia = graficar_convergencia_montecarlo(
                        valores_convergencia,
                        titulo=f"Convergencia del Avance Final Promedio — {frente_convergencia}"
                    )
                    st.plotly_chart(fig_convergencia, use_container_width=True)

                    fig_convergencia_pct = graficar_convergencia_percentiles(
                        valores_convergencia,
                        titulo=f"Convergencia de Percentiles del Avance Final — {frente_convergencia}"
                    )
                    st.plotly_chart(fig_convergencia_pct, use_container_width=True)

                    st.caption(
                        "Arriba: cada línea celeste es el promedio acumulado con un orden distinto de las "
                        "mismas simulaciones (para descartar que la 'estabilización' dependa del orden). "
                        "La línea roja es el promedio final con todas las simulaciones. Abajo: se ve si P10, "
                        "P50 y P90 también se estabilizan al aumentar N (a veces el promedio converge antes "
                        "que los percentiles extremos, que necesitan más datos)."
                    )

            # ----------------------------------------------------
            # 13b. Validacion del impacto de la Orquestacion de Flota por Turno
            # ----------------------------------------------------
            if st.session_state.get('registro_cambios_flota'):
                with st.expander("🔧 Validar impacto de los cambios de flota por turno", expanded=True):
                    st.markdown("##### Cambios de flota aplicados en esta simulación")
                    df_cambios_aplicados = pd.DataFrame(st.session_state.registro_cambios_flota).rename(columns={
                        'recurso': 'Recurso', 'turno': 'Turno', 'tipo': 'Tipo de Cambio',
                        'cantidad_anterior': 'Cantidad Anterior', 'cantidad_nueva': 'Cantidad Nueva',
                        'demora_horas': 'Demora Aplicada (h)', 'tiempo_efectivo': 'Hora Efectiva del Cambio'
                    })
                    st.dataframe(df_cambios_aplicados, use_container_width=True, hide_index=True)

                    resultados_base = st.session_state.get('resultados_base_sin_cambios_flota')
                    if resultados_base:
                        st.markdown("##### Comparación de avance: con cambios de flota vs. línea base (sin cambios)")
                        st.caption(
                            "Línea base = misma simulación pero con la cantidad de cada recurso fija en su valor "
                            "inicial durante todo el programa, sin aplicar el plan de orquestación por turno."
                        )
                        comparacion_data = []
                        hubo_diferencia = False
                        for frente in st.session_state.resultados.keys():
                            avances_con_plan = st.session_state.resultados.get(frente, [])
                            avances_sin_plan = resultados_base.get(frente, [])
                            if avances_con_plan and avances_sin_plan:
                                media_con_plan = float(np.mean(avances_con_plan))
                                media_sin_plan = float(np.mean(avances_sin_plan))
                                diferencia = media_con_plan - media_sin_plan
                                if abs(diferencia) > 1e-6:
                                    hubo_diferencia = True
                                comparacion_data.append({
                                    'Túnel': frente,
                                    'Avance Promedio Con Plan (m)': round(media_con_plan, 2),
                                    'Avance Promedio Sin Plan (m)': round(media_sin_plan, 2),
                                    'Diferencia (m)': round(diferencia, 2),
                                    'Diferencia (%)': round((diferencia / media_sin_plan) * 100, 2) if media_sin_plan > 0 else 0.0
                                })
                        df_comparacion = pd.DataFrame(comparacion_data)
                        st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

                        if hubo_diferencia:
                            st.success(
                                "✅ Se confirma que el plan de orquestación de flota modificó el avance respecto "
                                "a la línea base sin cambios."
                            )
                        else:
                            st.warning(
                                "⚠️ El avance obtenido con el plan de flota es igual a la línea base sin cambios. "
                                "Revise si los turnos/cantidades configurados realmente afectan a recursos usados "
                                "por los frentes seleccionados, o si otras restricciones (fallas de equipos, "
                                "restricciones geológicas) están limitando el avance de igual forma en ambos casos."
                            )
                    else:
                        st.caption("No se generó línea base de comparación para esta corrida.")

            st.markdown("---")

            # Avance Acumulado por Turno
            opciones_percentil_turno = {
                'P0 (Mínimo)': 'p0',
                'P10': 'p10',
                'P30': 'p30',
                'P50 (Mediana)': 'p50',
                'Esperanza (Promedio)': 'esperanza',
                'P70': 'p70',
                'P90': 'p90',
                'P100 (Máximo)': 'p100',
            }
            if sim_seleccionada_global is None:
                st.write(f"**Avance Acumulado por Turno**")
                etiqueta_col_turno = "Avance Prom."
                percentiles_turno_sel = st.multiselect(
                    "Percentiles a mostrar en la gráfica de avance por turno",
                    list(opciones_percentil_turno.keys()),
                    default=['Esperanza (Promedio)'],
                    key="percentiles_turno_seleccionados",
                    help="Se puede elegir más de un percentil para compararlos en la misma gráfica. "
                         "La tabla siempre muestra la Esperanza (promedio) por túnel/turno."
                )
                if not percentiles_turno_sel:
                    percentiles_turno_sel = ['Esperanza (Promedio)']
            else:
                st.write(f"**Avance Acumulado por Turno — Simulación N° {sim_seleccionada_global}**")
                etiqueta_col_turno = f"Avance Sim. {sim_seleccionada_global}"
                percentiles_turno_sel = None
            # Se usa el avance REAL por turno capturado durante la simulacion (misma fuente
            # que "resultados"/Avance Final), en vez de calcular_avance_por_turno (que lo
            # reconstruia contando ciclos en la traza de eventos y podia mostrar avance en
            # turnos donde el Avance Final real habia quedado en 0, por ciclos truncados por
            # tiempo_limite o por bloqueo de recurso). Se mantiene un fallback a la
            # reconstruccion antigua solo por compatibilidad con corridas ya guardadas en
            # session_state antes de este fix, que no tienen 'avance_por_turno_real'.
            if 'avance_por_turno_real' in st.session_state and st.session_state.avance_por_turno_real:
                avance_turnos = st.session_state.avance_por_turno_real
            else:
                df_traza_completa = st.session_state.df_traza_eventos_cache
                avance_turnos = calcular_avance_por_turno(
                    st.session_state.run_id,
                    df_traza_completa, 
                    st.session_state.tiempo_limite, 
                    st.session_state.sistema_turnos, 
                    st.session_state.frentes_info, 
                    st.session_state.metros_avance
                )

            datos_turno = []
            datos_turno_percentiles = []  # formato largo (Tunel, Turno_Num, Percentil, valor) para la grafica multi-percentil
            if avance_turnos and any(avance_turnos.values()):
                frentes_keys = list(st.session_state.frentes_info.keys())
                n_max_turnos = 0
                for frente in frentes_keys:
                    if frente in avance_turnos:
                        n_max_turnos = max(n_max_turnos, len(avance_turnos[frente]))
                
                turnos_keys = list(range(1, n_max_turnos + 1))
                
                for frente in frentes_keys:
                    if frente in avance_turnos:
                        distancia_total_frente = st.session_state.frentes_info.get(frente, {}).get('distancia', None)

                        # Serie de referencia (Esperanza/promedio, o el valor puntual de la
                        # simulación seleccionada): se usa para la tabla y para detectar cuándo
                        # el túnel se estabiliza/termina.
                        lista_valores_serie = []
                        # Series por percentil (solo en modo "Promedio") para la grafica comparativa.
                        series_percentiles = {p: [] for p in (percentiles_turno_sel or [])}

                        for turno in turnos_keys:
                            valores_turno = avance_turnos[frente].get(turno)
                            if not valores_turno:
                                lista_valores_serie.append(0.0)
                                for p in series_percentiles:
                                    series_percentiles[p].append(0.0)
                            elif sim_seleccionada_global is None:
                                arr_turno = np.array(valores_turno, dtype=float)
                                lista_valores_serie.append(float(np.mean(arr_turno)))
                                if series_percentiles:
                                    pct_turno = percentiles_reporte(arr_turno)
                                    for p in series_percentiles:
                                        series_percentiles[p].append(float(pct_turno[opciones_percentil_turno[p]]))
                            else:
                                idx_sim = min(sim_seleccionada_global - 1, len(valores_turno) - 1)
                                lista_valores_serie.append(float(valores_turno[idx_sim]))
                        valores_serie = np.array(lista_valores_serie)

                        # Umbral de "incremento relevante" entre turnos consecutivos. Se usa
                        # en vez de exigir que el promedio llegue exactamente al 100% de la
                        # meta, porque en el promedio de muchas simulaciones Monte Carlo
                        # siempre puede quedar un pequeño porcentaje de escenarios rezagados
                        # (demoras, fallas de equipos) que nunca alcanzan exactamente la
                        # distancia total, dejando el promedio asintotico sin llegar nunca
                        # al 99.9% exacto.
                        epsilon_avance = max(0.05, (distancia_total_frente or 0.0) * 0.002)

                        # Se busca, de atras hacia adelante, el ultimo turno en el que aun
                        # hubo un incremento relevante de avance. A partir de ahi (si el
                        # tunel sigue avanzando de forma real hasta el ultimo turno, no se
                        # corta nada) se considera que el tunel ya se estabilizo/termino.
                        turno_ultimo_avance_real = len(valores_serie)  # por defecto: no cortar
                        hubo_incremento = False
                        for i in range(len(valores_serie) - 1, 0, -1):
                            if (valores_serie[i] - valores_serie[i - 1]) >= epsilon_avance:
                                turno_ultimo_avance_real = i + 1  # turnos_keys es 1-based
                                hubo_incremento = True
                                break
                        if not hubo_incremento and len(valores_serie) > 0 and valores_serie[0] > 0:
                            # Ya venia avanzando desde el turno 1 y nunca hay un salto grande
                            # detectable (ej. avanza muy poco y parejo): no se corta, se
                            # muestra la serie completa tal cual.
                            turno_ultimo_avance_real = len(valores_serie)

                        fila_frente = {'Tunel': frente}
                        for idx, turno in enumerate(turnos_keys):
                            if turno > turno_ultimo_avance_real:
                                # El tunel ya se estabilizo (dejo de avanzar de forma
                                # relevante): no se siguen mostrando resultados.
                                fila_frente[f'T{turno} {etiqueta_col_turno} (m)'] = np.nan
                            else:
                                fila_frente[f'T{turno} {etiqueta_col_turno} (m)'] = r2(float(valores_serie[idx]))
                        datos_turno.append(fila_frente)

                        # Formato largo por percentil (solo en modo Promedio), recortado al mismo
                        # punto de estabilizacion que la serie de referencia.
                        if sim_seleccionada_global is None:
                            for p, serie_p in series_percentiles.items():
                                for idx, turno in enumerate(turnos_keys):
                                    if turno > turno_ultimo_avance_real:
                                        continue
                                    datos_turno_percentiles.append({
                                        'Tunel': frente,
                                        'Turno_Num': turno,
                                        'Percentil': p,
                                        'Avance Acumulado (m)': r2(float(serie_p[idx]))
                                    })

                if datos_turno:
                    df_turnos = pd.DataFrame(datos_turno)
                    st.dataframe(df_turnos, use_container_width=True, hide_index=True)

                    if sim_seleccionada_global is None:
                        # Formato largo con una fila por (Tunel, Turno, Percentil) para poder
                        # graficar varios percentiles a la vez, uno por cada linea.
                        df_melt = pd.DataFrame(datos_turno_percentiles)
                        nombres_percentiles = " / ".join(percentiles_turno_sel)
                        titulo_grafico_turno = f'Avance Acumulado por Turno ({nombres_percentiles}) - {st.session_state.sistema_turnos}'
                        etiqueta_y_turno = 'Avance Acumulado (m)'

                        fig_turnos = px.line(
                            df_melt.sort_values(by='Turno_Num'),
                            x='Turno_Num', y='Avance Acumulado (m)',
                            color='Tunel',
                            line_dash='Percentil' if len(percentiles_turno_sel) > 1 else None,
                            title=titulo_grafico_turno,
                            labels={'Turno_Num': 'Número de Turno', 'Avance Acumulado (m)': etiqueta_y_turno},
                            markers=True,
                            hover_data={'Percentil': True}
                        )
                    else:
                        df_melt = df_turnos.melt(id_vars='Tunel', var_name='Turno', value_name='Avance Acumulado (m)')
                        df_melt['Turno_Num'] = df_melt['Turno'].apply(lambda x: int(x.split(' ')[0][1:].replace('T', '')))
                        # Se eliminan los turnos posteriores a la finalizacion del tunel para que
                        # la linea del grafico termine ahi, en vez de mostrar una asintota plana.
                        df_melt = df_melt.dropna(subset=['Avance Acumulado (m)'])

                        titulo_grafico_turno = f'Avance Acumulado por Turno - Simulación N° {sim_seleccionada_global} - {st.session_state.sistema_turnos}'
                        etiqueta_y_turno = f'Avance Acumulado Sim. {sim_seleccionada_global} (m)'

                        fig_turnos = px.line(
                            df_melt.sort_values(by='Turno_Num'),
                            x='Turno_Num', y='Avance Acumulado (m)',
                            color='Tunel',
                            title=titulo_grafico_turno,
                            labels={'Turno_Num': 'Número de Turno', 'Avance Acumulado (m)': etiqueta_y_turno},
                            markers=True
                        )
                    fig_turnos.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
                    st.plotly_chart(fig_turnos, use_container_width=True)
            
            st.markdown("---")

            if sim_seleccionada_global is None:
                st.subheader("Utilizacion de Recursos")
            else:
                st.subheader(f"Utilizacion de Recursos — Simulación N° {sim_seleccionada_global}")
            datos_recursos = []
            if st.session_state.estadisticas_recursos:
                for recurso, stats_data in st.session_state.estadisticas_recursos.items():
                    if sim_seleccionada_global is None:
                        valor_utilizacion = np.mean(stats_data['utilizacion'])
                        valor_trabajo = np.mean(stats_data['tiempo_trabajando'])
                        valor_viaje = np.mean(stats_data['tiempo_viaje'])
                        col_uso, col_trabajo, col_viaje = (
                            'Uso Promedio (%)', 'Tiempo Productivo Promedio (h)', 'Tiempo de Viaje Promedio (h)'
                        )
                    else:
                        idx_sim = min(sim_seleccionada_global - 1, len(stats_data['utilizacion']) - 1)
                        valor_utilizacion = stats_data['utilizacion'][idx_sim]
                        valor_trabajo = stats_data['tiempo_trabajando'][idx_sim]
                        valor_viaje = stats_data['tiempo_viaje'][idx_sim]
                        col_uso, col_trabajo, col_viaje = (
                            f'Uso Sim. {sim_seleccionada_global} (%)',
                            f'Tiempo Productivo Sim. {sim_seleccionada_global} (h)',
                            f'Tiempo de Viaje Sim. {sim_seleccionada_global} (h)'
                        )

                    datos_recursos.append({
                        'Recurso': recurso,
                        'Cantidad': st.session_state.recursos_config[recurso]['cantidad'],
                        col_uso: round(valor_utilizacion, 2),
                        col_trabajo: round(valor_trabajo, 2),
                        col_viaje: round(valor_viaje, 2)
                    })
            
            if datos_recursos:
                df_recursos_res = pd.DataFrame(datos_recursos)
                st.dataframe(df_recursos_res, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ----------------------------------------------------
            # 14. Carta Gantt y Tiempos Detallados
            # ----------------------------------------------------
            st.subheader("14. Carta Gantt y Tiempos Detallados")

            # Cambios de flota por turno: son globales al recurso (no a un tunel especifico),
            # por lo que se muestran aparte, con su demora de activacion/desactivacion aplicada.
            df_fleet_changes = st.session_state.df_traza_eventos_cache
            if not df_fleet_changes.empty and 'type' in df_fleet_changes.columns:
                df_fleet_changes = df_fleet_changes[df_fleet_changes['type'] == 'fleet_change']
                if not df_fleet_changes.empty:
                    with st.expander("📅 Línea de tiempo de cambios de flota (con demora aplicada)"):
                        df_fc_sim1 = df_fleet_changes[df_fleet_changes['Simulation_ID'] == 1].sort_values('Start')
                        df_fc_display = df_fc_sim1[['Recurso', 'Actividad', 'Turno_Plan', 'Start', 'Demora_Aplicada_h']].rename(columns={
                            'Actividad': 'Cambio', 'Turno_Plan': 'Turno Configurado',
                            'Start': 'Hora Efectiva (h)', 'Demora_Aplicada_h': 'Demora Aplicada (h)'
                        })
                        st.dataframe(df_fc_display, use_container_width=True, hide_index=True)
            
            df_traza_completa = st.session_state.df_traza_eventos_cache
            
            if df_traza_completa.empty:
                st.warning("No hay datos de túneles en la traza.")
                tunel_seleccionado = None
            else:
                frentes_con_traza = sorted(df_traza_completa['Resource_Origin_Tunnel'].unique())
                tunel_seleccionado = st.selectbox("Seleccione Túnel para el Detalle", frentes_con_traza)

            if tunel_seleccionado:
                df_tunel = df_traza_completa[df_traza_completa['Resource_Origin_Tunnel'] == tunel_seleccionado].copy()
                n_sims_real = df_tunel['Simulation_ID'].nunique() if not df_tunel.empty else 1

                col_gantt, col_viaje = st.columns(2)
                
                with col_gantt:
                    if sim_seleccionada_global is None:
                        st.markdown("### Resumen de Tiempos por Actividad/Demora (Horas)")
                    else:
                        st.markdown(f"### Resumen de Tiempos por Actividad/Demora (Horas) — Sim. {sim_seleccionada_global}")
                    df_ciclo_events = df_tunel[df_tunel['type'].isin(['activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_equipment_failure', 'delay_fleet_change', 'blocked_no_resource'])].copy()
                    
                    if not df_ciclo_events.empty:
                        tiempos_agregados = calcular_tiempos_agregados_actividad(
                            df_tunel, st.session_state.run_id, tunel_seleccionado, sim_seleccionada_global
                        )
                        
                        if sim_seleccionada_global is None:
                            tiempos_resumen = tiempos_agregados.groupby('Actividad')['Duration'].mean().reset_index()
                            tiempos_resumen.rename(columns={'Duration': 'Tiempo Total Promedio (h)'}, inplace=True)
                        else:
                            tiempos_resumen = tiempos_agregados[
                                tiempos_agregados['Simulation_ID'] == sim_seleccionada_global
                            ][['Actividad', 'Duration']].reset_index(drop=True)
                            tiempos_resumen.rename(columns={'Duration': f'Tiempo Total Sim. {sim_seleccionada_global} (h)'}, inplace=True)
                        
                        if tiempos_resumen.empty:
                            st.info(f"La simulación N° {sim_seleccionada_global} no registró eventos de este tipo en este túnel.")
                        else:
                            cols_num_tiempos = tiempos_resumen.select_dtypes(include=[np.number]).columns.tolist()
                            tiempos_resumen[cols_num_tiempos] = tiempos_resumen[cols_num_tiempos].apply(r2)
                            st.dataframe(tiempos_resumen, use_container_width=True, hide_index=True)
                    else:
                        st.info("No hay eventos operacionales o demoras para este túnel.")

                with col_viaje:
                    if sim_seleccionada_global is None:
                        st.markdown("### Resumen de Viajes (Recursos)")
                    else:
                        st.markdown(f"### Resumen de Viajes (Recursos) — Sim. {sim_seleccionada_global}")
                    df_viajes = df_tunel[df_tunel['type'] == 'travel'].copy()
                    
                    if not df_viajes.empty:
                        viajes_agregados = calcular_viajes_agregados(
                            df_tunel, st.session_state.run_id, tunel_seleccionado, sim_seleccionada_global
                        )

                        if sim_seleccionada_global is None:
                            viajes_summary = viajes_agregados.drop(columns='Simulation_ID').groupby('Recurso_Tipo').mean(numeric_only=True).reset_index()
                            viajes_summary.rename(columns={
                                'Recurso_Tipo': 'Recurso',
                                'Distancia_Total': 'Distancia Total Promedio (m/sim)',
                                'Tiempo_Total': 'Tiempo Total Promedio (h/sim)',
                                'Num_Viajes': 'Número de Viajes Promedio (viajes/sim)'
                            }, inplace=True)
                        else:
                            viajes_summary = viajes_agregados[
                                viajes_agregados['Simulation_ID'] == sim_seleccionada_global
                            ].drop(columns='Simulation_ID').reset_index(drop=True)
                            viajes_summary.rename(columns={
                                'Recurso_Tipo': 'Recurso',
                                'Distancia_Total': f'Distancia Total Sim. {sim_seleccionada_global} (m)',
                                'Tiempo_Total': f'Tiempo Total Sim. {sim_seleccionada_global} (h)',
                                'Num_Viajes': f'Número de Viajes Sim. {sim_seleccionada_global}'
                            }, inplace=True)

                        if viajes_summary.empty:
                            st.info(f"La simulación N° {sim_seleccionada_global} no registró viajes de recursos en este túnel.")
                        else:
                            cols_num_viajes = viajes_summary.select_dtypes(include=[np.number]).columns.tolist()
                            viajes_summary[cols_num_viajes] = viajes_summary[cols_num_viajes].apply(r2)
                            st.dataframe(
                                viajes_summary, 
                                use_container_width=True, 
                                hide_index=True
                            )
                    else:
                        st.info("No se registraron viajes de recursos para este túnel.")

                st.markdown("---")
                
                if sim_seleccionada_global is None:
                    st.markdown("### Carta Gantt de Actividades")
                else:
                    st.markdown(f"### Carta Gantt de Actividades — Simulación N° {sim_seleccionada_global} (Determinística)")
                    st.caption(
                        "Al ver una simulación específica, la carta deja de ser probabilística: cada barra "
                        "muestra la duración exacta ocurrida en esa simulación, sin rango P10-P90."
                    )

                # ---- Controles de la Carta Gantt: modo (acumulado vs. promedio por ocurrencia)
                # y percentil a usar como largo de barra / eje de tiempo acumulado. ----
                col_gantt_modo, col_gantt_pct = st.columns(2)
                with col_gantt_modo:
                    modo_gantt_label = st.radio(
                        "¿Qué mostrar en cada barra?",
                        ["Tiempo acumulado (suma de todas las ocurrencias en la simulación)",
                         "Duración promedio por ocurrencia (ej. UNA perforación típica)"],
                        key="modo_gantt_radio",
                        help="Cada actividad ocurre una vez por ciclo. 'Tiempo acumulado' suma todas "
                             "las ocurrencias dentro de cada simulación (ej. 4 perforaciones de 1h = 4h). "
                             "'Duración promedio por ocurrencia' divide ese acumulado por el número de "
                             "ciclos completados en esa simulación (ej. 4h / 4 ciclos = 1h, la duración "
                             "típica de UNA perforación). Aplica también a las demoras (FH/OC, turno, "
                             "falla de equipo, cambio de recurso)."
                    )
                    modo_gantt = 'acumulado' if modo_gantt_label.startswith("Tiempo acumulado") else 'promedio_ciclo'
                with col_gantt_pct:
                    if sim_seleccionada_global is None:
                        percentil_gantt_label = st.selectbox(
                            "Percentil a usar como largo de la barra / eje acumulado",
                            list(PERCENTILES_DISPONIBLES_GANTT.keys()),
                            index=list(PERCENTILES_DISPONIBLES_GANTT.keys()).index('P50 (Mediana)'),
                            key="percentil_gantt_select",
                            help="Determina qué percentil de la distribución entre simulaciones se usa "
                                 "para el largo de cada barra y para acumular el eje de tiempo. El rango "
                                 "gris (P10–P90) se sigue mostrando como referencia de variabilidad, "
                                 "independiente del percentil elegido aquí."
                        )
                        percentil_gantt_col = PERCENTILES_DISPONIBLES_GANTT[percentil_gantt_label]
                    else:
                        percentil_gantt_col = 'P50'
                        st.caption("En modo simulación específica no hay percentiles: se muestra el valor exacto ocurrido.")

                df_ciclo_events = df_tunel[df_tunel['type'].isin(TIPOS_EVENTO_CICLO_GANTT)].copy()

                if not df_ciclo_events.empty:
                    gantt_data = calcular_datos_gantt(
                        df_tunel, st.session_state.run_id, tunel_seleccionado, sim_seleccionada_global,
                        modo=modo_gantt, percentil_barra=percentil_gantt_col
                    )
                    if gantt_data:
                        df_gantt = pd.DataFrame(gantt_data)
                        fig_gantt = go.Figure()

                        df_gantt['Start_Ref'] = df_gantt['Start_Barra'] + (df_gantt['Valor_Barra'] - df_gantt['P10']) / 2
                        df_gantt['End_Ref'] = df_gantt['Start_Barra'] + df_gantt['Valor_Barra'] + (df_gantt['P90'] - df_gantt['Valor_Barra']) / 2
                        df_gantt['Error_Plus'] = df_gantt['End_Ref'] - (df_gantt['Start_Barra'] + df_gantt['Valor_Barra'])
                        df_gantt['Error_Minus'] = (df_gantt['Start_Barra'] + df_gantt['Valor_Barra']) - df_gantt['Start_Ref']

                        etiqueta_modo = "prom./ocurrencia" if modo_gantt == 'promedio_ciclo' else "acumulado"

                        if sim_seleccionada_global is None:
                            texto_barras_gantt = df_gantt.apply(
                                lambda row: f'{percentil_gantt_col}: {fmt2(row["Valor_Barra"])}h ({etiqueta_modo})', axis=1
                            )
                            hover_extra_gantt = df_gantt.apply(
                                lambda row: (
                                    f'P0: {fmt2(row["P0"])}h | P10: {fmt2(row["P10"])}h | P30: {fmt2(row["P30"])}h<br>'
                                    f'P50: {fmt2(row["P50"])}h | Esperanza: {fmt2(row["Esperanza"])}h | '
                                    f'P70: {fmt2(row["P70"])}h<br>'
                                    f'P90: {fmt2(row["P90"])}h | P100: {fmt2(row["P100"])}h<br>'
                                    f'Modo: {"Duración promedio por ocurrencia" if modo_gantt == "promedio_ciclo" else "Tiempo acumulado"}'
                                ), axis=1
                            )
                            hovertemplate_gantt = '<b>%{y}</b><br>%{text}<br>%{customdata}<extra></extra>'
                        else:
                            texto_barras_gantt = df_gantt.apply(lambda row: f'Duración: {fmt2(row["Valor_Barra"])}h ({etiqueta_modo})', axis=1)
                            hover_extra_gantt = df_gantt['Actividad'].apply(lambda _: '')
                            hovertemplate_gantt = '<b>%{y}</b><br>%{text}<extra></extra>'

                        fig_gantt.add_trace(go.Bar(
                            y=df_gantt['Actividad'],
                            x=df_gantt['Valor_Barra'],
                            base=df_gantt['Start_Barra'],
                            orientation='h',
                            marker_color=df_gantt['Color'],
                            text=texto_barras_gantt,
                            customdata=hover_extra_gantt,
                            hovertemplate=hovertemplate_gantt,
                            error_x=dict(
                                type='data',
                                symmetric=False,
                                array=df_gantt['Error_Plus'],
                                arrayminus=df_gantt['Error_Minus'],
                                color='grey'
                            ) if sim_seleccionada_global is None else None
                        ))

                        etiqueta_titulo_modo = "Duración Promedio por Ocurrencia" if modo_gantt == 'promedio_ciclo' else "Tiempo Acumulado"
                        if sim_seleccionada_global is None:
                            titulo_gantt = (
                                f'Carta Gantt (Túnel: {tunel_seleccionado}) - {etiqueta_titulo_modo} '
                                f'[{percentil_gantt_col}, con Rango P10-P90]'
                            )
                            eje_x_titulo = (
                                'Duración Promedio por Ocurrencia Acumulada (Horas)' if modo_gantt == 'promedio_ciclo'
                                else 'Tiempo Acumulado (Horas)'
                            )
                        else:
                            titulo_gantt = f'Carta Gantt (Túnel: {tunel_seleccionado}) - Simulación N° {sim_seleccionada_global} ({etiqueta_titulo_modo}, Determinística)'
                            eje_x_titulo = (
                                'Duración Promedio por Ocurrencia (Horas)' if modo_gantt == 'promedio_ciclo'
                                else 'Tiempo Acumulado (Horas)'
                            )

                        fig_gantt.update_layout(
                            barmode='stack',
                            title=titulo_gantt,
                            xaxis_title=eje_x_titulo,
                            yaxis_title='Actividad',
                            yaxis=dict(autorange="reversed"),
                            height=min(600, len(gantt_data) * 50 + 150),
                            showlegend=False
                        )
                        st.plotly_chart(fig_gantt, use_container_width=True)

                        if modo_gantt == 'promedio_ciclo':
                            st.caption(
                                "💡 Cada barra muestra la duración TÍPICA de una sola ocurrencia de esa "
                                "actividad/demora (tiempo total de la simulación ÷ N° de ciclos completados "
                                "en esa simulación). El eje de tiempo acumulado ya NO representa la duración "
                                "total del programa del túnel — para eso, usa el modo 'Tiempo acumulado'."
                            )
                    elif sim_seleccionada_global is not None:
                        st.info(f"La simulación N° {sim_seleccionada_global} no registró actividades/demoras para este túnel.")

                st.markdown("---")

                # ----------------------------------------------------
                # 14.1 Distribucion de Resultados Monte Carlo por Actividad/Demora
                # ----------------------------------------------------
                st.markdown("### Distribucion de Resultados Monte Carlo por Actividad/Demora")
                st.caption(
                    "Histograma con el conjunto completo de valores que arrojo la simulacion "
                    "(todas las ocurrencias, en todas las simulaciones), no solo el promedio. "
                    "Cuando la actividad tiene una distribucion de entrada definida, se superpone "
                    "para comparar la distribucion original contra el resultado simulado."
                )

                df_resultados_mc = df_tunel[df_tunel['type'].isin(
                    ['activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_equipment_failure', 'delay_fleet_change']
                )].copy()

                if df_resultados_mc.empty or 'Actividad' not in df_resultados_mc.columns:
                    st.info("No hay resultados de actividades o demoras para graficar en este tunel.")
                else:
                    actividades_mc = sorted(
                        df_resultados_mc['Actividad'].dropna().unique(),
                        key=lambda x: (0 if str(x).startswith('Demora') else 1, str(x))
                    )
                    actividad_mc_sel = st.selectbox(
                        "Seleccione Actividad o Demora para ver su distribucion Monte Carlo",
                        actividades_mc,
                        key="actividad_mc_sel"
                    )

                    valores_mc = df_resultados_mc.loc[
                        df_resultados_mc['Actividad'] == actividad_mc_sel, 'Duration'
                    ].dropna().astype(float).values

                    if len(valores_mc) == 0:
                        st.info("No hay valores de duracion para esta actividad/demora en este tunel.")
                    else:
                        pct_mc = percentiles_reporte(valores_mc)

                        fila_metricas_mc_1 = st.columns(5)
                        fila_metricas_mc_1[0].metric("Muestras (N)", f"{len(valores_mc)}")
                        fila_metricas_mc_1[1].metric("P0 - Mínimo (h)", fmt2(pct_mc['p0']))
                        fila_metricas_mc_1[2].metric("P10 (h)", fmt2(pct_mc['p10']))
                        fila_metricas_mc_1[3].metric("P30 (h)", fmt2(pct_mc['p30']))
                        fila_metricas_mc_1[4].metric("P50 - Mediana (h)", fmt2(pct_mc['p50']))

                        fila_metricas_mc_2 = st.columns(4)
                        fila_metricas_mc_2[0].metric("Esperanza (Promedio) (h)", fmt2(pct_mc['esperanza']))
                        fila_metricas_mc_2[1].metric("P70 (h)", fmt2(pct_mc['p70']))
                        fila_metricas_mc_2[2].metric("P90 (h)", fmt2(pct_mc['p90']))
                        fila_metricas_mc_2[3].metric("P100 - Máximo (h)", fmt2(pct_mc['p100']))

                        # Busca si la actividad tiene una distribucion de entrada definida
                        # (original o modificada) para superponerla con el resultado simulado
                        df_actividades_input = st.session_state.get('df_actividades_simuladas', pd.DataFrame())
                        info_dist_teorica = None
                        if not df_actividades_input.empty and 'Actividad' in df_actividades_input.columns:
                            coincidencias_act = df_actividades_input[
                                df_actividades_input['Actividad'] == actividad_mc_sel
                            ]
                            if not coincidencias_act.empty:
                                info_dist_teorica = coincidencias_act.iloc[0]

                        col_hist1, col_hist2 = st.columns(2)

                        # --- Gráfico 1: Función de Densidad (curva teórica continua + resultados
                        # de Monte Carlo como líneas verticales individuales, sin histograma) ---
                        with col_hist1:
                            # Rango de X: desde 0 hasta un poco mas del valor maximo observado
                            x_max_rango = float(np.max(valores_mc)) * 1.15
                            x_vals = np.linspace(0, x_max_rango, 400)

                            fig_dens_h = go.Figure()

                            hay_dist_teorica = (
                                info_dist_teorica is not None
                                and str(info_dist_teorica.get('Distribucion', 'Cte')) != 'Cte'
                                and isinstance(info_dist_teorica.get('Tiempo'), dict)
                            )

                            y_pdf_teorica = None
                            if hay_dist_teorica:
                                y_pdf_teorica = calcular_pdf_teorica(
                                    info_dist_teorica['Tiempo'], info_dist_teorica['Distribucion'], x_vals
                                )

                            # Altura de referencia para las lineas verticales (rug plot).
                            # Se ancla a la altura de la curva teorica si existe; si no, se usa
                            # una referencia generica basada en un KDE auxiliar (solo para escala,
                            # no se dibuja como curva).
                            if y_pdf_teorica is not None and np.max(y_pdf_teorica) > 0:
                                altura_ref = float(np.max(y_pdf_teorica))
                            elif len(valores_mc) >= 2 and np.std(valores_mc) > 1e-9:
                                altura_ref = float(np.max(stats.gaussian_kde(valores_mc)(x_vals)))
                            else:
                                altura_ref = 1.0
                            altura_lineas = max(altura_ref * 0.12, 1e-6)

                            # Lineas verticales: una por cada resultado individual de la simulacion
                            rug_x, rug_y = [], []
                            for v in valores_mc:
                                rug_x.extend([v, v, None])
                                rug_y.extend([0, altura_lineas, None])

                            fig_dens_h.add_trace(go.Scatter(
                                x=rug_x, y=rug_y,
                                mode='lines',
                                name='Resultado Simulacion (Monte Carlo)',
                                line=dict(color='rgba(31, 119, 180, 0.45)', width=1),
                                hoverinfo='skip'
                            ))

                            if hay_dist_teorica:
                                fig_dens_h.add_trace(go.Scatter(
                                    x=x_vals, y=y_pdf_teorica,
                                    mode='lines',
                                    name=f"Distribucion de Entrada ({info_dist_teorica['Distribucion']})",
                                    line=dict(color='rgb(255, 127, 14)', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(255, 127, 14, 0.35)'
                                ))
                            else:
                                st.caption(
                                    "Esta actividad/demora no tiene una distribucion de entrada asociada "
                                    "en la hoja de Actividades (por ejemplo, demoras por restricciones o "
                                    "fallas de equipos), por lo que solo se muestran los resultados simulados."
                                )

                            fig_dens_h.update_layout(
                                title=f'Función de Densidad - {actividad_mc_sel}',
                                xaxis_title='Duracion (horas)',
                                yaxis_title='Densidad de Probabilidad',
                                xaxis=dict(rangemode='tozero'),
                                yaxis=dict(rangemode='tozero'),
                                height=450,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                            )
                            st.plotly_chart(fig_dens_h, use_container_width=True)

                        # --- Gráfico 2: Histograma de CONTEO por intervalo de tiempo ---
                        with col_hist2:
                            fig_conteo = go.Figure()
                            fig_conteo.add_trace(go.Histogram(
                                x=valores_mc,
                                name='N° de Ocurrencias (Monte Carlo)',
                                marker_color='rgb(31, 119, 180)',
                                opacity=0.85
                            ))

                            fig_conteo.update_layout(
                                title=f'N° de Veces por Intervalo de Tiempo - {actividad_mc_sel}',
                                xaxis_title='Duracion (horas)',
                                yaxis_title='N° de Ocurrencias',
                                height=450,
                                bargap=0.02,
                                showlegend=False
                            )
                            st.plotly_chart(fig_conteo, use_container_width=True)

                        st.caption(
                            f"Tunel: {tunel_seleccionado}. Izquierda: densidad de probabilidad (barras horizontales), "
                            "comparando el resultado simulado contra la distribucion de entrada (si existe). "
                            "Derecha: conteo real de cuántas veces la simulacion cayó en cada intervalo de tiempo."
                        )

                st.markdown("---")

                # ----------------------------------------------------
                # 15. Traza de Eventos Detallada
                # ----------------------------------------------------
                st.subheader("15. Traza de Eventos Detallada")

                # ===== NUEVA PESTAÑA: VISTA POR RECURSO (COMPLETA) =====
                tab1, tab2 = st.tabs(["Vista por Túnel (Filtrada)", "Vista por Recurso (Completa)"])
                
                with tab1:
                    # ===== CÓDIGO ORIGINAL (VISTA FILTRADA POR TÚNEL) =====
                    col_sim, col_filtro = st.columns([1, 1])

                    with col_sim:
                        simulacion_detallada = st.selectbox(
                            "Seleccione el ID de la Simulación para la traza", 
                            df_traza_completa['Simulation_ID'].unique(), 
                            index=0,
                            key="sim_filtrada"
                        )

                    with col_filtro:
                        # Filtro adicional por ciclo
                        ciclos_disponibles = sorted(df_traza_completa['Cycle'].unique())
                        ver_todos_ciclos = st.checkbox("Ver todos los ciclos", value=True, key="ciclos_filtrada")
                        
                        if not ver_todos_ciclos:
                            ciclo_seleccionado = st.selectbox(
                                "Seleccione un Ciclo específico",
                                ciclos_disponibles,
                                key="ciclo_filtrada"
                            )

                    if simulacion_detallada:
                        # Filtrar por simulación Y TÚNEL SELECCIONADO
                        df_traza_sim = df_traza_completa[
                            (df_traza_completa['Simulation_ID'] == simulacion_detallada) & 
                            (
                                (df_traza_completa['Resource_Origin_Tunnel'] == tunel_seleccionado) |
                                (df_traza_completa['Resource_Destination_Tunnel'] == tunel_seleccionado)
                            )
                        ].copy()
                        
                        # Filtrar por ciclo si no se quieren ver todos
                        if not ver_todos_ciclos:
                            df_traza_sim = df_traza_sim[df_traza_sim['Cycle'] == ciclo_seleccionado]
                        
                        # ORDENAR POR TIEMPO DE INICIO (CRÍTICO)
                        df_traza_sim = df_traza_sim.sort_values(by='Start').reset_index(drop=True)
                        
                        # Agregar número de secuencia temporal
                        df_traza_sim['Secuencia_Temporal'] = range(1, len(df_traza_sim) + 1)

                        df_output = df_traza_sim[[
                            'Secuencia_Temporal', 'Cycle', 'Actividad', 'Recurso', 'type', 'Start', 'Finish', 'Duration',
                            'Start_X_Front', 'Start_Y_Front', 'End_X_Front', 'End_Y_Front',
                            'Cycle_Advance_Length_m', 'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                            'Travel_Time_Actual', 'Travel_Speed_Used_m_h', 'Travel_Distance_m'
                        ]].rename(columns={
                            'Secuencia_Temporal': '#',
                            'Cycle': 'Ciclo', 'type': 'Tipo Evento', 'Start': 'Inicio (h)', 'Finish': 'Fin (h)',
                            'Duration': 'Duración (h)', 'Start_X_Front': 'Inicio X', 'Start_Y_Front': 'Inicio Y',
                            'End_X_Front': 'Fin X', 'End_Y_Front': 'Fin Y', 'Cycle_Advance_Length_m': 'Avance Ciclo (m)',
                            'Resource_Origin_Tunnel': 'Túnel Origen', 'Resource_Destination_Tunnel': 'Túnel Destino',
                            'Travel_Time_Actual': 'Tiempo Viaje (h)', 'Travel_Speed_Used_m_h': 'Velocidad Viaje (m/h)',
                            'Travel_Distance_m': 'Distancia Viaje (m)'
                        })

                        df_output['Tipo Evento'] = df_output['Tipo Evento'].map({
                            'travel': 'VIAJE', 
                            'activity': 'ACTIVIDAD', 
                            'delay_fh_oc': 'Demora (FH/OC)', 
                            'delay_res': 'Demora (R-Turno)',
                            'delay_equipment_failure': 'Demora (Falla Equipo)',
                            'delay_fleet_change': 'Demora (Cambio de Recurso)'
                        }).fillna(df_output['Tipo Evento'])
                        
                        # Columna visual para identificar eventos del túnel seleccionado
                        df_output['Ubicación'] = df_output.apply(
                            lambda row: f'{tunel_seleccionado}' if row['Túnel Destino'] == tunel_seleccionado 
                            else f'{row["Túnel Destino"]}' if row['Túnel Destino'] != 'N/A'
                            else 'Demora',
                            axis=1
                        )

                        # Estilo condicional mejorado
                        def highlight_events(row):
                            if '' in str(row['Ubicación']):
                                return ['background-color: #d4edda'] * len(row)  # Verde: en túnel seleccionado
                            elif row['Tipo Evento'] == 'VIAJE':
                                return ['background-color: #fff3cd'] * len(row)  # Amarillo: viajando
                            elif 'Demora' in str(row['Tipo Evento']):
                                return ['background-color: #f8d7da'] * len(row)  # Rojo claro: demora
                            else:
                                return ['background-color: #e7f3ff'] * len(row)  # Azul claro: otro túnel
                        
                        st.markdown(f"#### Eventos en **{tunel_seleccionado}** (Orden Cronológico)")
                        st.caption(f"Mostrando {len(df_output)} eventos relacionados con este túnel")
                        
                        # Nota: el Styler de pandas usa por defecto 6 decimales al mostrar
                        # columnas numericas, aunque el valor ya este redondeado a 2. Se fuerza
                        # el formato con fmt2 (maximo 2 decimales, con cifras significativas
                        # para valores muy pequeños) para que se vea limpio.
                        cols_numericas_output = df_output.select_dtypes(include=[np.number]).columns.tolist()
                        st.dataframe(
                            df_output.style.apply(highlight_events, axis=1).format(fmt2, subset=cols_numericas_output),
                            use_container_width=True, 
                            hide_index=True,
                            height=600
                        )
                        
                        st.info(
                            f"""
                            **Nota**: Esta vista muestra solo eventos donde **{tunel_seleccionado}** es origen o destino.
                            Para ver el recorrido completo de recursos entre TODOS los túneles, usa la pestaña **"Vista por Recurso"**.
                            """
                        )
                
                # ===== NUEVA TAB: VISTA COMPLETA POR RECURSO =====
                #

                # ===== NUEVA TAB: VISTA COMPLETA POR RECURSO =====
                with tab2:
                    st.markdown("### Seguimiento Completo de Recursos entre Tuneles")
                    
                    col_sim2, col_rec = st.columns([1, 1])
                    
                    with col_sim2:
                        simulacion_recurso = st.selectbox(
                            "Seleccione el ID de la Simulacion", 
                            df_traza_completa['Simulation_ID'].unique(), 
                            index=0,
                            key="sim_recurso"
                        )
                    
                    with col_rec:
                        # Obtener recursos únicos (sin N/A)
                        recursos_disponibles = sorted(
                            df_traza_completa[
                                (df_traza_completa['Recurso'].notna()) & 
                                (df_traza_completa['Recurso'] != 'N/A')
                            ]['Recurso'].unique()
                        )
                        
                        if recursos_disponibles:
                            recurso_seleccionado = st.selectbox(
                                "Seleccione el Recurso a seguir",
                                recursos_disponibles,
                                key="recurso_select"
                            )
                        else:
                            st.warning("No hay recursos con identificador en la traza")
                            recurso_seleccionado = None
                    
                    if simulacion_recurso and recurso_seleccionado:
                        # Filtrar SOLO por simulación y recurso (SIN filtro de túnel)
                        df_recurso_completo = df_traza_completa[
                            (df_traza_completa['Simulation_ID'] == simulacion_recurso) &
                            (df_traza_completa['Recurso'] == recurso_seleccionado)
                        ].copy()
                        
                        # Ordenar cronológicamente
                        df_recurso_completo = df_recurso_completo.sort_values(by='Start').reset_index(drop=True)
                        df_recurso_completo['Secuencia'] = range(1, len(df_recurso_completo) + 1)
                        
                        # Preparar tabla de salida
                        df_output_recurso = df_recurso_completo[[
                            'Secuencia', 'Cycle', 'Actividad', 'type', 'Start', 'Finish', 'Duration',
                            'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                            'Travel_Distance_m', 'Travel_Speed_Used_m_h'
                        ]].rename(columns={
                            'Secuencia': '#',
                            'Cycle': 'Ciclo',
                            'type': 'Tipo',
                            'Start': 'Inicio (h)',
                            'Finish': 'Fin (h)',
                            'Duration': 'Duracion (h)',
                            'Resource_Origin_Tunnel': 'Tunel Origen',
                            'Resource_Destination_Tunnel': 'Tunel Destino',
                            'Travel_Distance_m': 'Distancia (m)',
                            'Travel_Speed_Used_m_h': 'Velocidad (m/h)'
                        })
                        
                        # Traducir tipos de evento
                        df_output_recurso['Tipo'] = df_output_recurso['Tipo'].map({
                            'travel': 'VIAJE',
                            'activity': 'ACTIVIDAD',
                            'delay_fh_oc': 'Demora FH/OC',
                            'delay_res': 'Demora Turno',
                            'delay_equipment_failure': 'Demora Falla',
                            'delay_fleet_change': 'Demora Cambio Recurso'
                        }).fillna(df_output_recurso['Tipo'])
                        
                        # Crear columna resumen del movimiento
                        df_output_recurso['Movimiento'] = df_output_recurso.apply(
                            lambda row: f"{row['Tunel Origen']} -> {row['Tunel Destino']}" 
                            if row['Tipo'] == 'VIAJE' 
                            else f"Trabajando en {row['Tunel Destino']}",
                            axis=1
                        )
                        
                        # Estilo condicional por tipo de evento
                        def highlight_recurso_events(row):
                            if 'VIAJE' in str(row['Tipo']):
                                return ['background-color: #fff3cd'] * len(row)
                            elif 'ACTIVIDAD' in str(row['Tipo']):
                                return ['background-color: #d1ecf1'] * len(row)
                            elif 'Demora' in str(row['Tipo']):
                                return ['background-color: #f8d7da'] * len(row)
                            else:
                                return [''] * len(row)
                        
                        st.markdown(f"#### Recorrido de {recurso_seleccionado} (Simulacion #{simulacion_recurso})")
                        st.caption(f"Total de eventos: {len(df_output_recurso)} | Tuneles visitados: {df_output_recurso['Tunel Destino'].nunique()}")
                        
                        # Mostrar tabla con estilo
                        df_output_recurso_vista = df_output_recurso[[
                            '#', 'Ciclo', 'Tipo', 'Actividad', 'Movimiento', 
                            'Inicio (h)', 'Fin (h)', 'Duracion (h)', 
                            'Distancia (m)', 'Velocidad (m/h)'
                        ]]
                        cols_numericas_recurso = df_output_recurso_vista.select_dtypes(include=[np.number]).columns.tolist()
                        st.dataframe(
                            df_output_recurso_vista.style.apply(highlight_recurso_events, axis=1).format(fmt2, subset=cols_numericas_recurso),
                            use_container_width=True,
                            hide_index=True,
                            height=600
                        )
                        
                        # Metricas resumen
                        st.markdown("---")
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            tiempo_trabajando = df_recurso_completo[
                                df_recurso_completo['type'] == 'activity'
                            ]['Duration'].sum()
                            st.metric("Tiempo Trabajando", f"{tiempo_trabajando:.2f} h")
                        
                        with col_m2:
                            tiempo_viajando = df_recurso_completo[
                                df_recurso_completo['type'] == 'travel'
                            ]['Duration'].sum()
                            st.metric("Tiempo Viajando", f"{tiempo_viajando:.2f} h")
                        
                        with col_m3:
                            distancia_total = df_recurso_completo[
                                df_recurso_completo['type'] == 'travel'
                            ]['Travel_Distance_m'].sum()
                            st.metric("Distancia Total", f"{distancia_total:.0f} m")
                        
                        with col_m4:
                            tuneles_unicos = df_recurso_completo['Resource_Destination_Tunnel'].nunique()
                            st.metric("Tuneles Visitados", tuneles_unicos)
                        
                        # Grafico de linea de tiempo
                        st.markdown("---")
                        st.markdown("#### Linea de Tiempo Visual")
                        
                        fig_timeline_recurso = go.Figure()
                        
                        colors_tipo = {
                            'travel': '#FFC107',
                            'activity': '#4CAF50',
                            'delay_fh_oc': '#F44336',
                            'delay_res': '#FF5722',
                            'delay_equipment_failure': '#9C27B0',
                            'delay_fleet_change': '#E91E63'
                        }
                        
                        for idx, row in df_recurso_completo.iterrows():
                            color = colors_tipo.get(row['type'], '#2196F3')
                            
                            # Texto del hover con información completa
                            if row['type'] == 'travel':
                                hover_text = (
                                    f"<b>VIAJE</b><br>"
                                    f"De: {row['Resource_Origin_Tunnel']}<br>"
                                    f"Hacia: {row['Resource_Destination_Tunnel']}<br>"
                                    f"Distancia: {row['Travel_Distance_m']:.0f} m<br>"
                                    f"Velocidad: {row['Travel_Speed_Used_m_h']:.0f} m/h<br>"
                                    f"Duracion: {row['Duration']:.2f} h<br>"
                                    f"Inicio: {row['Start']:.2f} h"
                                )
                            else:
                                hover_text = (
                                    f"<b>{row['Actividad']}</b><br>"
                                    f"Tunel: {row['Resource_Destination_Tunnel']}<br>"
                                    f"Ciclo: {row['Cycle']}<br>"
                                    f"Duracion: {row['Duration']:.2f} h<br>"
                                    f"Inicio: {row['Start']:.2f} h<br>"
                                    f"Fin: {row['Finish']:.2f} h"
                                )
                            
                            fig_timeline_recurso.add_trace(go.Bar(
                                x=[row['Duration']],
                                y=[recurso_seleccionado],
                                base=[row['Start']],
                                orientation='h',
                                marker=dict(color=color),
                                text=row['Resource_Destination_Tunnel'] if row['type'] == 'activity' else 'Viaje',
                                hovertext=hover_text,
                                hoverinfo='text',
                                showlegend=False
                            ))
                        
                        fig_timeline_recurso.update_layout(
                            title=f'Linea de Tiempo Completa: {recurso_seleccionado}',
                            xaxis_title='Tiempo (Horas)',
                            yaxis_title='',
                            height=250,
                            barmode='overlay',
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_timeline_recurso, use_container_width=True)
                        
                        # Analisis de utilizacion por tunel
                        st.markdown("---")
                        st.markdown("#### Tiempo de Trabajo por Tunel")
                        
                        tiempo_por_tunel = df_recurso_completo[
                            df_recurso_completo['type'] == 'activity'
                        ].groupby('Resource_Destination_Tunnel')['Duration'].sum().sort_values(ascending=False)
                        
                        if not tiempo_por_tunel.empty:
                            fig_tuneles = px.bar(
                                x=tiempo_por_tunel.values,
                                y=tiempo_por_tunel.index,
                                orientation='h',
                                labels={'x': 'Tiempo Total (h)', 'y': 'Tunel'},
                                title=f'Distribucion de Trabajo de {recurso_seleccionado}',
                                color=tiempo_por_tunel.values,
                                color_continuous_scale='Blues'
                            )
                            fig_tuneles.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig_tuneles, use_container_width=True)
                        
                        st.info(
                            """
                            Verificacion de Liberacion Inmediata: Este recurso trabajo en multiples tuneles diferentes. 
                            Cada actividad esta seguida inmediatamente por un viaje o una nueva actividad sin tiempos muertos.
                            """
                        )

                # ===== SECCION DE EXPORTACION A EXCEL =====
                st.markdown("---")
                st.header("16. Exportacion de Resultados a Excel")

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    generar_zip_btn = st.button(
                        "Generar ZIP (6 archivos formato David Montenegro)",
                        type="primary",
                        use_container_width=True,
                        help="Genera los 6 archivos Excel en memoria y los empaqueta en un .zip para descargar."
                    )
                with col_exp2:
                    descargar_btn = st.button(
                        "Generar Reporte Descargable (1 archivo combinado)",
                        type="secondary",
                        use_container_width=True
                    )

                if generar_zip_btn:
                    with st.spinner("Generando archivos..."):
                        try:
                            zip_buf, nombres_archivos = generar_outputs_zip_en_memoria(
                                traza_eventos=st.session_state.traza_eventos,
                                resultados=st.session_state.resultados,
                                frentes_info=st.session_state.frentes_info,
                                df_frentes_nivel=st.session_state.get('df_frentes_nivel_simuladas', frentes_nivel),
                                df_actividades=st.session_state.get('df_actividades_simuladas', actividades_nivel),
                                recursos_config=st.session_state.recursos_config,
                                numero_turnos=st.session_state.get('numero_turnos_simulado', st.session_state.get('numero_turnos', 62)),
                                duracion_turno=st.session_state.get('duracion_turno_simulada', st.session_state.get('duracion_turno_base', 8.0)),
                                n_simulaciones=st.session_state.get('n_simulaciones_ejecutadas', int(st.session_state.get('n_simulaciones', 1000))),
                                nivel_seleccionado=st.session_state.get('nivel_simulado', nivel_seleccionado)
                            )
                            st.success(f"{len(nombres_archivos)} archivos listos:")
                            for nombre in nombres_archivos:
                                st.text(f"  • {nombre}")
                            nombre_nivel_zip = st.session_state.get('nivel_simulado', nivel_seleccionado)
                            st.download_button(
                                label="⬇️ Descargar ZIP",
                                data=zip_buf,
                                file_name=f"Resultados_David_Montenegro_{nombre_nivel_zip}.zip",
                                mime="application/zip",
                                type="primary",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error al generar los archivos: {e}")

                if descargar_btn:
                    with st.spinner("Generando reporte Excel..."):
                        output = BytesIO()
                        tuneles_a_exportar = frentes_seleccionados
                        
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            # HOJA 1: Resumen General
                            resumen_general = []
                            resumen_general.append(['REPORTE DE SIMULACION DE TUNELES', ''])
                            resumen_general.append(['Nivel Seleccionado', nivel_seleccionado])
                            resumen_general.append(['Sistema de Turnos', st.session_state.get('sistema_turnos', '')])
                            resumen_general.append(['Numero de Turnos', st.session_state.get('numero_turnos', '')])
                            resumen_general.append(['Tiempo Limite (horas)', st.session_state.get('tiempo_limite', 0)])
                            resumen_general.append(['Metros por Ciclo por Defecto', st.session_state.get('metros_avance', 3.5)])
                            resumen_general.append(['Numero de Simulaciones', st.session_state.get('n_simulaciones', 1000)])
                            resumen_general.append(['Ruta Critica', ", ".join(st.session_state.get('ruta_critica', []))])
                            resumen_general.append(['Radio FH/OC (m)', radio_restriccion])
                            resumen_general.append(['Demora por Restriccion (h)', demora_horas])
                            resumen_general.append(['', ''])
                            resumen_general.append(['TUNELES SIMULADOS', 'DISTANCIA TOTAL (m)', 'SECCION', 'METROS POR CICLO'])
                            for frente, info in st.session_state.frentes_info.items():
                                resumen_general.append([
                                    frente,
                                    round(info['distancia'], 2),
                                    info.get('seccion', ''),
                                    info.get('metros_por_ciclo', st.session_state.get('metros_avance', 3.5))
                                ])
                            
                            df_resumen = pd.DataFrame(resumen_general)
                            df_resumen.to_excel(writer, sheet_name='Resumen General', index=False, header=False)

                            pd.DataFrame(
                                [{'Seccion': k, 'Metros por Ciclo': v} for k, v in st.session_state.get('metros_por_seccion', {}).items()]
                            ).to_excel(writer, sheet_name='Metros por Seccion', index=False)
                            pd.DataFrame(
                                [{'Tipo': k, 'Radio (m)': v} for k, v in st.session_state.get('radios_por_tipo', {}).items()]
                            ).to_excel(writer, sheet_name='Radios Restricciones', index=False)
                            if st.session_state.get('restricciones_geologicas'):
                                pd.DataFrame(st.session_state.restricciones_geologicas).to_excel(writer, sheet_name='Restricciones Geo', index=False)
                            if st.session_state.get('fallas_equipos'):
                                pd.DataFrame(st.session_state.fallas_equipos).to_excel(writer, sheet_name='Fallas Equipos', index=False)
                            if not st.session_state.get('df_marina', pd.DataFrame()).empty:
                                st.session_state.df_marina.to_excel(writer, sheet_name='Vaciaderos Marina', index=False)
                            if not st.session_state.get('df_muckpile', pd.DataFrame()).empty:
                                st.session_state.df_muckpile.to_excel(writer, sheet_name='Muckpile Stockpile', index=False)
                            if st.session_state.get('cambios_recursos'):
                                pd.DataFrame(st.session_state.cambios_recursos).to_excel(writer, sheet_name='Cambios Recursos Turno', index=False)
                            if st.session_state.get('registro_cambios_flota'):
                                pd.DataFrame(st.session_state.registro_cambios_flota).to_excel(writer, sheet_name='Cambios Flota Aplicados', index=False)

                            # HOJA 2: Avance Final por Tunel
                            df_resultados.to_excel(writer, sheet_name='Avance Final por Tunel', index=False)
                            
                            # HOJA 3: Avance por Turno
                            if datos_turno:
                                df_turnos.to_excel(writer, sheet_name='Avance por Turno', index=False)
                            
                            # HOJA 4: Utilizacion de Recursos
                            if datos_recursos:
                                df_recursos_res.to_excel(writer, sheet_name='Utilizacion de Recursos', index=False)
                            
                            # HOJA 5-N: Detalle por Simulacion y Tunel
                            for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                for tunel in tuneles_a_exportar:
                                    df_sim_tunel = df_traza_completa[
                                        (df_traza_completa['Simulation_ID'] == sim_id) &
                                        ((df_traza_completa['Resource_Origin_Tunnel'] == tunel) |
                                         (df_traza_completa['Resource_Destination_Tunnel'] == tunel))
                                    ].copy()
                                    
                                    if not df_sim_tunel.empty:
                                        df_sim_tunel = df_sim_tunel.sort_values(by='Start')
                                        
                                        # Crear hoja con nombre limitado
                                        sheet_name = f"S{sim_id}_{tunel}"[:31]
                                        
                                        df_export = df_sim_tunel[[
                                            'Cycle', 'Actividad', 'Recurso', 'type', 'Start', 'Finish', 'Duration',
                                            'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                                            'Travel_Distance_m', 'Travel_Speed_Used_m_h', 'Cycle_Advance_Length_m'
                                        ]].copy()
                                        
                                        df_export.columns = [
                                            'Ciclo', 'Actividad', 'Recurso', 'Tipo', 'Inicio (h)', 'Fin (h)', 
                                            'Duracion (h)', 'Tunel Origen', 'Tunel Destino', 'Distancia Viaje (m)',
                                            'Velocidad (m/h)', 'Avance Ciclo (m)'
                                        ]
                                        
                                        df_export.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # HOJA: Resumen de Actividades por Tunel
                            resumen_actividades_data = []
                            for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                for tunel in tuneles_a_exportar:
                                    df_tunel_sim = df_traza_completa[
                                        (df_traza_completa['Simulation_ID'] == sim_id) &
                                        (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                        (df_traza_completa['type'].isin(['activity', 'delay_fh_oc', 'delay_res', 'delay_equipment_failure', 'delay_fleet_change', 'blocked_no_resource']))
                                    ]
                                    
                                    if not df_tunel_sim.empty:
                                        for actividad in df_tunel_sim['Actividad'].unique():
                                            df_act = df_tunel_sim[df_tunel_sim['Actividad'] == actividad]
                                            tiempo_total = df_act['Duration'].sum()
                                            num_ejecuciones = len(df_act)
                                            tiempo_promedio = tiempo_total / num_ejecuciones if num_ejecuciones > 0 else 0
                                            
                                            resumen_actividades_data.append({
                                                'Simulacion': sim_id,
                                                'Tunel': tunel,
                                                'Actividad': actividad,
                                                'Num Ejecuciones': num_ejecuciones,
                                                'Tiempo Total (h)': round(tiempo_total, 2),
                                                'Tiempo Promedio (h)': round(tiempo_promedio, 2)
                                            })
                            
                            if resumen_actividades_data:
                                df_resumen_act = pd.DataFrame(resumen_actividades_data)
                                df_resumen_act.to_excel(writer, sheet_name='Resumen Actividades', index=False)
                            
                            # HOJA: Resumen de Viajes por Recurso
                            resumen_viajes_data = []
                            for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                df_viajes_sim = df_traza_completa[
                                    (df_traza_completa['Simulation_ID'] == sim_id) &
                                    (df_traza_completa['type'] == 'travel')
                                ].copy()
                                
                                if not df_viajes_sim.empty:
                                    df_viajes_sim['Recurso_Tipo'] = df_viajes_sim['Recurso'].apply(
                                        lambda x: x.split('_')[0] if pd.notna(x) else 'N/A'
                                    )
                                    
                                    for recurso_tipo in df_viajes_sim['Recurso_Tipo'].unique():
                                        df_rec = df_viajes_sim[df_viajes_sim['Recurso_Tipo'] == recurso_tipo]
                                        
                                        resumen_viajes_data.append({
                                            'Simulacion': sim_id,
                                            'Recurso': recurso_tipo,
                                            'Num Viajes': len(df_rec),
                                            'Distancia Total (m)': round(df_rec['Travel_Distance_m'].sum(), 2),
                                            'Tiempo Total Viaje (h)': round(df_rec['Travel_Time_Actual'].sum(), 2),
                                            'Velocidad Promedio (m/h)': round(df_rec['Travel_Speed_Used_m_h'].mean(), 2)
                                        })
                            
                            if resumen_viajes_data:
                                df_resumen_viajes = pd.DataFrame(resumen_viajes_data)
                                df_resumen_viajes.to_excel(writer, sheet_name='Resumen Viajes', index=False)
                            
                            # HOJA: Ciclos Completados por Tunel
                            ciclos_data = []
                            for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                for tunel in tuneles_a_exportar:
                                    df_tunel_sim = df_traza_completa[
                                        (df_traza_completa['Simulation_ID'] == sim_id) &
                                        (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                        (df_traza_completa['type'] == 'activity')
                                    ]
                                    
                                    if not df_tunel_sim.empty:
                                        ciclos_completados = df_tunel_sim['Cycle'].max()
                                        avance_total = st.session_state.resultados.get(tunel, [0])[sim_id - 1] if sim_id <= len(st.session_state.resultados.get(tunel, [])) else 0
                                        
                                        ciclos_data.append({
                                            'Simulacion': sim_id,
                                            'Tunel': tunel,
                                            'Ciclos Completados': ciclos_completados,
                                            'Avance Total (m)': round(avance_total, 2),
                                            'Distancia Objetivo (m)': round(st.session_state.frentes_info[tunel]['distancia'], 2),
                                            'Porcentaje Completado (%)': round((avance_total / st.session_state.frentes_info[tunel]['distancia']) * 100, 2)
                                        })
                            
                            if ciclos_data:
                                df_ciclos = pd.DataFrame(ciclos_data)
                                df_ciclos.to_excel(writer, sheet_name='Ciclos Completados', index=False)
                            
                            # HOJA: Estadisticas de Demoras
                            demoras_data = []
                            for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                for tunel in tuneles_a_exportar:
                                    df_demoras = df_traza_completa[
                                        (df_traza_completa['Simulation_ID'] == sim_id) &
                                        (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                        (df_traza_completa['type'].isin(['delay_fh_oc', 'delay_res', 'delay_equipment_failure', 'delay_fleet_change']))
                                    ]
                                    
                                    if not df_demoras.empty:
                                        for tipo_demora in df_demoras['type'].unique():
                                            df_tipo = df_demoras[df_demoras['type'] == tipo_demora]
                                            
                                            demoras_data.append({
                                                'Simulacion': sim_id,
                                                'Tunel': tunel,
                                                'Tipo Demora': tipo_demora,
                                                'Num Ocurrencias': len(df_tipo),
                                                'Tiempo Total Demora (h)': round(df_tipo['Duration'].sum(), 2),
                                                'Tiempo Promedio (h)': round(df_tipo['Duration'].mean(), 2)
                                            })
                            
                            if demoras_data:
                                df_demoras_excel = pd.DataFrame(demoras_data)
                                df_demoras_excel.to_excel(writer, sheet_name='Estadisticas Demoras', index=False)

                            # ---- HOJAS EQUIVALENTES A ARCHIVOS DAVID MONTENEGRO ----
                            # Advances_total_scenary (equivalente)
                            df_adv_total = generar_advances_total_scenary(
                                st.session_state.resultados,
                                st.session_state.frentes_info,
                                st.session_state.get('nivel_simulado', nivel_seleccionado)
                            )
                            df_adv_total.to_excel(writer, sheet_name='Advances_Total_Scenary', index=False)

                            # Simulation_Front_Actual (una hoja por escenario, solo primeros 5 para el zip)
                            n_sim_exec = st.session_state.get('n_simulaciones_ejecutadas', 1)
                            dur_turno_exec = st.session_state.get('duracion_turno_simulada', st.session_state.get('duracion_turno_base', 8.0))
                            n_turnos_exec = st.session_state.get('numero_turnos_simulado', st.session_state.get('numero_turnos', 62))

                            sheets_front_act = generar_simulation_front_actual(
                                st.session_state.traza_eventos,
                                st.session_state.frentes_info,
                                st.session_state.get('df_frentes_nivel_simuladas', frentes_nivel),
                                min(n_sim_exec, 10)
                            )
                            for sname, sdf in sheets_front_act.items():
                                sdf.to_excel(writer, sheet_name=f'FrontActual_{sname}'[:31], index=False)

                            # Time Program (primeros 5 escenarios para no sobrecargar)
                            sheets_time_prog = generar_simulation_time_program(
                                st.session_state.traza_eventos,
                                st.session_state.recursos_config,
                                n_turnos_exec,
                                dur_turno_exec,
                                min(n_sim_exec, 5),
                                st.session_state.get('ventanas_flota', {})
                            )
                            for sname, sdf in sheets_time_prog.items():
                                sdf.to_excel(writer, sheet_name=f'TimeProgram_{sname}'[:31], index=False)

                        output.seek(0)

                        nombre_nivel = st.session_state.get('nivel_simulado', nivel_seleccionado)
                        st.download_button(
                            label="⬇️ Descargar Reporte Excel Completo",
                            data=output,
                            file_name=f"Reporte_Simulacion_{nombre_nivel}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True
                        )

                        st.success("Reporte Excel generado. Haga clic en el botón para descargar.")
                        st.markdown("#### Contenido del Reporte (hojas):")
                        contenido_reporte = [
                            "Resumen General — Parámetros de simulación",
                            "Avance Final por Tunel — P0, P10, P30, P50, Esperanza, P70, P90, P100 por frente",
                            "Avance por Turno — Progreso temporal acumulado",
                            "Utilizacion de Recursos — Uso y eficiencia",
                            "Detalle por Simulacion/Tunel — Eventos ordenados cronológicamente",
                            "Resumen Actividades — Tiempos por actividad y túnel",
                            "Resumen Viajes — Estadísticas de desplazamiento de recursos",
                            "Ciclos Completados — Avance y % de completitud",
                            "Estadisticas Demoras — Análisis de tiempos perdidos",
                            "Advances_Total_Scenary — Avance por frente/escenario (formato David Montenegro)",
                            "FrontActual_scenary N — Posiciones finales por escenario",
                            "TimeProgram_scenary N — Tiempo disponible por recurso/turno",
                            "Vaciaderos Marina — Sitios de depósito (si se cargó Input_Marina)",
                            "Muckpile Stockpile — Estado del stockpile (si se cargó Muckpile)",
                            "Cambios Recursos Turno — Cambios de cantidad de equipos por turno"
                        ]
                        for i, item in enumerate(contenido_reporte, 1):
                            st.text(f"{i}. {item}")
