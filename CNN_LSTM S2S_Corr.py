# -*- coding: utf-8 -*-
"""
==============================================================================
PIPELINE UNIFICADO DE PREDICCIÓN DE DEMANDA ELÉCTRICA
==============================================================================

Soporta resolución temporal "15min" y "1h" desde una única sección CONFIG.
Arquitectura CNN-LSTM seq2seq con corrector residual (HistGradientBoosting).

Estructura del código:
    1.  CONFIG              — toda la configuración en un único lugar
    2.  Imports y semillas
    3.  Utilidades generales
    4.  Carga de datos
    5.  Preprocesamiento
    6.  Feature engineering
    7.  Construcción de ventanas
    8.  Escalado y split temporal
    9.  Modelo base CNN-LSTM
    10. Corrector residual day-ahead
    11. Evaluación y métricas
    12. Visualización
    13. Guardado de resultados
    14. main()

CÓMO CAMBIAR LA RESOLUCIÓN:
    Modificar CONFIG["RESOLUTION"] a "15min" o "1h".
    El resto del pipeline se adapta automáticamente.

CÓMO AGREGAR HIPERPARÁMETROS:
    Agregar la clave en CONFIG["MODEL"] o CONFIG["TRAINING"]
    y referenciarla en la función correspondiente.
==============================================================================
"""


# ==============================================================================
# SECCIÓN 1 — CONFIGURACIÓN CENTRALIZADA
# ==============================================================================

CONFIG = {

    # --------------------------------------------------------------------------
    # Resolución temporal.
    # Opciones: "15min" | "1h"
    # --------------------------------------------------------------------------
    "RESOLUTION": "15min",

    # --------------------------------------------------------------------------
    # Rutas de datos
    # --------------------------------------------------------------------------
    "ENERGY_PATH": "./Data_Ibarlucea/",
    "SMN_PATH": "./SMN",

    # --------------------------------------------------------------------------
    # Nombres de columnas target
    # --------------------------------------------------------------------------
    "TARGET_RAW": "Potencia Activa",
    "TARGET_CLEAN": "Potencia Activa_clean",

    # --------------------------------------------------------------------------
    # Horizonte y ventana histórica expresados en DÍAS.
    # El código los convierte a pasos según la resolución.
    #
    # Ejemplos:
    #   RESOLUTION="1h",   HORIZON_DAYS=1  → 24 pasos
    #   RESOLUTION="15min",HORIZON_DAYS=1  → 96 pasos
    #   RESOLUTION="1h",   WINDOW_DAYS=4   → 96 pasos
    #   RESOLUTION="15min",WINDOW_DAYS=7   → 672 pasos
    # --------------------------------------------------------------------------
    "HORIZON_DAYS": 3,
    "WINDOW_DAYS": 4,       # para "1h"  cambia a 4; para "15min" suele ser 7

    # --------------------------------------------------------------------------
    # Features a activar / desactivar
    # --------------------------------------------------------------------------
    "FEATURES": {
        "use_meteo": True,        # variables meteorológicas en X pasado
        "use_meteo_future": True, # meteo/calendario en X futuro (forecast horizon)
        "use_cyclic_time": True,  # encodings seno/coseno de hora, dow, mes
        "use_calendar_raw": True, # hora, dow, mes, dia_mes, dia_anio como entero
        "use_holidays": True,     # feriado argentino (0/1)
        "use_minute": True,       # minuto del día (solo relevante en 15min)
        # Columnas de meteo a excluir del input futuro
        # (DD y FF son dirección/velocidad de viento, poco predecibles a futuro)
        "meteo_future_drop": ["DD", "FF"],
    },

    # --------------------------------------------------------------------------
    # Limpieza de outliers del target
    # --------------------------------------------------------------------------
    "OUTLIER": {
        "window": 24,   # ventana de rolling en pasos (se aplica antes de resampleo)
        "k": 2.0,       # límite en múltiplos de desvío estándar
    },

    # --------------------------------------------------------------------------
    # Split train / val / test
    # --------------------------------------------------------------------------
    "SPLIT": {
        "test_ratio": 0.30,
        "val_ratio_from_train": 0.20,
    },

    # --------------------------------------------------------------------------
    # Arquitectura del modelo base
    # --------------------------------------------------------------------------
    "MODEL": {
        "type": "cnn_lstm",     # actualmente soportado: "cnn_lstm"
        # Rama CNN (encoder pasado)
        "cnn_filters": 32,
        # Una rama Conv1D por cada Kernel Size
        # El tamaño de filtro determina la cantidad de muestras que "mira"
        # Por lo tanto configurar en función de la resolucón (15 min o 1 h)
        "cnn_kernel_1h": [6, 12, 24],
        "cnn_kernel_15min": [24, 48, 96],
        "cnn_dilations": 1,   
        # LSTM encoder (después de CNN)
        "lstm_enc_units": 64,
        "lstm_enc_dropout": 0.2,
        "lstm_enc_recurrent_dropout": 0.2,
        # LSTM futuro
        "lstm_fut_units": 32,
        # LSTM decoder
        "lstm_dec_units": 64,
        # Capa densa intermedia del decoder
        "dense_dec_units": 32,
        "dense_dec_activation": "relu",
    },

    # --------------------------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------------------------
    "TRAINING": {
        "epochs": 40,
        "batch_size": 64,       # 128 para 15min si hay memoria suficiente
        "learning_rate": 0.0005,
        "optimizer": "adam",    # "adam" | "rmsprop" | "sgd"
        "loss": "huber",        # "huber" | "mse" | "mae"
        "huber_delta": 1.0,
        "early_stopping_patience": 8,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 4,
        "seed": 123,
        "mixed_precision": False,   # True requiere GPU compatible
    },

    # --------------------------------------------------------------------------
    # Corrector residual
    # --------------------------------------------------------------------------
    "CORRECTOR": {
        "peak_quantile": 0.90,
        "peak_metric_weight": 0.70,     # 70% MAE picos + 30% MAE global
        "max_horizons_buffer": 365,     # horizontes observados guardados en buffer
        # Hiperparámetros de HistGradientBoostingRegressor
        "hgb_loss": "absolute_error",
        "hgb_learning_rate": 0.05,
        "hgb_max_iter": 150,
        "hgb_max_leaf_nodes": 25,
        "hgb_l2": 0.01,
    },

    # --------------------------------------------------------------------------
    # Opciones de ejecución
    # --------------------------------------------------------------------------
    "RE_TRAIN_BASE": False,         # False = cargar modelo si existe
    "RE_TRAIN_CORRECTOR": False,    # False = cargar corrector si existe
    "DOWNLOAD_SMN": False,
    "SKIP_EXISTING_SMN": True,

    # --------------------------------------------------------------------------
    # Rango de descarga SMN
    # --------------------------------------------------------------------------
    "SMN_START": "2024-07-16",
    "SMN_END": "2026-04-21",
}


# ==============================================================================
# SECCIÓN 2 — IMPORTS Y SEMILLAS
# ==============================================================================

import os
import glob
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import holidays
import joblib
import json
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input

import plotly.graph_objects as go
import plotly.io as pio
import webbrowser

warnings.filterwarnings("ignore")
pio.renderers.default = "browser"


def _setup_environment(cfg: Dict) -> None:
    """Fija semillas y habilita mixed precision si se solicita."""
    seed = cfg["TRAINING"]["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if cfg["TRAINING"]["mixed_precision"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision activada.")


# ==============================================================================
# SECCIÓN 2b — PARÁMETROS DERIVADOS DE LA RESOLUCIÓN
# ==============================================================================

def resolve_temporal_params(cfg: Dict) -> Dict:
    """
    Calcula parámetros derivados de la resolución temporal y los agrega al cfg.

    Retorna un dict con:
        FREQ, STEPS_PER_DAY, HORIZON, WINDOW_SIZE, FORECAST_STRIDE,
        OUTPUT_DIR, MODEL_DIR, BASE_MODEL_PATH, SCALERS_PATH,
        CORRECTOR_PATH, META_INFO_PATH
    """
    resolution = cfg["RESOLUTION"]

    if resolution == "1h":
        freq = "1h"
        steps_per_day = 24
        # Nota: para 1h la ventana por defecto es 4 días
        # El usuario puede cambiar WINDOW_DAYS en CONFIG
    elif resolution == "15min":
        freq = "15min"
        steps_per_day = 96
    else:
        raise ValueError(
            f"RESOLUTION debe ser '1h' o '15min', recibido: '{resolution}'"
        )

    horizon = steps_per_day * cfg["HORIZON_DAYS"]
    window_size = steps_per_day * cfg["WINDOW_DAYS"]
    forecast_stride = horizon  # predicción no solapada (day-ahead)

    tag = resolution.replace("min", "m")
    output_dir = Path(f"plots_prediccion_{tag}")
    model_dir = Path(f"modelos_prediccion_{tag}")

    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    exp_tag = (
        f"{resolution}_"
        f"H{cfg['HORIZON_DAYS']}d_"
        f"W{cfg['WINDOW_DAYS']}d_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    return {
        "FREQ": freq,
        "STEPS_PER_DAY": steps_per_day,
        "HORIZON": horizon,
        "WINDOW_SIZE": window_size,
        "FORECAST_STRIDE": forecast_stride,
        
        "EXP_TAG": exp_tag,
        
        "OUTPUT_DIR": output_dir,
        "MODEL_DIR": model_dir,

        "BASE_MODEL_PATH": model_dir / f"modelo_base_cnn_lstm_{tag}.keras",
        "SCALERS_PATH": model_dir / f"scalers_base_{tag}.joblib",
        "CORRECTOR_PATH": model_dir / f"modelo_corrector_residual_{tag}.joblib",
        "META_INFO_PATH": model_dir / f"meta_info_{tag}.joblib",
        "RESULTS_PATH": model_dir / f"resultados_test_{tag}.joblib",
        }


# ==============================================================================
# SECCIÓN 3 — UTILIDADES GENERALES
# ==============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_and_open_plotly(fig: go.Figure, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, auto_open=False)
    webbrowser.open(output_path.resolve().as_uri())
    print("Figura guardada:", output_path.resolve())


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Modelo",
    peak_quantile: float = 0.90,
) -> Dict:
    """Calcula MAE, RMSE, R², sMAPE, MAE/RMSE en picos y bias en picos."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    smape_val = smape(y_true_flat, y_pred_flat)

    threshold = np.quantile(y_true_flat, peak_quantile)
    peak_mask = y_true_flat >= threshold

    if peak_mask.sum() > 0:
        mae_peak = mean_absolute_error(
            y_true_flat[peak_mask], y_pred_flat[peak_mask]
        )
        rmse_peak = np.sqrt(
            mean_squared_error(y_true_flat[peak_mask], y_pred_flat[peak_mask])
        )
        bias_peak = np.mean(y_pred_flat[peak_mask] - y_true_flat[peak_mask])
    else:
        mae_peak = rmse_peak = bias_peak = np.nan

    q_tag = int(peak_quantile * 100)
    return {
        "modelo": label,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE_%": smape_val,
        f"MAE_picos_q{q_tag}": mae_peak,
        f"RMSE_picos_q{q_tag}": rmse_peak,
        f"Bias_picos_q{q_tag}": bias_peak,
    }


def inverse_y(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    """Desescala arrays de shape (samples, horizon) o (samples, horizon, 1)."""
    original_shape = y_scaled.shape
    flat = y_scaled.reshape(-1, 1)
    inv = y_scaler.inverse_transform(flat)
    if len(original_shape) == 3:
        return inv.reshape(original_shape[0], original_shape[1])
    return inv.reshape(original_shape)


# ==============================================================================
# SECCIÓN 4 — CARGA DE DATOS
# ==============================================================================

def _extract_date_from_filename(filepath: str) -> Tuple[int, int]:
    """Extrae (año, mes) del nombre de archivo 'YYYY_MM.csv'."""
    base = os.path.basename(filepath)
    parts = base.split("_")
    year = int(parts[0])
    month = int(parts[1].replace(".csv", ""))
    return year, month


def load_energy_data(cfg: Dict) -> pd.DataFrame:
    """
    Carga todos los CSV de energía, los concatena y devuelve un DataFrame
    a resolución NATIVA (15 min).

    El resampleo a la resolución objetivo se hace en build_dataset(),
    no aquí, para evitar contaminar los datos crudos.
    """
    print_header("CARGANDO DATOS DE ENERGÍA")

    files = glob.glob(os.path.join(cfg["ENERGY_PATH"], "*.csv"))
    files = sorted(files, key=_extract_date_from_filename)

    if not files:
        raise FileNotFoundError(
            f"No se encontraron CSV en {cfg['ENERGY_PATH']}"
        )

    print(f"Archivos encontrados: {len(files)}")

    dfs = []
    for f in files:
        print("  Leyendo:", os.path.basename(f))
        temp = pd.read_csv(f, sep=";", encoding="latin-1")
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)

    # Normalizar nombres con encoding roto
    df = df.rename(columns={
        "ï»¿Fecha": "Fecha",
        "EnergÃ­a Activa": "Energía Activa",
        "EnergÃ­a Reactiva": "Energía Reactiva",
    })

    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)
    df = df.sort_values("Fecha").set_index("Fecha")

    for col in ["Energía Activa", "Energía Reactiva", cfg["TARGET_RAW"]]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    df = df.select_dtypes(include=["number"])
    print("Shape energía (nativa):", df.shape)
    return df


def load_meteo_data(cfg: Dict) -> pd.DataFrame:
    """
    Carga archivos horarios del SMN, filtra por estación ROSARIO y
    devuelve DataFrame a resolución HORARIA (nativa SMN).

    El resampleo a la resolución objetivo se hace en build_dataset().
    """
    print_header("CARGANDO DATOS METEOROLÓGICOS")

    smn_path = Path(cfg["SMN_PATH"])
    meteo_files = sorted(smn_path.glob("datohorario*.txt"))

    if not meteo_files:
        raise FileNotFoundError(
            f"No se encontraron archivos en {smn_path}"
        )

    print(f"Archivos meteorológicos: {len(meteo_files)}")

    dfs = []
    for f in meteo_files:
        with open(f, encoding="latin-1") as fh:
            lines = fh.readlines()
        lines = lines[2:]   # omitir cabecera de 2 líneas

        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                fecha, hora = parts[0], parts[1]
                nombre = " ".join(parts[7:])
                if "ROSARIO" not in nombre.upper():
                    continue
                dt = pd.to_datetime(fecha + hora.zfill(2), format="%d%m%Y%H")
                rows.append({
                    "datetime": dt,
                    "TEMP": parts[2],
                    "HUM": parts[3],
                    "PNM": parts[4],
                    "DD": parts[5],
                    "FF": parts[6],
                })
            except Exception:
                continue

        if rows:
            dfs.append(pd.DataFrame(rows))

    if not dfs:
        raise ValueError("No se pudieron leer datos meteorológicos válidos.")

    df_met = pd.concat(dfs).set_index("datetime").sort_index()

    for col in ["TEMP", "HUM", "PNM", "DD", "FF"]:
        df_met[col] = pd.to_numeric(df_met[col], errors="coerce")

    df_met = df_met.select_dtypes(include=["number"])
    print("Shape meteo (horaria nativa):", df_met.shape)
    return df_met


def download_smn_data(cfg: Dict) -> None:
    """Descarga archivos diarios del SMN si se solicita."""
    print_header("DESCARGANDO DATOS SMN")

    path = Path(cfg["SMN_PATH"])
    path.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(cfg["SMN_START"])
    end = pd.to_datetime(cfg["SMN_END"])
    base_url = (
        "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
        "?file=observaciones/datohorario"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    current = start

    counts = {"desc": 0, "salt": 0, "sin": 0, "err": 0}

    while current <= end:
        fecha = current.strftime("%Y%m%d")
        filepath = path / f"datohorario{fecha}.txt"

        if (
            cfg["SKIP_EXISTING_SMN"]
            and filepath.exists()
            and filepath.stat().st_size > 100
        ):
            counts["salt"] += 1
            current += timedelta(days=1)
            continue

        try:
            r = requests.get(f"{base_url}{fecha}.txt", headers=headers, timeout=20)
            if r.status_code == 200 and len(r.content) > 100:
                filepath.write_bytes(r.content)
                counts["desc"] += 1
            else:
                counts["sin"] += 1
        except Exception as e:
            print(f"  ERROR {fecha}: {e}")
            counts["err"] += 1

        current += timedelta(days=1)

    print("Resumen SMN:", counts)


# ==============================================================================
# SECCIÓN 5 — PREPROCESAMIENTO
# ==============================================================================

def resample_energy(df_energy: pd.DataFrame, freq: str, resolution: str) -> pd.DataFrame:
    """
    Resamplea los datos de energía a la resolución objetivo.

    - "1h"   : promedia bloques de 4 lecturas de 15 min → 1 valor/hora.
    - "15min": usa asfreq (los datos ya deberían estar en 15 min).

    NOTA LEAKAGE:
        Se usa .mean() y .asfreq() que no miran hacia el futuro.
        No hay look-ahead.
    """
    if resolution == "1h":
        df_resampled = df_energy.resample(freq).mean()
    else:
        # Si los datos ya son 15min, asfreq preserva la grilla exacta.
        # Si hubiese huecos, quedan como NaN para ser tratados después.
        df_resampled = df_energy.asfreq(freq)

    print(f"  Shape energía tras resampleo ({freq}):", df_resampled.shape)
    return df_resampled


def resample_meteo(df_met: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resamplea meteo (horaria nativa) a la resolución objetivo.

    Para "1h": el resampleo es trivial (ya es horaria), solo alinea la grilla.
    Para "15min": interpolación temporal entre horas consecutivas.

    NOTA LEAKAGE:
        La interpolación lineal temporal es causal: rellena el interior
        de un intervalo horario con valores que por definición son pasados
        o contemporáneos al instante observado.
        NO genera leakage porque la meteo es una covariable exógena conocida
        (o pronosticada por NWP), no el target.
    """
    df_resampled = (
        df_met
        .resample(freq)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    print(f"  Shape meteo tras resampleo ({freq}):", df_resampled.shape)
    return df_resampled


def align_and_join(
    df_energy: pd.DataFrame,
    df_met: pd.DataFrame,
    freq: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye un índice maestro con la intersección de ambos rangos,
    alinea los dos DataFrames y los une.

    Retorna (df_joined, df_met_aligned).

    NOTA LEAKAGE:
        Se toma la intersección (start=max, end=min) por lo que no se
        extrapolará fuera del rango común. El ffill/bfill en meteo solo
        cubre huecos puntuales, no proyecta al futuro.
    """
    start = max(df_energy.index.min(), df_met.index.min())
    end = min(df_energy.index.max(), df_met.index.max())

    master_index = pd.date_range(start=start, end=end, freq=freq)

    df_energy = df_energy.reindex(master_index)
    df_met = (
        df_met
        .reindex(master_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    df_joined = df_energy.join(df_met)
    return df_joined, df_met


def clean_target_outliers(
    df: pd.DataFrame,
    target_col: str,
    clean_col: str,
    window: int,
    k: float,
) -> pd.DataFrame:
    """
    Reemplaza outliers del target por el límite de la banda rolling.

    Usa rolling con center=True para detectar anomalías locales.

    NOTA LEAKAGE:
        center=True usa puntos pasados Y futuros dentro de la ventana.
        Esto introduce una fuente menor de look-ahead en la LIMPIEZA
        del target, no en la predicción.

        Si se desea eliminar este look-ahead, cambiar center=False y
        aumentar la ventana (p.ej. window*2). En la práctica, la limpieza
        de outliers con center=True es un preprocesamiento estándar aceptable
        porque:
            a) se aplica a datos históricos antes del split,
            b) los outliers extremos dañarían más que esta suavización mínima.

        RIESGO REAL: bajo. No se usa TARGET_CLEAN como feature directa
        en el horizonte futuro.
    """
    df = df.copy()

    rolling_mean = df[target_col].rolling(window=window, center=True).mean()
    rolling_std = df[target_col].rolling(window=window, center=True).std()

    upper = rolling_mean + k * rolling_std
    lower = rolling_mean - k * rolling_std

    df[clean_col] = df[target_col].clip(lower=lower, upper=upper)

    outliers = (df[target_col] > upper) | (df[target_col] < lower)
    print(f"  Outliers corregidos: {int(outliers.sum())} ({100*outliers.mean():.2f}%)")

    return df


# ==============================================================================
# SECCIÓN 6 — FEATURE ENGINEERING
# ==============================================================================

def add_time_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Agrega features temporales cíclicas y calendáricas.

    Para "15min" se agrega codificación del minuto dentro de la hora.
    Para "1h" no aplica.
    """
    feat_cfg = cfg["FEATURES"]
    df = df.copy()

    if feat_cfg["use_calendar_raw"]:
        df["hora"] = df.index.hour
        df["dia_semana"] = df.index.dayofweek
        df["mes"] = df.index.month
        df["dia_mes"] = df.index.day
        df["dia_anio"] = df.index.dayofyear

        if cfg["RESOLUTION"] == "15min" and feat_cfg["use_minute"]:
            df["minuto"] = df.index.minute

    if feat_cfg["use_cyclic_time"]:
        hora = df.index.hour
        dow = df.index.dayofweek
        mes = df.index.month

        # Para 15min usamos frecuencia doble en la hora (2 ciclos/día)
        # para capturar patrones intra-día con mayor resolución.
        # Para 1h un ciclo es suficiente.
        if cfg["RESOLUTION"] == "15min":
            df["hora_sin"] = np.sin(4 * np.pi * hora / 24)
        else:
            df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
            df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

        if cfg["RESOLUTION"] == "15min" and feat_cfg["use_minute"]:
            df["min_cos"] = np.cos(2 * np.pi * df.index.minute / 60)

        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["mes_sin"] = np.sin(2 * np.pi * mes / 12)

        if cfg["RESOLUTION"] == "1h":
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df["mes_cos"] = np.cos(2 * np.pi * mes / 12)

    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna binaria de feriados argentinos."""
    df = df.copy()
    start_year = df.index.min().year
    end_year = df.index.max().year
    ar_holidays = holidays.Argentina(years=range(start_year, end_year + 1))
    df["feriado"] = df.index.map(
        lambda x: 1 if x.date() in ar_holidays else 0
    )
    return df


# ==============================================================================
# SECCIÓN 7 — CONSTRUCCIÓN DE VENTANAS MULTISTEP
# ==============================================================================

def create_windows_multistep(
    X: pd.DataFrame,
    X_future: pd.DataFrame,
    y: pd.Series,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Genera ventanas deslizantes (X_past, X_future, y) para entrenamiento.

    Para cada muestra i:
        X_past[i]   = features[i : i+window_size]           (pasado)
        X_future[i] = features_future[origin : origin+H]    (futuro conocido)
        y[i]        = target[origin : origin+H]             (a predecir)

    donde origin = i + window_size.

    NOTA LEAKAGE (crítico):
        X_future[i] contiene features del período FUTURO [origin, origin+H).
        Esto es correcto SOLO si esas features son genuinamente conocidas
        antes de la predicción (meteo pronosticada, calendario).
        Si se incluyera el target futuro o cualquier derivada de él en
        X_future → data leakage directo. Por eso TARGET_CLEAN se excluye
        de features_future en build_features_and_windows().

        El índice origin_times[i] = X.index[origin] apunta al PRIMER paso
        predicho, no al último paso observado. Esto es correcto para
        representar el instante de inicio de la predicción.
    """
    X_vals = X.values
    Xf_vals = X_future.values
    y_vals = y.values

    X_wins, Xf_wins, y_wins, origins = [], [], [], []

    n = len(X_vals)
    for i in range(n - window_size - horizon):
        origin = i + window_size
        X_wins.append(X_vals[i: i + window_size])
        Xf_wins.append(Xf_vals[origin: origin + horizon])
        y_wins.append(y_vals[origin: origin + horizon])
        origins.append(X.index[origin])

    return (
        np.array(X_wins),
        np.array(Xf_wins),
        np.array(y_wins),
        pd.DatetimeIndex(origins),
    )


# ==============================================================================
# SECCIÓN 8 — SPLIT TEMPORAL Y ESCALADO
# ==============================================================================

def temporal_split_and_scale(
    X: np.ndarray,
    X_future: np.ndarray,
    y: np.ndarray,
    origin_times: pd.DatetimeIndex,
    cfg: Dict,
) -> Tuple[Dict, Dict]:
    """
    Realiza el split cronológico train / val / test y escala los datos.

    NOTA LEAKAGE (escalado):
        Los scalers se fitean ÚNICAMENTE con datos de entrenamiento (X_train).
        Val y test se transforman con el scaler ya fiteado, como corresponde.
        Usar fit_transform sobre todo el dataset antes del split sería
        data leakage porque el scaler vería la distribución del test.
    """
    print_header("SPLIT TEMPORAL Y ESCALADO")

    split_cfg = cfg["SPLIT"]
    n = len(X)

    split_test = int(n * (1 - split_cfg["test_ratio"]))

    X_train_full = X[:split_test]
    X_test = X[split_test:]

    Xf_train_full = X_future[:split_test]
    Xf_test = X_future[split_test:]

    y_train_full = y[:split_test]
    y_test = y[split_test:]

    origin_train_full = origin_times[:split_test]
    origin_test = origin_times[split_test:]

    val_n = int(len(X_train_full) * split_cfg["val_ratio_from_train"])

    X_train = X_train_full[:-val_n]
    X_val = X_train_full[-val_n:]

    Xf_train = Xf_train_full[:-val_n]
    Xf_val = Xf_train_full[-val_n:]

    y_train = y_train_full[:-val_n]
    y_val = y_train_full[-val_n:]

    origin_train = origin_train_full[:-val_n]
    origin_val = origin_train_full[-val_n:]

    # -------------------------------------------------------------------------
    # Fit SOLO en train
    # -------------------------------------------------------------------------
    X_scaler = MinMaxScaler()
    Xf_scaler = MinMaxScaler()
    y_scaler = StandardScaler()

    def _scale_3d(scaler, arr_train, *arrs_other, fit=True):
        """Fitea el scaler en arr_train y transforma todos los arrs."""
        n_feat = arr_train.shape[-1]
        if fit:
            scaled_train = scaler.fit_transform(
                arr_train.reshape(-1, n_feat)
            ).reshape(arr_train.shape).astype(np.float32)
        else:
            scaled_train = scaler.transform(
                arr_train.reshape(-1, n_feat)
            ).reshape(arr_train.shape).astype(np.float32)

        scaled_others = [
            scaler.transform(a.reshape(-1, n_feat)).reshape(a.shape).astype(np.float32)
            for a in arrs_other
        ]
        return (scaled_train, *scaled_others)

    X_tr_s, X_v_s, X_te_s = _scale_3d(X_scaler, X_train, X_val, X_test)
    Xf_tr_s, Xf_v_s, Xf_te_s = _scale_3d(Xf_scaler, Xf_train, Xf_val, Xf_test)

    y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape).astype(np.float32)
    y_v_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape).astype(np.float32)
    y_te_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape).astype(np.float32)

    data = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "Xf_train": Xf_train, "Xf_val": Xf_val, "Xf_test": Xf_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_scaled": X_tr_s, "X_val_scaled": X_v_s, "X_test_scaled": X_te_s,
        "Xf_train_scaled": Xf_tr_s, "Xf_val_scaled": Xf_v_s, "Xf_test_scaled": Xf_te_s,
        "y_train_scaled": y_tr_s, "y_val_scaled": y_v_s, "y_test_scaled": y_te_s,
        "origin_train": origin_train, "origin_val": origin_val, "origin_test": origin_test,
    }

    scalers = {
        "X_scaler": X_scaler,
        "Xf_scaler": Xf_scaler,
        "y_scaler": y_scaler,
    }

    print(f"  Train  : {X_tr_s.shape} | {Xf_tr_s.shape} | {y_tr_s.shape}")
    print(f"  Val    : {X_v_s.shape} | {Xf_v_s.shape} | {y_v_s.shape}")
    print(f"  Test   : {X_te_s.shape} | {Xf_te_s.shape} | {y_te_s.shape}")

    return data, scalers


# ==============================================================================
# SECCIÓN 9 — MODELO BASE CNN-LSTM
# ==============================================================================

def build_cnn_lstm_model(
    n_past_steps: int,
    n_past_features: int,
    n_future_steps: int,
    n_future_features: int,
    cfg: Dict,
) -> keras.Model:
    """
    Construye el modelo CNN-LSTM seq2seq configurable.

    Arquitectura:
        Encoder:
            - Varias Conv1D en paralelo con diferentes dilation_rates
              (capturan dependencias a distintas escalas temporales)
            - Concatenación de ramas CNN
            - LSTM que resume el pasado en un vector de contexto

        Decoder:
            - LSTM sobre features futuras conocidas
            - RepeatVector del contexto + Concatenate con LSTM futuro
            - LSTM decoder
            - TimeDistributed Dense → 1 salida por paso de horizonte
    """
    m_cfg = cfg["MODEL"]

    past_input = Input(shape=(n_past_steps, n_past_features), name="past_input")

    resolution = cfg["RESOLUTION"]
    # Ramas CNN con distintas dilation rates
    cnn_branches = []
    if resolution == "1h":
        for ks in m_cfg["cnn_kernel_1h"]:
            branch = layers.Conv1D(
                filters=m_cfg["cnn_filters"],
                kernel_size=ks,
                padding="same",
                activation="relu",
                dilation_rate=m_cfg["cnn_dilations"],
                name=f"conv1d_dr{ks}",
                )(past_input)
            cnn_branches.append(branch)
    else:
        for ks in m_cfg["cnn_kernel_15min"]:
            branch = layers.Conv1D(
                filters=m_cfg["cnn_filters"],
                kernel_size=ks,
                padding="same",
                activation="relu",
                dilation_rate=m_cfg["cnn_dilations"],
                name=f"conv1d_dr{ks}",
                )(past_input)
            cnn_branches.append(branch)

    cnn_out = layers.Concatenate(name="cnn_concat")(cnn_branches) if len(cnn_branches) > 1 else cnn_branches[0]

    enc_context = layers.LSTM(
        units=m_cfg["lstm_enc_units"],
        return_sequences=False,
        dropout=m_cfg["lstm_enc_dropout"],
        recurrent_dropout=m_cfg["lstm_enc_recurrent_dropout"],
        name="lstm_encoder",
    )(cnn_out)

    future_input = Input(shape=(n_future_steps, n_future_features), name="future_input")

    fut_seq = layers.LSTM(
        units=m_cfg["lstm_fut_units"],
        return_sequences=True,
        name="lstm_future",
    )(future_input)

    # Expandir contexto para concatenar con secuencia futura
    context_repeated = layers.RepeatVector(n_future_steps, name="repeat_context")(enc_context)
    dec_input = layers.Concatenate(name="decoder_concat")([context_repeated, fut_seq])

    dec_seq = layers.LSTM(
        units=m_cfg["lstm_dec_units"],
        return_sequences=True,
        name="lstm_decoder",
    )(dec_input)

    dec_dense = layers.TimeDistributed(
        layers.Dense(m_cfg["dense_dec_units"], activation=m_cfg["dense_dec_activation"]),
        name="td_dense",
    )(dec_seq)

    output = layers.TimeDistributed(layers.Dense(1), name="td_output")(dec_dense)

    model = Model(inputs=[past_input, future_input], outputs=output)

    # Selección de loss
    loss_name = cfg["TRAINING"]["loss"]
    if loss_name == "huber":
        loss_fn = keras.losses.Huber(delta=cfg["TRAINING"]["huber_delta"])
    elif loss_name == "mse":
        loss_fn = "mse"
    else:
        loss_fn = "mae"

    # Selección de optimizador
    lr = cfg["TRAINING"]["learning_rate"]
    opt_name = cfg["TRAINING"]["optimizer"]
    if opt_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])
    return model


def train_or_load_base_model(
    data: Dict,
    scalers: Dict,
    cfg: Dict,
    tparams: Dict,
) -> Tuple[keras.Model, Dict]:
    """
    Entrena el modelo base CNN-LSTM o lo carga desde disco.

    tparams: dict con rutas y parámetros temporales derivados de la resolución.
    """
    print_header("MODELO BASE CNN-LSTM")

    base_path = tparams["BASE_MODEL_PATH"]
    scalers_path = tparams["SCALERS_PATH"]

    if (not cfg["RE_TRAIN_BASE"]) and base_path.exists() and scalers_path.exists():
        print("  Cargando modelo existente...")
        model = keras.models.load_model(base_path)
        scalers_loaded = joblib.load(scalers_path)
        return model, scalers_loaded

    model = build_cnn_lstm_model(
        n_past_steps=data["X_train_scaled"].shape[1],
        n_past_features=data["X_train_scaled"].shape[2],
        n_future_steps=data["Xf_train_scaled"].shape[1],
        n_future_features=data["Xf_train_scaled"].shape[2],
        cfg=cfg,
    )

    model.summary()

    train_cfg = cfg["TRAINING"]
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
        ),
    ]

    model.fit(
        [data["X_train_scaled"], data["Xf_train_scaled"]],
        data["y_train_scaled"],
        validation_data=(
            [data["X_val_scaled"], data["Xf_val_scaled"]],
            data["y_val_scaled"],
        ),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    model.save(base_path)
    joblib.dump(scalers, scalers_path)
    print(f"  Modelo guardado en: {base_path}")

    return model, scalers


def predict_base_real(
    model: keras.Model,
    X_scaled: np.ndarray,
    Xf_scaled: np.ndarray,
    y_scaler: StandardScaler,
) -> np.ndarray:
    """Predicción del modelo base desescalada a unidades reales."""
    y_pred_scaled = model.predict([X_scaled, Xf_scaled], verbose=0)
    return inverse_y(y_pred_scaled, y_scaler)


# ==============================================================================
# SECCIÓN 10 — CORRECTOR RESIDUAL DAY-AHEAD
# ==============================================================================

class HorizonResidualErrorBuffer:
    """
    Buffer de errores observados por horizonte completo (day-ahead).

    Garantiza que el corrector NO vea:
        - errores del horizonte que está corrigiendo,
        - errores de horizontes futuros.

    Solo ve errores de horizontes ANTERIORES completos.

    Orden de operación para cada horizonte i:
        1) buffer.features() → genera features para corregir horizonte i
        2) corregir horizonte i
        3) buffer.update(error_real_i) → registra el error DESPUÉS de predecir
    """

    def __init__(
        self,
        horizon: int,
        max_horizons: int = 365,
        initial_error_vectors: Optional[np.ndarray] = None,
    ):
        self.horizon = int(horizon)
        self.max_horizons = int(max_horizons)
        self.error_vectors: List[np.ndarray] = []

        if initial_error_vectors is not None:
            for ev in initial_error_vectors:
                self.update(ev)

    def update(self, error_vector: np.ndarray) -> None:
        """Registra un vector de errores de longitud self.horizon."""
        if error_vector is None:
            return
        err = np.asarray(error_vector, dtype=float).reshape(-1)
        if err.size != self.horizon:
            raise ValueError(
                f"Vector de error debe tener longitud {self.horizon}, recibido {err.size}."
            )
        if not np.all(np.isfinite(err)):
            return
        self.error_vectors.append(err)
        if len(self.error_vectors) > self.max_horizons:
            self.error_vectors = self.error_vectors[-self.max_horizons:]

    def _last_curve(self, k: int) -> np.ndarray:
        if len(self.error_vectors) >= k:
            return self.error_vectors[-k]
        return np.full(self.horizon, np.nan)

    def _mean_curve(self, k: int) -> np.ndarray:
        if len(self.error_vectors) >= k:
            return np.mean(np.vstack(self.error_vectors[-k:]), axis=0)
        return np.full(self.horizon, np.nan)

    def _std_curve(self, k: int) -> np.ndarray:
        if len(self.error_vectors) >= k:
            arr = np.vstack(self.error_vectors[-k:])
            return np.std(arr, axis=0, ddof=1) if k > 1 else np.zeros(self.horizon)
        return np.full(self.horizon, np.nan)

    def _global_stats(self, k: int) -> np.ndarray:
        if len(self.error_vectors) < k:
            return np.full(10, np.nan)
        arr = np.vstack(self.error_vectors[-k:]).reshape(-1)
        peaks = arr[arr >= np.quantile(arr, 0.90)] if arr.size > 0 else np.array([np.nan])
        return np.array([
            np.mean(arr),
            np.std(arr, ddof=1) if arr.size > 1 else 0.0,
            np.min(arr),
            np.max(arr),
            np.mean(np.abs(arr)),
            np.sqrt(np.mean(arr ** 2)),
            np.quantile(arr, 0.10),
            np.quantile(arr, 0.50),
            np.quantile(arr, 0.90),
            np.mean(peaks) if peaks.size > 0 else np.nan,
        ])

    def features(self) -> np.ndarray:
        """
        Devuelve un vector de features de longitud fija.

        Composición (con HORIZON=H):
            - curva error H anterior   : H valores
            - curva error H-2 anterior : H valores
            - curva media 3 horizontes : H valores
            - curva media 7 horizontes : H valores
            - curva std 7 horizontes   : H valores
            - stats globales 1, 3, 7   : 30 valores
        Total: 5*H + 30
        """
        return np.concatenate([
            self._last_curve(1),
            self._last_curve(2),
            self._mean_curve(3),
            self._mean_curve(7),
            self._std_curve(7),
            self._global_stats(1),
            self._global_stats(3),
            self._global_stats(7),
        ]).astype(float)


def _compute_observed_horizon_errors(
    y_true: np.ndarray,
    y_base_pred: np.ndarray,
) -> np.ndarray:
    """Calcula la matrix de errores observados (samples, horizon)."""
    return np.asarray(y_true) - np.asarray(y_base_pred)


def _build_day_ahead_residual_history(
    y_true: np.ndarray,
    y_base_pred: np.ndarray,
    horizon: int,
    max_horizons: int,
    initial_error_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construye features históricas de error en modo day-ahead realista.

    Para cada fila i:
        hist_features[i] = buffer.features()  ← solo horizontes < i
        buffer.update(error_i)                ← actualiza DESPUÉS de crear features

    Así hist_features[i] NO contiene información del horizonte i ni futura.
    """
    buffer = HorizonResidualErrorBuffer(
        horizon=horizon,
        max_horizons=max_horizons,
        initial_error_vectors=initial_error_vectors,
    )
    hist_features = []
    error_vectors = _compute_observed_horizon_errors(y_true, y_base_pred)

    for ev in error_vectors:
        hist_features.append(buffer.features())
        buffer.update(ev)

    return np.vstack(hist_features)


def build_corrector_features(
    X_past: np.ndarray,
    X_future: np.ndarray,
    y_base_pred: np.ndarray,
    origin_times: pd.DatetimeIndex,
    residual_history_features: Optional[np.ndarray],
    horizon: int,
    max_horizons: int,
) -> np.ndarray:
    """
    Construye la matriz de features 2D para el corrector residual.

    NOTA LEAKAGE:
        Esta función NO recibe y_true. Eso previene filtración accidental
        del target futuro. Los únicos inputs son:
            - predicción base (y_base_pred): info ya conocida
            - features futuras exógenas (X_future): meteo/calendario
            - resumen de la ventana pasada (X_past): info conocida
            - historial de errores pasados (residual_history_features): causal
    """
    n = X_past.shape[0]

    base_flat = y_base_pred  # (n, horizon)

    base_stats = np.column_stack([
        np.mean(y_base_pred, axis=1),
        np.std(y_base_pred, axis=1),
        np.max(y_base_pred, axis=1),
        np.min(y_base_pred, axis=1),
        y_base_pred[:, -1] - y_base_pred[:, 0],
        np.quantile(y_base_pred, 0.90, axis=1),
        np.quantile(y_base_pred, 0.95, axis=1),
    ])

    future_flat = X_future.reshape(n, -1)

    past_last = X_past[:, -1, :]
    past_mean = np.mean(X_past, axis=1)
    past_std = np.std(X_past, axis=1)
    past_max = np.max(X_past, axis=1)
    past_min = np.min(X_past, axis=1)

    origin_df = pd.DataFrame(index=origin_times)
    origin_df["hour"] = origin_df.index.hour
    origin_df["dow"] = origin_df.index.dayofweek
    origin_df["month"] = origin_df.index.month
    origin_df["hour_sin"] = np.sin(2 * np.pi * origin_df["hour"] / 24)
    origin_df["dow_cos"] = np.cos(2 * np.pi * origin_df["dow"] / 7)
    origin_df["month_sin"] = np.sin(2 * np.pi * origin_df["month"] / 12)
    origin_features = origin_df.values

    if residual_history_features is None:
        empty_dim = HorizonResidualErrorBuffer(horizon=horizon).features().shape[0]
        residual_history_features = np.full((n, empty_dim), np.nan)

    return np.concatenate([
        base_flat,
        base_stats,
        future_flat,
        past_last,
        past_mean,
        past_std,
        past_max,
        past_min,
        origin_features,
        residual_history_features,
    ], axis=1)


def _build_corrector_feature_set_day_ahead(
    X_past: np.ndarray,
    X_future: np.ndarray,
    y_true: np.ndarray,
    y_base_pred: np.ndarray,
    origin_times: pd.DatetimeIndex,
    horizon: int,
    max_horizons: int,
    initial_error_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrapper que ensambla historial causal + features del corrector."""
    hist_err = _build_day_ahead_residual_history(
        y_true=y_true,
        y_base_pred=y_base_pred,
        horizon=horizon,
        max_horizons=max_horizons,
        initial_error_vectors=initial_error_vectors,
    )
    return build_corrector_features(
        X_past=X_past,
        X_future=X_future,
        y_base_pred=y_base_pred,
        origin_times=origin_times,
        residual_history_features=hist_err,
        horizon=horizon,
        max_horizons=max_horizons,
    )


def take_every_forecast_stride(
    data: Dict,
    base_preds: Dict,
    stride: int,
) -> Tuple[Dict, Dict]:
    """
    Reduce train/val/test a ventanas no solapadas (una cada HORIZON pasos).

    El modelo base puede entrenarse con todas las ventanas deslizantes,
    pero el corrector day-ahead debe evaluarse una vez por período de predicción.

    Con FREQ=15min y HORIZON=96: stride=96 → 1 predicción/día.
    Con FREQ=1h y HORIZON=24: stride=24 → 1 predicción/día.
    """
    def _subsample(arr, idx):
        if arr is None:
            return None
        if isinstance(arr, pd.DatetimeIndex):
            return arr[idx]
        return arr[idx]

    idx_tr = np.arange(0, len(data["X_train"]), stride)
    idx_va = np.arange(0, len(data["X_val"]), stride)
    idx_te = np.arange(0, len(data["X_test"]), stride)

    data_da = {}
    for key, val in data.items():
        if "train" in key:
            data_da[key] = _subsample(val, idx_tr)
        elif "val" in key:
            data_da[key] = _subsample(val, idx_va)
        elif "test" in key:
            data_da[key] = _subsample(val, idx_te)
        else:
            data_da[key] = val

    base_preds_da = {
        "train": base_preds["train"][idx_tr],
        "val": base_preds["val"][idx_va],
        "test": base_preds["test"][idx_te],
    }

    print_header("SUBMUESTREO DAY-AHEAD PARA EL CORRECTOR")
    print(f"  stride={stride}")
    print(f"  Train corrector : {data_da['X_train'].shape}")
    print(f"  Val corrector   : {data_da['X_val'].shape}")
    print(f"  Test corrector  : {data_da['X_test'].shape}")

    return data_da, base_preds_da


def _get_corrector_model(cfg: Dict, seed: int) -> MultiOutputRegressor:
    """Construye el corrector HistGradientBoosting configurable."""
    c_cfg = cfg["CORRECTOR"]
    return MultiOutputRegressor(
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                loss=c_cfg["hgb_loss"],
                learning_rate=c_cfg["hgb_learning_rate"],
                max_iter=c_cfg["hgb_max_iter"],
                max_leaf_nodes=c_cfg["hgb_max_leaf_nodes"],
                l2_regularization=c_cfg["hgb_l2"],
                random_state=seed,
                verbose=0,
            )),
        ])
    )


def select_and_train_corrector(
    data: Dict,
    base_preds: Dict,
    cfg: Dict,
    tparams: Dict,
) -> Tuple[MultiOutputRegressor, Dict]:
    """
    Entrena (o carga) el corrector residual day-ahead.

    Flujo:
        1) Construir Z_train y Z_val con historial causal.
        2) Entrenar corrector en Z_train.
        3) Evaluar en Z_val y seleccionar según score ponderado.
        4) Re-entrenar el mejor con train+val para usar más datos antes del test.
        5) Guardar modelo y meta_info.
    """
    print_header("CORRECTOR RESIDUAL DAY-AHEAD")

    corrector_path = tparams["CORRECTOR_PATH"]
    meta_path = tparams["META_INFO_PATH"]
    horizon = tparams["HORIZON"]
    max_h = cfg["CORRECTOR"]["max_horizons_buffer"]
    seed = cfg["TRAINING"]["seed"]

    if (not cfg["RE_TRAIN_CORRECTOR"]) and corrector_path.exists() and meta_path.exists():
        print("  Cargando corrector residual...")
        return joblib.load(corrector_path), joblib.load(meta_path)

    y_train, y_val = data["y_train"], data["y_val"]
    pred_train, pred_val = base_preds["train"], base_preds["val"]
    residual_train = y_train - pred_train

    # Features de entrenamiento (sin warmup: el buffer comienza vacío)
    Z_train = _build_corrector_feature_set_day_ahead(
        X_past=data["X_train"],
        X_future=data["Xf_train"],
        y_true=y_train,
        y_base_pred=pred_train,
        origin_times=data["origin_train"],
        horizon=horizon,
        max_horizons=max_h,
        initial_error_vectors=None,
    )

    # Features de validación (warmup con errores de entrenamiento)
    train_errors = _compute_observed_horizon_errors(y_train, pred_train)
    Z_val = _build_corrector_feature_set_day_ahead(
        X_past=data["X_val"],
        X_future=data["Xf_val"],
        y_true=y_val,
        y_base_pred=pred_val,
        origin_times=data["origin_val"],
        horizon=horizon,
        max_horizons=max_h,
        initial_error_vectors=train_errors,
    )

    corrector = _get_corrector_model(cfg, seed)
    print("  Entrenando corrector...")
    corrector.fit(Z_train, residual_train)

    residual_val_pred = corrector.predict(Z_val)
    y_val_corrected = pred_val + residual_val_pred

    peak_q = cfg["CORRECTOR"]["peak_quantile"]
    weight = cfg["CORRECTOR"]["peak_metric_weight"]
    q_tag = int(peak_q * 100)

    metrics_val = compute_metrics(y_val, y_val_corrected, "corrector", peak_q)
    mae_g = metrics_val["MAE"]
    mae_p = metrics_val[f"MAE_picos_q{q_tag}"]
    score = weight * mae_p + (1 - weight) * mae_g
    print(f"  Val → MAE={mae_g:.4f} | MAE_picos={mae_p:.4f} | score={score:.4f}")

    # Re-entrenar con train+val
    print("  Re-entrenando corrector en train+val...")
    X_tv = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    Xf_tv = np.concatenate([data["Xf_train"], data["Xf_val"]], axis=0)
    y_tv = np.vstack([y_train, y_val])
    pred_tv = np.vstack([pred_train, pred_val])
    origin_tv = data["origin_train"].append(data["origin_val"])
    residual_tv = y_tv - pred_tv

    Z_tv = _build_corrector_feature_set_day_ahead(
        X_past=X_tv,
        X_future=Xf_tv,
        y_true=y_tv,
        y_base_pred=pred_tv,
        origin_times=origin_tv,
        horizon=horizon,
        max_horizons=max_h,
        initial_error_vectors=None,
    )

    best_corrector = _get_corrector_model(cfg, seed)
    best_corrector.fit(Z_tv, residual_tv)

    meta_info = {
        "peak_quantile": peak_q,
        "peak_metric_weight": weight,
        "score_val": score,
        "mae_global_val": mae_g,
        "mae_peak_val": mae_p,
        "horizon": horizon,
        "forecast_stride": tparams["FORECAST_STRIDE"],
        "resolution": cfg["RESOLUTION"],
    }

    joblib.dump(best_corrector, corrector_path)
    joblib.dump(meta_info, meta_path)
    print(f"  Corrector guardado: {corrector_path}")

    return best_corrector, meta_info


def apply_corrector(
    corrector: MultiOutputRegressor,
    data: Dict,
    base_preds: Dict,
    cfg: Dict,
    tparams: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica el corrector en test con simulación day-ahead realista.

    Para cada horizonte de test i:
        1) buffer tiene errores completos de horizontes < i
        2) genera Z_i SIN usar y_test[i]
        3) predice residual_i
        4) corrige: y_corr_i = y_base_i + residual_i
        5) actualiza buffer con error observado DESPUÉS de predecir

    NOTA LEAKAGE:
        El buffer NO se actualiza con el error del horizonte actual hasta
        después de generar la predicción corregida de ese horizonte.
        Así se evita toda filtración de información futura.
    """
    horizon = tparams["HORIZON"]
    max_h = cfg["CORRECTOR"]["max_horizons_buffer"]

    # Warmup con errores de train+val
    y_trainval = np.vstack([data["y_train"], data["y_val"]])
    pred_trainval = np.vstack([base_preds["train"], base_preds["val"]])
    initial_errors = _compute_observed_horizon_errors(y_trainval, pred_trainval)

    buffer = HorizonResidualErrorBuffer(
        horizon=horizon,
        max_horizons=max_h,
        initial_error_vectors=initial_errors,
    )

    residual_preds, y_corrected_list = [], []

    for i in range(len(data["X_test"])):
        hist_i = buffer.features().reshape(1, -1)

        Z_i = build_corrector_features(
            X_past=data["X_test"][i: i + 1],
            X_future=data["Xf_test"][i: i + 1],
            y_base_pred=base_preds["test"][i: i + 1],
            origin_times=data["origin_test"][i: i + 1],
            residual_history_features=hist_i,
            horizon=horizon,
            max_horizons=max_h,
        )

        residual_i = corrector.predict(Z_i)[0]
        y_corr_i = base_preds["test"][i] + residual_i

        residual_preds.append(residual_i)
        y_corrected_list.append(y_corr_i)

        # Actualizar buffer con el error REAL del horizonte ya predicho
        observed_error = data["y_test"][i] - base_preds["test"][i]
        buffer.update(observed_error)

    return np.vstack(y_corrected_list), np.vstack(residual_preds)


# ==============================================================================
# SECCIÓN 11 — PIPELINE COMPLETO DE DATASET
# ==============================================================================

def build_dataset(cfg: Dict, tparams: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orquesta la carga, resampleo, alineación, feature engineering y limpieza.

    Retorna (df_final, df_met_aligned).

    df_final   : energía + meteo + features temporales + target limpio.
    df_met_aligned : meteo + features temporales (usada como input futuro).
    """
    print_header("PIPELINE DATASET")

    if cfg["DOWNLOAD_SMN"]:
        download_smn_data(cfg)

    df_energy_raw = load_energy_data(cfg)
    df_met_raw = load_meteo_data(cfg)

    freq = tparams["FREQ"]
    resolution = cfg["RESOLUTION"]

    # -------------------------------------------------------------------------
    # Resampleo a resolución objetivo
    # -------------------------------------------------------------------------
    print("\n[Resampleo]")
    df_energy = resample_energy(df_energy_raw, freq, resolution)
    df_met = resample_meteo(df_met_raw, freq)

    # -------------------------------------------------------------------------
    # Alineación temporal y join
    # -------------------------------------------------------------------------
    print("\n[Alineación]")
    df_final, df_met_aligned = align_and_join(df_energy, df_met, freq)

    # -------------------------------------------------------------------------
    # Feature engineering en el dataset principal
    # -------------------------------------------------------------------------
    feat_cfg = cfg["FEATURES"]

    if feat_cfg["use_cyclic_time"] or feat_cfg["use_calendar_raw"]:
        df_final = add_time_features(df_final, cfg)

    if feat_cfg["use_holidays"]:
        df_final = add_holidays(df_final)

    # Dropna antes de limpieza de outliers
    df_final = df_final.dropna()

    # -------------------------------------------------------------------------
    # Limpieza de outliers del target
    # -------------------------------------------------------------------------
    out_cfg = cfg["OUTLIER"]
    df_final = clean_target_outliers(
        df_final,
        target_col=cfg["TARGET_RAW"],
        clean_col=cfg["TARGET_CLEAN"],
        window=out_cfg["window"],
        k=out_cfg["k"],
    )
    df_final = df_final.dropna()

    # -------------------------------------------------------------------------
    # Feature engineering en meteo (usada como input futuro)
    # -------------------------------------------------------------------------
    if feat_cfg["use_cyclic_time"] or feat_cfg["use_calendar_raw"]:
        df_met_aligned = add_time_features(df_met_aligned, cfg)

    if feat_cfg["use_holidays"]:
        df_met_aligned = add_holidays(df_met_aligned)

    # Re-alinear meteo al índice final (puede haber reducido por dropna)
    df_met_aligned = (
        df_met_aligned
        .reindex(df_final.index)
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    print(f"\n[Dataset final] shape={df_final.shape}")
    print(f"[Meteo alineada] shape={df_met_aligned.shape}")
    print(f"[Rango] {df_final.index.min()} → {df_final.index.max()}")

    return df_final, df_met_aligned


def build_features_and_windows(
    df_final: pd.DataFrame,
    df_met: pd.DataFrame,
    cfg: Dict,
    tparams: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Separa features pasadas / futuras / target y construye ventanas.

    Features pasadas (X): todo df_final excepto TARGET_CLEAN.
    Features futuras (X_future): df_met sin columnas de alta incertidumbre.
    Target (y): TARGET_CLEAN.

    NOTA LEAKAGE:
        TARGET_CLEAN se excluye explícitamente de X_future. Si se dejara,
        el decoder vería el target futuro real durante entrenamiento
        (teacher forcing accidental) y en inferencia habría que imputarlo
        de alguna manera, introduciendo inconsistencia train/test.

        Potencia Activa (raw) sí puede estar en X_pasado porque dentro
        de la ventana histórica es información ya observada.
    """
    print_header("CREANDO FEATURES Y VENTANAS")

    common_idx = df_final.index.intersection(df_met.index)
    df_final = df_final.loc[common_idx]
    df_met = df_met.loc[common_idx]

    assert df_final.index.equals(df_met.index), \
        "Índices de df_final y df_met no coinciden tras intersección."

    features = df_final.drop(columns=[cfg["TARGET_CLEAN"]], errors="ignore")

    features_future = df_met.drop(
        columns=cfg["FEATURES"]["meteo_future_drop"],
        errors="ignore",
    )

    target = df_final[cfg["TARGET_CLEAN"]]

    print(f"  Features pasadas  : {features.shape[1]} columnas")
    print(f"  Features futuras  : {features_future.shape[1]} columnas")
    print(f"  Window size       : {tparams['WINDOW_SIZE']} pasos")
    print(f"  Horizon           : {tparams['HORIZON']} pasos")

    X, X_future, y, origin_times = create_windows_multistep(
        X=features,
        X_future=features_future,
        y=target,
        window_size=tparams["WINDOW_SIZE"],
        horizon=tparams["HORIZON"],
    )

    print(f"  X         : {X.shape}")
    print(f"  X_future  : {X_future.shape}")
    print(f"  y         : {y.shape}")
    print(f"  Orígenes  : {origin_times.min()} → {origin_times.max()}")

    return X, X_future, y, origin_times


# ==============================================================================
# SECCIÓN 12 — VISUALIZACIÓN
# ==============================================================================

def plot_first_step(
    origin_times: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    output_dir: Path,
    filename: str = "predicciones_t1.html",
) -> None:
    """Grafica el primer paso del horizonte (t+1) para todos los samples de test."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=origin_times, y=y_real[:, 0], mode="lines", name="Real", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=origin_times, y=y_base[:, 0], mode="lines", name="CNN-LSTM", opacity=0.75))
    fig.add_trace(go.Scatter(x=origin_times, y=y_corrected[:, 0], mode="lines", name="CNN-LSTM + Corrector", opacity=0.85))
    fig.update_layout(title="Predicción t+1", template="plotly_white", hovermode="x unified")
    save_and_open_plotly(fig, output_dir / filename)


def plot_selected_range(
    origin_times: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    fecha_inicio: str,
    fecha_fin: str,
    output_dir: Path,
    filename: str = "rango_predicciones.html",
) -> None:
    """Grafica t+1 en un rango de fechas específico."""
    mask = (origin_times >= pd.to_datetime(fecha_inicio)) & (origin_times <= pd.to_datetime(fecha_fin))
    if mask.sum() == 0:
        print(f"  Sin datos en [{fecha_inicio}, {fecha_fin}]. Rango disponible: {origin_times.min()} → {origin_times.max()}")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=origin_times[mask], y=y_real[mask, 0], mode="lines", name="Real", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=origin_times[mask], y=y_base[mask, 0], mode="lines", name="CNN-LSTM", opacity=0.75))
    fig.add_trace(go.Scatter(x=origin_times[mask], y=y_corrected[mask, 0], mode="lines", name="CNN-LSTM + Corrector", opacity=0.85))
    fig.update_layout(title=f"Predicción t+1: {fecha_inicio} → {fecha_fin}", template="plotly_white", hovermode="x unified")
    save_and_open_plotly(fig, output_dir / filename)


def plot_horizon_example(
    origin_times: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    freq: str,
    horizon: int,
    output_dir: Path,
    sample_idx: int = -1,
    filename: str = "horizonte_completo.html",
) -> None:
    """Grafica el horizonte completo de un sample específico."""
    if sample_idx < 0:
        sample_idx = len(origin_times) + sample_idx
    t0 = origin_times[sample_idx]
    h_idx = pd.date_range(start=t0, periods=horizon, freq=freq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_idx, y=y_real[sample_idx], mode="lines+markers", name="Real"))
    fig.add_trace(go.Scatter(x=h_idx, y=y_base[sample_idx], mode="lines+markers", name="CNN-LSTM", opacity=0.75))
    fig.add_trace(go.Scatter(x=h_idx, y=y_corrected[sample_idx], mode="lines+markers", name="CNN-LSTM + Corrector", opacity=0.85))
    fig.update_layout(title=f"Horizonte completo desde {t0}", template="plotly_white", hovermode="x unified")
    save_and_open_plotly(fig, output_dir / filename)


def plot_all_day_ahead(
    origin_times: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    freq: str,
    horizon: int,
    output_dir: Path,
    filename: str = "todas_predicciones_day_ahead.html",
) -> None:
    """Concatena todas las predicciones day-ahead y las grafica en serie."""
    xs, real_v, base_v, corr_v = [], [], [], []
    for i, t0 in enumerate(origin_times):
        h_idx = pd.date_range(start=t0, periods=horizon, freq=freq)
        xs.extend(h_idx)
        real_v.extend(y_real[i])
        base_v.extend(y_base[i])
        corr_v.extend(y_corrected[i])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=real_v, mode="lines", name="Real", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=xs, y=base_v, mode="lines", name="CNN-LSTM", opacity=0.75))
    fig.add_trace(go.Scatter(x=xs, y=corr_v, mode="lines", name="CNN-LSTM + Corrector", opacity=0.85))
    fig.update_layout(title="Predicciones day-ahead concatenadas", template="plotly_white", hovermode="x unified")
    save_and_open_plotly(fig, output_dir / filename)


def plot_peak_scatter(
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    peak_quantile: float,
    output_dir: Path,
    filename: str = "scatter_picos.html",
) -> None:
    """Scatter real vs predicho enfocado en valores de pico."""
    rf = y_real.reshape(-1)
    bf = y_base.reshape(-1)
    cf = y_corrected.reshape(-1)
    threshold = np.quantile(rf, peak_quantile)
    mask = rf >= threshold
    minv = min(rf[mask].min(), bf[mask].min(), cf[mask].min())
    maxv = max(rf[mask].max(), bf[mask].max(), cf[mask].max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rf[mask], y=bf[mask], mode="markers", name="CNN-LSTM", opacity=0.65))
    fig.add_trace(go.Scatter(x=rf[mask], y=cf[mask], mode="markers", name="CNN-LSTM + Corrector", opacity=0.65))
    fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="Ideal"))
    fig.update_layout(title=f"Scatter picos q{int(peak_quantile*100)}", template="plotly_white")
    save_and_open_plotly(fig, output_dir / filename)


# def plot_metrics_bar(
#     metrics_df: pd.DataFrame,
#     output_dir: Path,
#     filename: str = "metricas_comparacion.html",
# ) -> None:
#     """Gráfico de barras comparando métricas entre modelos."""
#     metric_cols = [c for c in metrics_df.columns if c != "modelo"]
#     fig = go.Figure()
#     for col in metric_cols:
#         fig.add_trace(go.Bar(x=metrics_df["modelo"], y=metrics_df[col], name=col))
#     fig.update_layout(barmode="group", template="plotly_white", title="Comparación de métricas")
#     save_and_open_plotly(fig, output_dir / filename)
    
def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "metricas_comparacion.html",
) -> None:
    
    df_plot = metrics_df.copy()
    if "R2" in df_plot.columns:
        df_plot["R2 (x100)"] = df_plot["R2"] * 100
        df_plot = df_plot.drop(columns=["R2"])

    metric_cols = [c for c in df_plot.columns if c != "modelo"]
    fig = go.Figure()
    for col in metric_cols:
        fig.add_trace(go.Bar(x=df_plot["modelo"], y=df_plot[col], name=col))
    fig.update_layout(barmode="group", template="plotly_white", title="Comparacion de metricas")
    save_and_open_plotly(fig, output_dir / filename)


def plot_residuals_over_time(
    origin_times: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    output_dir: Path,
    filename: str = "residuos_t1.html",
) -> None:
    """Residuos temporales del primer paso para base y corrector."""
    res_base = y_real[:, 0] - y_base[:, 0]
    res_corr = y_real[:, 0] - y_corrected[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=origin_times, y=res_base, mode="lines", name="Residual base", opacity=0.75))
    fig.add_trace(go.Scatter(x=origin_times, y=res_corr, mode="lines", name="Residual corregido", opacity=0.85))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Residuos t+1 en test", template="plotly_white", hovermode="x unified")
    save_and_open_plotly(fig, output_dir / filename)


# ==============================================================================
# SECCIÓN 13 — GUARDADO DE RESULTADOS
# ==============================================================================

def save_results(
    results: Dict,
    metrics_df: pd.DataFrame,
    tparams: Dict,
    cfg: Dict,
) -> None:
    """Guarda métricas en CSV y resultados completos en joblib."""
    output_dir = tparams["OUTPUT_DIR"]
    model_dir = tparams["MODEL_DIR"]

    metrics_path = output_dir / "metricas_test.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Métricas guardadas: {metrics_path}")

    joblib.dump(results, tparams["RESULTS_PATH"])
    print(f"  Resultados guardados: {tparams['RESULTS_PATH']}")
    
    # -------------------------------------------------------------------------
    # .txt de configuración del experimento
    # Registra todos los parámetros activos para reproducibilidad. NO ANDA
    # -------------------------------------------------------------------------
    exp_tag = tparams["EXP_TAG"]
    config_path = output_dir / f"config_{exp_tag}.txt"

    def _serializable(obj):
        """Convierte tipos no serializables (Path, bool numpy, etc.) a str/nativo."""
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    lines = []
    lines.append("=" * 60)
    lines.append(f"CONFIGURACION DEL EXPERIMENTO: {exp_tag}")
    lines.append(f"Fecha de ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Parámetros temporales derivados
    lines.append("--- PARAMETROS TEMPORALES DERIVADOS ---")
    for k, v in tparams.items():
        lines.append(f"  {k}: {_serializable(v)}")
    lines.append("")

    # CONFIG completo, sección por sección
    lines.append("--- CONFIG COMPLETO ---")
    for section, value in cfg.items():
        if isinstance(value, dict):
            lines.append(f"  [{section}]")
            for k, v in value.items():
                lines.append(f"    {k}: {_serializable(v)}")
        else:
            lines.append(f"  {section}: {_serializable(value)}")
    lines.append("")

    # Métricas resumen
    lines.append("--- METRICAS TEST ---")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("=" * 60)

    config_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Config guardada   : {config_path}")


# ==============================================================================
# SECCIÓN 14 — MAIN
# ==============================================================================

def main(cfg: Optional[Dict] = None) -> Dict:
    """
    Pipeline completo de extremo a extremo.

    Parámetros
    ----------
    cfg : dict, opcional
        Si se pasa, sobreescribe el CONFIG global. Útil para experimentos.

    Retorna
    -------
    dict con modelos, predicciones, métricas y datos de test.
    """
    if cfg is None:
        cfg = CONFIG

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    _setup_environment(cfg)
    tparams = resolve_temporal_params(cfg)

    print_header("CONFIGURACIÓN ACTIVA")
    print(f"  Resolución       : {cfg['RESOLUTION']}")
    print(f"  Frecuencia       : {tparams['FREQ']}")
    print(f"  Pasos/día        : {tparams['STEPS_PER_DAY']}")
    print(f"  Horizonte        : {tparams['HORIZON']} pasos ({cfg['HORIZON_DAYS']} día/s)")
    print(f"  Ventana histórica: {tparams['WINDOW_SIZE']} pasos ({cfg['WINDOW_DAYS']} días)")

    # -------------------------------------------------------------------------
    # 1. Dataset
    # -------------------------------------------------------------------------
    df_final, df_met = build_dataset(cfg, tparams)

    # -------------------------------------------------------------------------
    # 2. Features y ventanas
    # -------------------------------------------------------------------------
    X, X_future, y, origin_times = build_features_and_windows(
        df_final, df_met, cfg, tparams
    )

    # -------------------------------------------------------------------------
    # 3. Split + escalado
    # -------------------------------------------------------------------------
    data, scalers = temporal_split_and_scale(X, X_future, y, origin_times, cfg)
    y_scaler = scalers["y_scaler"]

    # -------------------------------------------------------------------------
    # 4. Modelo base
    # -------------------------------------------------------------------------
    base_model, scalers = train_or_load_base_model(data, scalers, cfg, tparams)
    y_scaler = scalers["y_scaler"]

    # -------------------------------------------------------------------------
    # 5. Predicciones base
    # -------------------------------------------------------------------------
    print_header("PREDICCIONES BASE")
    y_base_train = predict_base_real(base_model, data["X_train_scaled"], data["Xf_train_scaled"], y_scaler)
    y_base_val = predict_base_real(base_model, data["X_val_scaled"], data["Xf_val_scaled"], y_scaler)
    y_base_test = predict_base_real(base_model, data["X_test_scaled"], data["Xf_test_scaled"], y_scaler)

    base_preds = {"train": y_base_train, "val": y_base_val, "test": y_base_test}

    # -------------------------------------------------------------------------
    # 6. Métricas base
    # -------------------------------------------------------------------------
    peak_q = cfg["CORRECTOR"]["peak_quantile"]
    metrics_base_val = compute_metrics(data["y_val"], y_base_val, "CNN-LSTM - Val", peak_q)
    metrics_base_test = compute_metrics(data["y_test"], y_base_test, "CNN-LSTM - Test", peak_q)
    print("Métricas base (val) :", pd.DataFrame([metrics_base_val]).to_string(index=False))
    print("Métricas base (test):", pd.DataFrame([metrics_base_test]).to_string(index=False))

    # -------------------------------------------------------------------------
    # 7. Submuestreo day-ahead para el corrector
    # -------------------------------------------------------------------------
    data_corr, base_preds_corr = take_every_forecast_stride(
        data, base_preds, stride=tparams["FORECAST_STRIDE"]
    )

    # -------------------------------------------------------------------------
    # 8. Corrector residual
    # -------------------------------------------------------------------------
    corrector, meta_info = select_and_train_corrector(
        data_corr, base_preds_corr, cfg, tparams
    )

    # -------------------------------------------------------------------------
    # 9. Aplicar corrector en test
    # -------------------------------------------------------------------------
    y_corrected_test, residual_pred_test = apply_corrector(
        corrector, data_corr, base_preds_corr, cfg, tparams
    )

    # -------------------------------------------------------------------------
    # 10. Métricas finales
    # -------------------------------------------------------------------------
    metrics_base_da = compute_metrics(
        data_corr["y_test"], base_preds_corr["test"], "CNN-LSTM - Test day-ahead", peak_q
    )
    metrics_corr = compute_metrics(
        data_corr["y_test"], y_corrected_test, "CNN-LSTM + Corrector - Test day-ahead", peak_q
    )
    metrics_df = pd.DataFrame([metrics_base_da, metrics_corr])

    print_header("RESULTADOS FINALES")
    print(metrics_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 11. Gráficas
    # -------------------------------------------------------------------------
    out_dir = tparams["OUTPUT_DIR"]

    plot_first_step(
        data_corr["origin_test"], data_corr["y_test"],
        base_preds_corr["test"], y_corrected_test,
        output_dir=out_dir
    )

    plot_all_day_ahead(
        data_corr["origin_test"], data_corr["y_test"],
        base_preds_corr["test"], y_corrected_test,
        freq=tparams["FREQ"], horizon=tparams["HORIZON"],
        output_dir=out_dir
    )

    plot_horizon_example(
        data_corr["origin_test"], data_corr["y_test"],
        base_preds_corr["test"], y_corrected_test,
        freq=tparams["FREQ"], horizon=tparams["HORIZON"],
        output_dir=out_dir, sample_idx=-1
    )

    plot_peak_scatter(
        data_corr["y_test"], base_preds_corr["test"], y_corrected_test,
        peak_quantile=peak_q, output_dir=out_dir
    )

    plot_metrics_bar(metrics_df, output_dir=out_dir)

    plot_residuals_over_time(
        data_corr["origin_test"], data_corr["y_test"],
        base_preds_corr["test"], y_corrected_test,
        output_dir=out_dir
    )

    # Rango específico (ajustar fechas según datos disponibles)
    plot_selected_range(
        data_corr["origin_test"], data_corr["y_test"],
        base_preds_corr["test"], y_corrected_test,
        fecha_inicio=str(data_corr["origin_test"].min())[:10],
        fecha_fin=str((data_corr["origin_test"].min() + pd.Timedelta(days=8)).date()),
        output_dir=out_dir
    )

    # -------------------------------------------------------------------------
    # 12. Guardar
    # -------------------------------------------------------------------------
    results = {
        "origin_test": data_corr["origin_test"],
        "y_test": data_corr["y_test"],
        "y_base_test": base_preds_corr["test"],
        "y_corrected_test": y_corrected_test,
        "residual_pred_test": residual_pred_test,
        "metrics_df": metrics_df,
        "meta_info": meta_info,
        "config": cfg,
        "tparams": {k: str(v) if isinstance(v, Path) else v for k, v in tparams.items()},
    }

    save_results(results, metrics_df, tparams, cfg)

    return {
        "base_model": base_model,
        "corrector": corrector,
        "metrics_df": metrics_df,
        "data": data,
        "data_corr": data_corr,
        "base_preds": base_preds,
        "base_preds_corr": base_preds_corr,
        "y_corrected_test": y_corrected_test,
        "meta_info": meta_info,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    resultados = main()