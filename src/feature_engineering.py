"""
Feature Engineering Module for AMI Classification
Transforms time-series vital-sign observations into fixed-length patient-level feature vectors.
Each patient receives exactly ONE row with comprehensive statistical features.
Also provides inference-time feature extraction for the web application.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CORE_VITALS = ["HR", "SBP", "DBP", "SpO2"]

def compute_vital_statistics(values: np.ndarray, time_hours: np.ndarray) -> dict:
    """
    Computes statistical features for a single vital sign time-series.
    
    Parameters:
    -----------
    values : np.ndarray
        Array of numerical vital sign readings.
    time_hours : np.ndarray
        Array of elapsed hours corresponding to the readings.
        
    Returns:
    --------
    dict containing mean, min, max, std, range, first, last, change, trend, count.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "range": np.nan,
            "first": np.nan,
            "last": np.nan,
            "change": np.nan,
            "trend": np.nan,
            "count": 0,
        }
    
    v_mean = float(np.mean(values))
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    v_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    v_range = float(v_max - v_min)
    v_first = float(values[0])
    v_last = float(values[-1])
    v_change = float(v_last - v_first)
    
    # Calculate trend (linear slope: units per hour)
    if n >= 2:
        dt = time_hours - np.mean(time_hours)
        dt_var = np.sum(dt ** 2)
        if dt_var > 1e-6:
            trend = float(np.sum(dt * (values - v_mean)) / dt_var)
        else:
            trend = 0.0
    else:
        trend = 0.0
        
    return {
        "mean": v_mean,
        "min": v_min,
        "max": v_max,
        "std": v_std,
        "range": v_range,
        "first": v_first,
        "last": v_last,
        "change": v_change,
        "trend": trend,
        "count": n,
    }

def extract_features_from_windowed_df(
    df: pd.DataFrame,
    vitals: list = None
) -> pd.DataFrame:
    """
    Aggregates windowed vital observations into one feature vector per patient.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Windowed and cleaned vitals DataFrame.
    vitals : list
        List of vital categories to process (default: HR, SBP, DBP, SpO2).
        
    Returns:
    --------
    pd.DataFrame
        One row per patient/admission containing statistical features and the target label.
    """
    if vitals is None:
        vitals = CORE_VITALS
        
    patient_rows = []
    
    for (subject_id, hadm_id), p_df in df.groupby(["subject_id", "hadm_id"]):
        p_df = p_df.sort_values(by="charttime")
        
        # Determine target label if present
        target_label = p_df["ami_during_admission"].iloc[0] if "ami_during_admission" in p_df.columns else np.nan
        
        row = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "ami_during_admission": target_label
        }
        
        for vital in vitals:
            sub = p_df[p_df["vital_category"] == vital]
            if len(sub) > 0:
                vals = sub["valuenum"].values.astype(float)
                # Elapsed hours
                t0 = p_df["charttime"].min()
                t_hours = ((sub["charttime"] - t0).dt.total_seconds() / 3600.0).values.astype(float)
                stats = compute_vital_statistics(vals, t_hours)
            else:
                stats = compute_vital_statistics(np.array([]), np.array([]))
                
            for stat_name, val in stats.items():
                row[f"{vital}_{stat_name}"] = val
                
        # Clinically informative derived ratios
        hr_mean = row.get("HR_mean", np.nan)
        sbp_mean = row.get("SBP_mean", np.nan)
        sbp_min = row.get("SBP_min", np.nan)
        hr_max = row.get("HR_max", np.nan)
        dbp_mean = row.get("DBP_mean", np.nan)
        
        row["shock_index_mean"] = hr_mean / sbp_mean if (pd.notnull(hr_mean) and pd.notnull(sbp_mean) and sbp_mean > 0) else np.nan
        row["shock_index_max"] = hr_max / sbp_min if (pd.notnull(hr_max) and pd.notnull(sbp_min) and sbp_min > 0) else np.nan
        row["pulse_pressure_mean"] = sbp_mean - dbp_mean if (pd.notnull(sbp_mean) and pd.notnull(dbp_mean)) else np.nan
        
        patient_rows.append(row)
        
    feature_df = pd.DataFrame(patient_rows)
    return feature_df

def extract_features_from_single_reading(
    hr: float,
    sbp: float,
    dbp: float,
    spo2: float
) -> pd.DataFrame:
    """
    Transforms a single clinical reading (e.g. from web UI form) into the exact
    same feature vector representation as the 24-hour training pipeline.
    """
    row = {
        "subject_id": 0,
        "hadm_id": 0,
    }
    inputs = {"HR": float(hr), "SBP": float(sbp), "DBP": float(dbp), "SpO2": float(spo2)}
    
    for vital, val in inputs.items():
        row[f"{vital}_mean"] = val
        row[f"{vital}_min"] = val
        row[f"{vital}_max"] = val
        row[f"{vital}_std"] = 0.0
        row[f"{vital}_range"] = 0.0
        row[f"{vital}_first"] = val
        row[f"{vital}_last"] = val
        row[f"{vital}_change"] = 0.0
        row[f"{vital}_trend"] = 0.0
        row[f"{vital}_count"] = 1
        
    row["shock_index_mean"] = inputs["HR"] / inputs["SBP"] if inputs["SBP"] > 0 else 0.0
    row["shock_index_max"] = inputs["HR"] / inputs["SBP"] if inputs["SBP"] > 0 else 0.0
    row["pulse_pressure_mean"] = inputs["SBP"] - inputs["DBP"]
    
    return pd.DataFrame([row])
