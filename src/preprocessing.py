"""
Preprocessing Pipeline for AMI Vital Signs Dataset
Handles timestamp parsing, chronological sorting, physiological filtering,
blood pressure signal reconciliation, and observation window enforcement.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Physiological plausibility ranges (ICU clinical standards)
VITAL_PHYSIOLOGICAL_RANGES = {
    "HR": (20.0, 240.0),    # Heart Rate (bpm)
    "SBP": (30.0, 260.0),   # Systolic BP (mmHg)
    "DBP": (20.0, 160.0),   # Diastolic BP (mmHg)
    "MBP": (30.0, 200.0),   # Mean Arterial Pressure (mmHg)
    "SpO2": (50.0, 100.0),  # Oxygen Saturation (%)
}

# Mapping of MIMIC-IV item IDs to canonical categories and signal source
ITEM_ID_MAPPING = {
    220045: ("HR", "HR"),
    220050: ("SBP", "ABP"),   # Arterial Line Systolic
    220179: ("SBP", "NIBP"),  # Non-Invasive Cuff Systolic
    220051: ("DBP", "ABP"),   # Arterial Line Diastolic
    220180: ("DBP", "NIBP"),  # Non-Invasive Cuff Diastolic
    220052: ("MBP", "ABP"),   # Arterial Line Mean
    220181: ("MBP", "NIBP"),  # Non-Invasive Cuff Mean
    220277: ("SpO2", "SpO2"), # Pulse Oximetry
}

def map_vital_category(itemid: int):
    """Maps MIMIC itemid to canonical vital category and measurement method."""
    return ITEM_ID_MAPPING.get(itemid, ("Other", "Other"))

def preprocess_vitals(
    df: pd.DataFrame,
    observation_window_hours: float = 24.0,
    filter_implausible: bool = True
) -> pd.DataFrame:
    """
    Cleans raw vitals DataFrame and restricts observations to the defined window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw vitals dataframe containing subject_id, hadm_id, charttime, itemid, valuenum, etc.
    observation_window_hours : float
        Length of the observation window in hours from each patient's first recorded vital.
    filter_implausible : bool
        Whether to filter out sensor artifacts outside physiological bounds.
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed, filtered, and windowed observations.
    """
    df = df.copy()
    
    # 1. Parse timestamps
    df["charttime"] = pd.to_datetime(df["charttime"], format="mixed")
    
    # 2. Map item IDs to canonical categories
    df[["vital_category", "measurement_type"]] = df["itemid"].apply(
        lambda x: pd.Series(map_vital_category(x))
    )
    df = df[df["vital_category"].isin(["HR", "SBP", "DBP", "MBP", "SpO2"])].copy()
    
    # 3. Drop records with missing or non-positive measurements
    df = df.dropna(subset=["valuenum", "charttime", "subject_id"]).copy()
    
    # 4. Filter implausible physiological outliers
    if filter_implausible:
        initial_len = len(df)
        valid_masks = []
        for cat, (low, high) in VITAL_PHYSIOLOGICAL_RANGES.items():
            mask = (df["vital_category"] == cat) & (df["valuenum"] >= low) & (df["valuenum"] <= high)
            valid_masks.append(mask)
        df = df[pd.concat(valid_masks, axis=1).any(axis=1)].copy()
        filtered_out = initial_len - len(df)
        print(f"Filtered out {filtered_out:,} implausible measurements ({filtered_out/initial_len*100:.2f}%).")
        
    # 5. Sort chronologically per patient
    df = df.sort_values(by=["subject_id", "charttime"]).reset_index(drop=True)
    
    # 6. Reconcile dual BP measurements (ABP vs NIBP) at identical charttime
    # If both ABP and NIBP are present at the exact same minute, ABP (arterial catheter) takes precedence
    priority_order = {"ABP": 0, "HR": 1, "SpO2": 1, "NIBP": 2, "Other": 3}
    df["priority"] = df["measurement_type"].map(priority_order).fillna(3)
    df = df.sort_values(by=["subject_id", "charttime", "vital_category", "priority"])
    df = df.drop_duplicates(subset=["subject_id", "charttime", "vital_category"], keep="first")
    df = df.drop(columns=["priority"])
    
    # 7. Apply patient-specific observation window (e.g. first 24 hours of vitals)
    first_charttime = df.groupby("subject_id")["charttime"].transform("min")
    df["elapsed_hours"] = (df["charttime"] - first_charttime).dt.total_seconds() / 3600.0
    
    windowed_df = df[
        (df["elapsed_hours"] >= 0.0) & (df["elapsed_hours"] <= observation_window_hours)
    ].copy()
    
    print(f"Total windowed observations ([0, {observation_window_hours}h]): {len(windowed_df):,} across {windowed_df['subject_id'].nunique()} patients.")
    return windowed_df

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    raw_vitals_path = root / "data" / "final" / "cardiac_patient_labelled_vitals.csv"
    if raw_vitals_path.exists():
        raw_df = pd.read_csv(raw_vitals_path, low_memory=False)
        clean_df = preprocess_vitals(raw_df, observation_window_hours=24.0)
        print("Cleaned DataFrame sample:")
        print(clean_df.head())
