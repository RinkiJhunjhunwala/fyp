"""
End-to-End Execution Script for Phase 2:
Data Preprocessing & Feature Engineering
Produces the final patient-level tabular dataset (one row per patient).
"""

from pathlib import Path
import pandas as pd
from src.preprocessing import preprocess_vitals
from src.feature_engineering import extract_features_from_windowed_df

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_VITALS_PATH = PROJECT_ROOT / "data" / "final" / "cardiac_patient_labelled_vitals.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FEATURES_PATH = PROCESSED_DIR / "patient_vital_features_24h.csv"

def run_phase_2(window_hours: float = 24.0):
    print("=" * 70)
    print("PHASE 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    
    print(f"Loading raw vitals from: {RAW_VITALS_PATH}")
    raw_df = pd.read_csv(RAW_VITALS_PATH, low_memory=False)
    print(f"Raw measurements loaded: {len(raw_df):,} rows")
    
    print(f"\nPreprocessing observations (Window: first {window_hours} hours)...")
    windowed_df = preprocess_vitals(
        raw_df,
        observation_window_hours=window_hours,
        filter_implausible=True
    )
    
    # Save the windowed time-series data as well for audit/visualization
    windowed_output_path = PROCESSED_DIR / "windowed_vitals_24h.csv"
    windowed_df.to_csv(windowed_output_path, index=False)
    print(f"Saved windowed time-series to: {windowed_output_path}")
    
    print("\nExtracting patient-level statistical features (one row per patient)...")
    feature_df = extract_features_from_windowed_df(windowed_df)
    
    print(f"Patient features extracted:")
    print(f"- Total patients (rows): {len(feature_df)}")
    print(f"- Total features (columns): {len(feature_df.columns)}")
    print(f"- Unique patients: {feature_df['subject_id'].nunique()}")
    print(f"- Unique admissions: {feature_df['hadm_id'].nunique()}")
    print(f"- Target distribution (ami_during_admission):")
    print(feature_df["ami_during_admission"].value_counts(dropna=False))
    
    print("\nMissing values across extracted features:")
    null_counts = feature_df.isnull().sum()
    if null_counts.sum() == 0:
        print("Zero missing values across all features!")
    else:
        print(null_counts[null_counts > 0])
        
    feature_df.to_csv(OUTPUT_FEATURES_PATH, index=False)
    print(f"\nSaved patient-level features to: {OUTPUT_FEATURES_PATH}")
    print("=" * 70)
    return feature_df

if __name__ == "__main__":
    run_phase_2()
