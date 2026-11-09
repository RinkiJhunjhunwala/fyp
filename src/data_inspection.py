"""
Data Inspection Script for Privacy-Preserving AMI Classification Project
Inspects all supplied files, schemas, distributions, identifiers, timestamps,
and data quality without modifying raw data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def inspect_file(filepath: Path, nrows: int = 5):
    print("=" * 80)
    print(f"FILE: {filepath.relative_to(PROJECT_ROOT)} (Size: {filepath.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 80)
    
    # Read sample
    df_sample = pd.read_csv(filepath, nrows=nrows)
    print("\nColumns:", list(df_sample.columns))
    print(f"Sample (first {nrows} rows):")
    print(df_sample.to_string(index=False))

def full_analysis_selected_cohort():
    cohort_path = DATA_DIR / "final" / "selected_90_patients.csv"
    if not cohort_path.exists():
        print(f"Cohort file not found: {cohort_path}")
        return
        
    print("\n" + "#" * 80)
    print("ANALYSIS: selected_90_patients.csv")
    print("#" * 80)
    df = pd.read_csv(cohort_path)
    print(f"Total rows: {len(df)}")
    print(f"Unique subject_id: {df['subject_id'].nunique()}")
    print(f"Unique hadm_id: {df['hadm_id'].nunique()}")
    print("\nColumn info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    if "ami_during_admission" in df.columns:
        print("\nLabel distribution (ami_during_admission):")
        print(df["ami_during_admission"].value_counts(dropna=False))
        
    print("\nSample records:")
    print(df.head(10).to_string())

def full_analysis_vitals():
    vitals_path = DATA_DIR / "final" / "cardiac_patient_labelled_vitals.csv"
    if not vitals_path.exists():
        print(f"Vitals file not found: {vitals_path}")
        return
        
    print("\n" + "#" * 80)
    print("ANALYSIS: cardiac_patient_labelled_vitals.csv")
    print("#" * 80)
    
    # Read full vitals or in chunks
    df = pd.read_csv(vitals_path, low_memory=False)
    print(f"Total vitals records: {len(df):,}")
    print(f"Unique subject_id: {df['subject_id'].nunique()}")
    print(f"Unique hadm_id: {df['hadm_id'].nunique()}")
    if 'stay_id' in df.columns:
        print(f"Unique stay_id: {df['stay_id'].nunique()}")
    
    print("\nColumns & Types:")
    print(df.dtypes)
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    print("\nDuplicates (exact duplicate rows):", df.duplicated().sum())
    
    if "label" in df.columns and "itemid" in df.columns:
        print("\nVital signs types (itemid & label counts):")
        vital_summary = df.groupby(["itemid", "label", "valueuom"]).agg(
            count=("valuenum", "count"),
            null_count=("valuenum", lambda x: x.isnull().sum()),
            min_val=("valuenum", "min"),
            mean_val=("valuenum", "mean"),
            max_val=("valuenum", "max"),
            std_val=("valuenum", "std")
        ).reset_index()
        print(vital_summary.to_string(index=False))
        
    print("\nTarget Label Distribution in vitals (ami_during_admission):")
    if "ami_during_admission" in df.columns:
        patient_labels = df.groupby("subject_id")["ami_during_admission"].first()
        print("At patient level:")
        print(patient_labels.value_counts(dropna=False))
        print("At measurement level:")
        print(df["ami_during_admission"].value_counts(dropna=False))
        
    # Check observation counts per patient
    obs_per_patient = df.groupby("subject_id").size()
    print("\nObservations per patient:")
    print(f"Min: {obs_per_patient.min()}, Median: {obs_per_patient.median()}, Mean: {obs_per_patient.mean():.1f}, Max: {obs_per_patient.max()}")
    
    # Check timestamps and observation windows
    if "charttime" in df.columns and "admittime" in df.columns:
        df["charttime_dt"] = pd.to_datetime(df["charttime"], errors="coerce")
        df["admittime_dt"] = pd.to_datetime(df["admittime"], errors="coerce")
        df["time_since_admit_hours"] = (df["charttime_dt"] - df["admittime_dt"]).dt.total_seconds() / 3600.0
        
        print("\nTime since admission (hours):")
        print(f"Min: {df['time_since_admit_hours'].min():.2f}h, 25%: {df['time_since_admit_hours'].quantile(0.25):.2f}h, Median: {df['time_since_admit_hours'].median():.2f}h, 75%: {df['time_since_admit_hours'].quantile(0.75):.2f}h, Max: {df['time_since_admit_hours'].max():.2f}h")
        
        # Observations within first 24h
        obs_within_24h = df[df["time_since_admit_hours"].between(0, 24)]
        print(f"Total observations in [0, 24h] since admission: {len(obs_within_24h):,} ({len(obs_within_24h)/len(df)*100:.1f}%)")
        patients_with_24h_vitals = obs_within_24h["subject_id"].nunique()
        print(f"Patients with observations in first 24h: {patients_with_24h_vitals} / {df['subject_id'].nunique()}")

        # Also check relative to first charttime per patient
        first_charttime = df.groupby("subject_id")["charttime_dt"].transform("min")
        df["time_since_first_vital_hours"] = (df["charttime_dt"] - first_charttime).dt.total_seconds() / 3600.0
        obs_within_24h_vitals = df[df["time_since_first_vital_hours"].between(0, 24)]
        print(f"Total observations in [0, 24h] since first vital recorded: {len(obs_within_24h_vitals):,} ({len(obs_within_24h_vitals)/len(df)*100:.1f}%)")
        patients_with_first_24h = obs_within_24h_vitals["subject_id"].nunique()
        print(f"Patients with observations in first 24h of vitals: {patients_with_first_24h} / {df['subject_id'].nunique()}")

def inspect_other_files():
    other_files = [
        DATA_DIR / "troponin" / "cardiac_patient_troponin.csv",
        DATA_DIR / "adjudication" / "cardiac_adjudication_review.csv",
        DATA_DIR / "candidates" / "cardiac_admissions.csv"
    ]
    for f in other_files:
        if f.exists():
            inspect_file(f, nrows=3)

if __name__ == "__main__":
    full_analysis_selected_cohort()
    full_analysis_vitals()
    inspect_other_files()
