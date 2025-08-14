import re, os, joblib
import numpy as np
import pandas as pd

DIR_MAP_PATH = os.getenv("DIR_MAP_PATH", "models/dir_freq_map.joblib")

def _load_dir_freq_map(path: str = DIR_MAP_PATH) -> dict:
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return {}
    return {}

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create the exact feature set expected by the trained pipeline."""
    df = df_raw.copy()

    # Dates
    df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
    df["Year"]  = df["Release_Date"].dt.year
    df["Month"] = df["Release_Date"].dt.month

    # Duration
    def parse_duration(s):
        if pd.isna(s): return np.nan, np.nan
        s = str(s).lower()
        m_min = re.search(r"(\d+)\s*min", s)
        m_sea = re.search(r"(\d+)\s*season", s)
        return (int(m_min.group(1)) if m_min else np.nan,
                int(m_sea.group(1)) if m_sea else 0)
    mins, seas = zip(*df["Duration"].apply(parse_duration))
    df["Duration_Min"] = mins
    df["Seasons"] = seas

    # Simple signals
    df["DescLen"]  = df["Description"].fillna("").str.len()
    df["TitleLen"] = df["Title"].fillna("").str.len()
    df["Cast_Count"] = df["Cast"].fillna("").apply(lambda s: 0 if s=="" else len([x for x in s.split(",") if x.strip()]))
    df["Has_Director"] = df["Director"].notna().astype(int)
    df["Country"] = df["Country"].fillna("").astype(str).str.split(",").str[0].str.strip()
    df["Rating"]  = df["Rating"].astype(str).str.upper().str.strip()

    # Director frequency
    dir_map = _load_dir_freq_map()
    df["Director_Freq"] = df["Director"].map(dir_map).fillna(0).astype(int)

    # Column order expected by the pipeline
    use_text = [c for c in ["Title","Description"] if c in df.columns]
    use_num  = [c for c in ["Year","Month","Duration_Min","Seasons","DescLen","TitleLen","Cast_Count","Director_Freq"] if c in df.columns]
    use_cat  = [c for c in ["Country","Rating","Type","Has_Director"] if c in df.columns]
    return df[use_text + use_num + use_cat]
