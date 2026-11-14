"""Cleaning, clinical grouping, and categorical encoding from the notebook."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MEDICINES = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
AGE_MAP = {f"[{i}-{i+10})": i+5 for i in range(0, 100, 10)}


def _icd9_group(value):
    value = str(value)
    if "V" in value or "E" in value:
        return 18
    try:
        code = float(value)
    except (TypeError, ValueError):
        return pd.NA
    ranges = [(1,139),(140,239),(240,279),(280,289),(290,319),(320,389),
              (390,459),(460,519),(520,579),(580,629),(630,679),(680,709),
              (710,739),(740,759),(760,779),(780,799),(800,999)]
    for group, (lo, hi) in enumerate(ranges, start=1):
        if lo <= code <= hi:
            return group
    return pd.NA


def preprocess_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the study's cleaning and encoding steps; returns model-ready data.

    LabelEncoder mappings are learned from the complete cleaned dataset, as in
    the supplied notebook. For deployment, persist encoders and fit them only
    on training data; this function is intended to reproduce the research flow.
    """
    df = raw.copy()
    drop_cols = [c for c in ["weight", "citoglipton", "examide", "payer_code", "medical_specialty"] if c in df]
    df.drop(columns=drop_cols, inplace=True)
    if "race" in df:
        df["race"] = df["race"].replace("?", "Other")
    if "gender" in df:
        df = df.loc[~df["gender"].isin(["Unknown/Invalid", "?"])].copy()
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df]
    for col in diag_cols:
        df = df.loc[df[col].notna() & (df[col] != "?")].copy()
    if "age" in df:
        df["age"] = df["age"].map(AGE_MAP)
    if "admission_type_id" in df:
        df["admission_type_id"] = df["admission_type_id"].replace({6:5, 8:5, 4:1, 7:2})
    if "discharge_disposition_id" in df:
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace({6:1,8:1,25:18,26:18,12:18,19:11,20:11,21:11,14:13,9:2,10:2,23:2,27:2,28:2,29:2,4:3,5:3,15:3,24:3,17:16})
    if "admission_source_id" in df:
        df["admission_source_id"] = df["admission_source_id"].replace({15:9,17:9,20:9,21:9,2:1,3:1,10:4,22:4,6:5,18:5,19:5,25:5,26:5,12:11,13:11,14:11})
    for col in diag_cols:
        df[col] = df[col].map(_icd9_group)
    if "change" in df:
        df["change"] = df["change"].replace("Ch", "Yes")
    for col in MEDICINES:
        if col in df:
            df[col] = (df[col] != "No").astype(int)
    for col in ["A1Cresult", "max_glu_serum"]:
        if col in df:
            df[col] = df[col].fillna("NotMeasured")
    for col in ["change", "diabetesMed"]:
        if col in df:
            df[col] = (df[col] != "No").astype(int)
    if "readmitted" in df:
        df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    if "patient_nbr" in df:
        df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
    df.drop(columns=[c for c in ["encounter_id", "patient_nbr"] if c in df], inplace=True)
    categorical = df.select_dtypes(include="object").columns
    for col in categorical:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))
    return df
