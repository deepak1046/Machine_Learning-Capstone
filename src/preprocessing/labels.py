"""Label standardization for CICDDoS2019.

The two capture days label the same attacks differently: `01-12` (training day) uses a
`DrDoS_` prefix and spells UDP lag as `UDP-lag`, while `03-11` (testing day) uses bare
names. Both are folded into one class per attack type.
"""

import json
from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DrDoS_DNS": "DNS",
    "DrDoS_LDAP": "LDAP",
    "LDAP": "LDAP",
    "DrDoS_MSSQL": "MSSQL",
    "MSSQL": "MSSQL",
    "DrDoS_NetBIOS": "NetBIOS",
    "NetBIOS": "NetBIOS",
    "DrDoS_NTP": "NTP",
    "DrDoS_SNMP": "SNMP",
    "DrDoS_SSDP": "SSDP",
    "DrDoS_UDP": "UDP",
    "UDP": "UDP",
    "Syn": "Syn",
    "TFTP": "TFTP",
    "UDP-lag": "UDPLag",
    "UDPLag": "UDPLag",
    "Portmap": "Portmap",
}

# 439 rows out of 69.9M, 45,000x smaller than the largest class -- not learnable.
DROPPED_LABELS = {"WebDDoS"}

CLASSES = sorted(set(LABEL_MAP.values()))
LABEL_ENCODING = {name: i for i, name in enumerate(CLASSES)}

# Family-level grouping: collapses the three class clusters found (2026-09-27 EDA,
# 2026-10-02 ceiling analysis) to be behaviorally indistinguishable from each other
# without ports -- i.e. flows that are byte-for-byte identical except for the
# (partly garbage, fragment-polluted) port fields but carry different ground-truth
# labels. Collapsing them raises the behavior-only ceiling from macro-F1 0.808 to
# 0.966 (verified 2026-10-10 against training_sample.parquet) because the ambiguity
# was always BETWEEN members of the same family, never across families.
FAMILY_MAP = {
    "BENIGN": "BENIGN",
    "MSSQL": "MSSQL",
    "NTP": "NTP",
    "Syn": "Syn",
    "TFTP": "TFTP",
    "DNS": "Reflection-Amp",
    "LDAP": "Reflection-Amp",
    "SNMP": "Reflection-Amp",
    "NetBIOS": "NetBIOS-Portmap",
    "Portmap": "NetBIOS-Portmap",
    "SSDP": "UDP-Family",
    "UDP": "UDP-Family",
    "UDPLag": "UDP-Family",
}
FAMILY_CLASSES = sorted(set(FAMILY_MAP.values()))
FAMILY_ENCODING = {name: i for i, name in enumerate(FAMILY_CLASSES)}


def family_labels(labels: pd.Series) -> pd.Series:
    """Map standardized per-attack labels to the coarser family grouping."""
    return labels.map(FAMILY_MAP).astype("string")


def standardize_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Fold raw label variants into canonical class names, drop the excluded classes,
    and add an integer `Label_Encoded` column.

    Raises on an unrecognized label rather than silently passing it through -- a new
    spelling appearing mid-merge would otherwise corrupt the class set unnoticed.
    """
    df = df[~df[label_col].isin(DROPPED_LABELS)].copy()

    unknown = set(df[label_col].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unrecognized label values: {sorted(unknown)}")

    df[label_col] = df[label_col].map(LABEL_MAP).astype("string")
    df["Label_Encoded"] = df[label_col].map(LABEL_ENCODING).astype("int8")
    return df


def save_encoding(dest_file: Path) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_text(json.dumps(LABEL_ENCODING, indent=2))
