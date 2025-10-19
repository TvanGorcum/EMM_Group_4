#Define functions that will evualuate:
#1. The models seperate for each subgroup
#2. The baseline models and the global model with subgroups as a dummy only
from typing import List, Dict
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
import ast

def ensure_dict(x):
    if isinstance(x, dict):
        return x.copy()
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, pd.DataFrame) and len(x) == 1:
        return x.iloc[0].to_dict()
    if isinstance(x, tuple) and len(x) >= 1:
        return ensure_dict(x[0])
    return dict(x)
#We still need to settle and explain appropriate
def evaluate_linear_model(
    model,
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str
) -> Dict[str, float]:
    X = df[X_cols].values
    y = df[y_col].values
    y_pred = model.predict(X)

    return {
        "r2": round(r2_score(y, y_pred), 3),
        "mae": round(mean_absolute_error(y, y_pred), 3),
        "mse": round(mean_squared_error(y, y_pred), 3)
    }

def _split_and(description: str):
    """
    Split a description on logical AND (either '∧' or '&') but ignore any
    ampersands that appear inside single/double quotes.
    """
    parts, buf = [], []
    in_s, in_d = False, False  # in single/double quotes
    for ch in description:
        if ch == "'" and not in_d:
            in_s = not in_s
            buf.append(ch)
        elif ch == '"' and not in_s:
            in_d = not in_d
            buf.append(ch)
        elif (ch == '∧' or ch == '&') and not in_s and not in_d:
            # logical AND outside quotes -> split
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

def _parse_atom(df: pd.DataFrame, atom: str) -> pd.Series:
    atom = atom.strip()

    # Interval form: <col> in (a, b], etc.
    m = re.fullmatch(r"(\w+)\s+in\s+([\(\[])\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*([\)\]])", atom)
    if m:
        col, lo_br, lo, hi, hi_br = m.groups()
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        lo = float(lo); hi = float(hi)
        series = pd.to_numeric(df[col], errors='coerce')
        mask = series.notna()
        mask &= (series > lo if lo_br == '(' else series >= lo)
        mask &= (series < hi if hi_br == ')' else series <= hi)
        return mask

    # Equality form: <col> == <value>
    m = re.fullmatch(r"(\w+)\s*==\s*(.+)", atom)
    if m:
        col, val = m.groups()
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        val = val.strip()
        if val in ("np.True_", "True"):
            py_val = True
        elif val in ("np.False_", "False"):
            py_val = False
        else:
            try:
                py_val = ast.literal_eval(val)  # handles quoted strings & numbers
            except Exception:
                py_val = val.strip("'").strip('"')
        return df[col] == py_val

    raise ValueError(f"Unrecognized condition syntax: '{atom}'")

def _description_to_mask(df: pd.DataFrame, description: str) -> pd.Series:
    parts = _split_and(description)
    if not parts:
        raise ValueError(f"Empty description: {description}")
    mask = pd.Series(True, index=df.index)
    for atom in parts:
        mask &= _parse_atom(df, atom)
    return mask

def get_rows_subgroup(models, df: pd.DataFrame) -> dict:
    out = {}
    for i, model in enumerate(models):
        desc = model.get('description', '')
        if not desc:
            raise KeyError(f"Model at index {i} has no 'description' key.")
        sub = df.loc[_description_to_mask(df, desc)].copy()
        sub['__subgroup_description'] = desc
        out[desc] = sub
    return out