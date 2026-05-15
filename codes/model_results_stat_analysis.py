from __future__ import annotations

import argparse
import csv
import itertools
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# ============================================================
# Configurações globais
# ============================================================

NUMERIC_FAKE_VALUES = {1}
NUMERIC_REAL_VALUES = {0}

ASSUME_SHARED_TEST_ORDER_WHEN_IDX_MISSING = True
BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 42

POSITIVE_LABEL = "FAKE"
NEGATIVE_LABEL = "REAL"

DEFAULT_MODEL_ORDER = None

PHASE_INPUTS = {
    "test": {
        "classic_models": {
            "path": r"results\classical_models\classical_models_test_predictions.csv",
            "filters": {},
            "true_col": "true_label_id",
            "pred_col": "pred_label_id",
            "model_col": "model",
        },
        "zero_shot": {
            "path": r"model_result\zero-shot_result.csv",
            "filters": {},
            "true_col": "true",
            "pred_col": "pred",
            "model_col": "model",
            "idx_prefix_filters": [
                "COVID19B_R",
                "FAKEBR_",
                "FNEWSSET_",
                "FRECOGNA_",
                "MUMINPT_",
                "CENTRALFATOS_",
                "FCN_",
                "TRE300_",
                "FAKETRUEBR_",
                "FACTCKBR_",
            ],
        },
    },
    "val": {
        "classic_models": {
            "path": r"results\classical_models\classical_models_val_predictions.csv",
            "filters": {},
            "true_col": "true_label_id",
            "pred_col": "pred_label_id",
            "model_col": "model",
        },
        "zero_shot": {
            "path": r"model_result\zero-shot_result.csv",
            "filters": {},
            "true_col": "true",
            "pred_col": "pred",
            "model_col": "model",
            "idx_prefix_filters": [
                "COVID19B_R",
                "FAKEBR_",
                "FNEWSSET_",
                "FRECOGNA_",
                "MUMINPT_",
                "CENTRALFATOS_",
                "FCN_",
                "TRE300_",
                "FAKETRUEBR_",
                "FACTCKBR_",
            ],
        },
    },
}

RAW_RESPONSE_COL_CANDIDATES = [
    "raw_response",
    "response",
    "full_response",
    "model_response",
]

IDX_CANDIDATES = [
    "idx",
    "id",
    "sample_id",
    "news_id",
    "item_id",
    "claim_id",
    "instance_id",
]

TRUE_COL_CANDIDATES = [
    "true_label_id",
    "true",
    "label",
    "gold",
    "gold_label",
    "y_true",
    "target",
]

PRED_COL_CANDIDATES = [
    "pred_label_id",
    "pred",
    "prediction",
    "predicted_label",
    "y_pred",
    "output",
]

MODEL_COL_CANDIDATES = [
    "model",
    "model_name",
]


# ============================================================
# Estruturas auxiliares
# ============================================================

@dataclass
class SourceConfig:
    name: str
    path: Path
    filters: Dict[str, object]
    true_col: Optional[str] = None
    pred_col: Optional[str] = None
    model_col: Optional[str] = None
    idx_prefix_filters: Optional[List[str]] = None


# ============================================================
# Utilidades de texto e colunas
# ============================================================

def canonicalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def canonicalize_text_loose(x: object) -> str:
    return (
        canonicalize_text(x)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def to_ascii_upper(x: object) -> str:
    s = canonicalize_text(x).upper()
    return (
        s.replace("Á", "A")
        .replace("À", "A")
        .replace("Â", "A")
        .replace("Ã", "A")
        .replace("É", "E")
        .replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ô", "O")
        .replace("Õ", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
    )


def find_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    existing = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]

    normalized_map = {canonicalize_text_loose(c): c for c in df.columns}
    for cand in candidates:
        key = canonicalize_text_loose(cand)
        if key in normalized_map:
            return normalized_map[key]
    return None


def infer_idx_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df, IDX_CANDIDATES)


def infer_model_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df, MODEL_COL_CANDIDATES)


def infer_raw_response_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df, RAW_RESPONSE_COL_CANDIDATES)


# ============================================================
# Zero-shot: heurísticas para indefinido
# ============================================================

def looks_like_fake_variant(text: object) -> bool:
    s = to_ascii_upper(text)
    if not s:
        return False

    tokens = re.findall(r"[A-Z]+", s)
    for tok in tokens:
        if tok.startswith("FAK") and len(tok) >= 4:
            return True
    return False


def looks_like_apology_or_refusal(text: object) -> bool:
    s = to_ascii_upper(text)
    if not s:
        return False

    refusal_patterns = [
        "DESCULPA",
        "DESCULPE",
        "SINTO MUITO",
        "NAO POSSO",
        "NAO POSSO AJUDAR",
        "NAO CONSIGO",
        "NAO POSSO FORNECER",
        "NAO POSSO CLASSIFICAR",
        "NAO E POSSIVEL",
        "CANNOT",
        "SORRY",
        "I CANNOT",
        "I CAN'T",
    ]

    return any(pat in s for pat in refusal_patterns)


# ============================================================
# Normalização de rótulos
# ============================================================

def normalize_label(
    value: object,
    raw_response: object = None,
    source_name: Optional[str] = None,
    is_prediction: bool = False,
) -> str:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        if int(value) in NUMERIC_FAKE_VALUES:
            return POSITIVE_LABEL
        if int(value) in NUMERIC_REAL_VALUES:
            return NEGATIVE_LABEL

    if isinstance(value, float) and value.is_integer():
        ivalue = int(value)
        if ivalue in NUMERIC_FAKE_VALUES:
            return POSITIVE_LABEL
        if ivalue in NUMERIC_REAL_VALUES:
            return NEGATIVE_LABEL

    s_ascii = to_ascii_upper(value)

    fake_tokens = {
        "FAKE",
        "FALSO",
        "INVERIDICO",
        "INVERIDICA",
        "FALSE",
        "FAKENEWS",
        "FAKE NEWS",
        "MISINFORMATION",
        "RUMOR",
        "RUMOUR",
    }

    real_tokens = {
        "REAL",
        "VERDADEIRO",
        "VERIDICO",
        "VERIDICA",
        "TRUE",
        "NONFAKE",
        "NOT FAKE",
    }

    undefined_tokens = {
        "INDEFINIDO",
        "INDEFINIDA",
        "UNDEFINED",
        "N/A",
        "NA",
        "NONE",
        "NULL",
    }

    if s_ascii in fake_tokens:
        return POSITIVE_LABEL
    if s_ascii in real_tokens:
        return NEGATIVE_LABEL

    if s_ascii.isdigit():
        ivalue = int(s_ascii)
        if ivalue in NUMERIC_FAKE_VALUES:
            return POSITIVE_LABEL
        if ivalue in NUMERIC_REAL_VALUES:
            return NEGATIVE_LABEL

    if is_prediction and source_name == "zero_shot" and s_ascii in undefined_tokens:
        if looks_like_fake_variant(raw_response):
            return POSITIVE_LABEL
        if looks_like_apology_or_refusal(raw_response):
            return np.nan
        return np.nan

    raise ValueError(f"Rótulo não reconhecido: {value!r}")


# ============================================================
# Leitura robusta de CSV
# ============================================================

def read_csv_flex(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    encodings = ["utf-8", "latin1"]

    for encoding in encodings:
        try:
            with open(path, encoding=encoding, newline="") as f:
                first_line = f.readline().rstrip("\r\n")

            orig_header = next(
                csv.reader(
                    [first_line],
                    delimiter=",",
                    quotechar='"',
                    doublequote=True,
                    escapechar="\\",
                )
            )
            n_header = len(orig_header)

            max_cols = n_header
            with open(path, encoding=encoding, newline="") as f:
                rdr = csv.reader(
                    f,
                    delimiter=",",
                    quotechar='"',
                    doublequote=True,
                    escapechar="\\",
                )
                for row in rdr:
                    if len(row) > max_cols:
                        max_cols = len(row)

            if max_cols == n_header:
                return pd.read_csv(
                    path,
                    sep=",",
                    engine="python",
                    encoding=encoding,
                    dtype=str,
                )

            extra_names = [f"extra_{i+1}" for i in range(max_cols - n_header)]
            names = orig_header + extra_names

            df = pd.read_csv(
                path,
                sep=",",
                header=None,
                names=names,
                encoding=encoding,
                dtype=str,
            )

            df = df.iloc[1:].reset_index(drop=True)
            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Não foi possível decodificar o arquivo: {path}")


# ============================================================
# Filtros e padronização
# ============================================================

def apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()

    for col, expected in filters.items():
        real_col = find_existing_column(out, [col])
        if real_col is None:
            raise KeyError(
                f"Coluna de filtro '{col}' não encontrada. Colunas disponíveis: {list(out.columns)}"
            )

        if isinstance(expected, str):
            out = out[
                out[real_col].astype(str).str.strip().str.lower()
                == expected.strip().lower()
            ]
        else:
            out = out[out[real_col] == expected]

    return out


def standardize_source(cfg: SourceConfig) -> pd.DataFrame:
    df = read_csv_flex(cfg.path)
    df = apply_filters(df, cfg.filters)

    idx_col = infer_idx_column(df)
    true_col = cfg.true_col or find_existing_column(df, TRUE_COL_CANDIDATES)
    pred_col = cfg.pred_col or find_existing_column(df, PRED_COL_CANDIDATES)
    model_col = cfg.model_col or infer_model_column(df)
    raw_response_col = infer_raw_response_column(df)

    if true_col is None:
        raise KeyError(f"[{cfg.name}] Coluna de rótulo verdadeiro não encontrada.")
    if pred_col is None:
        raise KeyError(f"[{cfg.name}] Coluna de predição não encontrada.")

    if cfg.idx_prefix_filters:
        if idx_col is None:
            warnings.warn(
                f"[{cfg.name}] Não foi possível aplicar idx_prefix_filters={cfg.idx_prefix_filters} "
                f"porque não existe coluna idx detectável."
            )
        else:
            prefixes = tuple(str(p) for p in cfg.idx_prefix_filters)
            df = df[df[idx_col].astype(str).str.startswith(prefixes, na=False)].copy()

    if len(df) == 0:
        warnings.warn(f"[{cfg.name}] Nenhuma linha restou após os filtros.")
        return pd.DataFrame(
            columns=[
                "source_group",
                "source_name",
                "row_number_in_source",
                "idx",
                "y_true_raw",
                "y_pred_raw",
                "raw_response",
                "y_true",
                "y_pred",
                "excluded_from_metrics",
                "exclusion_reason",
            ]
        )

    if model_col is not None:
        model_values = (
            df[model_col]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .fillna("UNKNOWN_MODEL")
        )
        source_name_values = cfg.name + "::" + model_values
    else:
        source_name_values = pd.Series([cfg.name] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "source_group": cfg.name,
            "source_name": source_name_values.values,
            "row_number_in_source": np.arange(len(df), dtype=int),
            "idx": (
                df[idx_col].astype(str).values
                if idx_col is not None
                else pd.Series([pd.NA] * len(df), dtype="object")
            ),
            "y_true_raw": df[true_col].values,
            "y_pred_raw": df[pred_col].values,
            "raw_response": (
                df[raw_response_col].values
                if raw_response_col is not None
                else pd.Series([pd.NA] * len(df), dtype="object")
            ),
        }
    )

    out["y_true"] = out["y_true_raw"].map(lambda x: normalize_label(x))

    out["y_pred"] = out.apply(
        lambda row: normalize_label(
            row["y_pred_raw"],
            raw_response=row["raw_response"],
            source_name=row["source_group"],
            is_prediction=True,
        ),
        axis=1,
    )

    out["excluded_from_metrics"] = False
    out["exclusion_reason"] = pd.NA

    zero_mask = out["source_group"].eq("zero_shot")
    undef_mask = zero_mask & out["y_pred"].isna()

    apology_mask = undef_mask & out["raw_response"].map(looks_like_apology_or_refusal)
    other_undef_mask = undef_mask & ~apology_mask

    out.loc[apology_mask, "excluded_from_metrics"] = True
    out.loc[apology_mask, "exclusion_reason"] = "undefined_due_to_apology_or_refusal"

    out.loc[other_undef_mask, "excluded_from_metrics"] = True
    out.loc[other_undef_mask, "exclusion_reason"] = "undefined_zero_shot_prediction"

    out = out.dropna(subset=["y_true"]).reset_index(drop=True)
    return out


def resolve_missing_idx(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    dfs = {k: v.copy() for k, v in dfs.items()}

    reference_name = None
    reference_idx = None

    for name, df in dfs.items():
        non_null = df["idx"].notna().sum()
        if non_null > 0:
            reference_name = name
            reference_idx = df["idx"].astype(str).tolist()
            break

    for name, df in dfs.items():
        missing_mask = df["idx"].isna() | (df["idx"].astype(str).str.strip() == "")
        if not missing_mask.any():
            continue

        if (
            reference_idx is not None
            and ASSUME_SHARED_TEST_ORDER_WHEN_IDX_MISSING
            and len(df) == len(reference_idx)
        ):
            warnings.warn(
                f"[{name}] idx ausente. Herdando idx de '{reference_name}' pela ordem das linhas. "
                f"Isto só é válido se os arquivos tiverem o mesmo ordenamento."
            )
            df.loc[missing_mask, "idx"] = [
                reference_idx[i] for i in df.loc[missing_mask, "row_number_in_source"]
            ]
        else:
            warnings.warn(
                f"[{name}] idx ausente e não foi possível herdar de referência. "
                f"Gerando idx sintético local."
            )
            synthetic = [f"{name.upper()}_ROW_{i:06d}" for i in range(len(df))]
            df.loc[missing_mask, "idx"] = [
                synthetic[i] for i in df.loc[missing_mask, "row_number_in_source"]
            ]

        dfs[name] = df

    return dfs


# ============================================================
# Colunas de erro / confusão
# ============================================================

def add_confusion_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_correct"] = np.where(
        out["excluded_from_metrics"],
        np.nan,
        (out["y_true"] == out["y_pred"]).astype(float),
    )

    def _error_type(row: pd.Series) -> str:
        if row["excluded_from_metrics"] or pd.isna(row["y_pred"]):
            return "UNDEFINED"

        yt = row["y_true"]
        yp = row["y_pred"]

        if yt == POSITIVE_LABEL and yp == POSITIVE_LABEL:
            return "TP"
        if yt == NEGATIVE_LABEL and yp == NEGATIVE_LABEL:
            return "TN"
        if yt == NEGATIVE_LABEL and yp == POSITIVE_LABEL:
            return "FP"
        if yt == POSITIVE_LABEL and yp == NEGATIVE_LABEL:
            return "FN"
        return "UNK"

    out["error_type"] = out.apply(_error_type, axis=1)
    return out


# ============================================================
# Métricas
# ============================================================

def compute_binary_counts(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, int]:
    y_true_bin = np.array([1 if y == POSITIVE_LABEL else 0 for y in y_true], dtype=int)
    y_pred_bin = np.array([1 if y == POSITIVE_LABEL else 0 for y in y_pred], dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def metric_bundle(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    y_true_bin = np.array([1 if y == POSITIVE_LABEL else 0 for y in y_true], dtype=int)
    y_pred_bin = np.array([1 if y == POSITIVE_LABEL else 0 for y in y_pred], dtype=int)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        labels=[1, 0],
        zero_division=0,
    )

    fake_prec, real_prec = prec[0], prec[1]
    fake_rec, real_rec = rec[0], rec[1]
    fake_f1, real_f1 = f1[0], f1[1]
    fake_sup, real_sup = support[0], support[1]

    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    counts = compute_binary_counts(y_true, y_pred)

    return {
        "n": int(len(y_true)),
        "support_fake": int(fake_sup),
        "support_real": int(real_sup),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fake_precision": float(fake_prec),
        "fake_recall": float(fake_rec),
        "fake_f1": float(fake_f1),
        "real_precision": float(real_prec),
        "real_recall": float(real_rec),
        "real_f1": float(real_f1),
        "mcc": float(mcc),
        **counts,
    }


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric_fn: Callable[[Sequence[str], Sequence[str]], float],
    n_boot: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
    random_state: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y_true = np.array(list(y_true), dtype=object)
    y_pred = np.array(list(y_pred), dtype=object)
    n = len(y_true)

    if n == 0:
        return (np.nan, np.nan, np.nan)

    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        try:
            values.append(metric_fn(yt, yp))
        except Exception:
            values.append(np.nan)

    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)

    point = metric_fn(y_true, y_pred)
    low = np.quantile(arr, alpha / 2)
    high = np.quantile(arr, 1 - alpha / 2)
    return float(point), float(low), float(high)


def bootstrap_metrics_for_model(df_model: pd.DataFrame) -> pd.DataFrame:
    y_true = df_model["y_true"].tolist()
    y_pred = df_model["y_pred"].tolist()

    metric_functions = {
        "accuracy": lambda yt, yp: accuracy_score(
            [1 if y == POSITIVE_LABEL else 0 for y in yt],
            [1 if y == POSITIVE_LABEL else 0 for y in yp],
        ),
        "macro_f1": lambda yt, yp: f1_score(
            [1 if y == POSITIVE_LABEL else 0 for y in yt],
            [1 if y == POSITIVE_LABEL else 0 for y in yp],
            average="macro",
            zero_division=0,
        ),
        "fake_f1": lambda yt, yp: f1_score(
            [1 if y == POSITIVE_LABEL else 0 for y in yt],
            [1 if y == POSITIVE_LABEL else 0 for y in yp],
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "mcc": lambda yt, yp: matthews_corrcoef(
            [1 if y == POSITIVE_LABEL else 0 for y in yt],
            [1 if y == POSITIVE_LABEL else 0 for y in yp],
        ),
    }

    rows = []
    for metric_name, metric_fn in metric_functions.items():
        point, low, high = bootstrap_ci(y_true, y_pred, metric_fn)
        rows.append(
            {
                "metric": metric_name,
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "alpha": BOOTSTRAP_ALPHA,
                "n_bootstrap": BOOTSTRAP_N,
            }
        )

    return pd.DataFrame(rows)


def evaluate_all_models(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    ci_frames = []

    for model_name, g in df_all.groupby("source_name", sort=False):
        g_valid = g[~g["excluded_from_metrics"] & g["y_pred"].notna()].copy()

        n_total = len(g)
        n_used_metrics = len(g_valid)
        n_excluded = int(g["excluded_from_metrics"].sum())

        if n_used_metrics == 0:
            metrics = {
                "source_name": model_name,
                "n_total": n_total,
                "n_used_metrics": 0,
                "n_excluded_undefined": n_excluded,
                "support_fake": np.nan,
                "support_real": np.nan,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "fake_precision": np.nan,
                "fake_recall": np.nan,
                "fake_f1": np.nan,
                "real_precision": np.nan,
                "real_recall": np.nan,
                "real_f1": np.nan,
                "mcc": np.nan,
                "TP": np.nan,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
            }
            summary_rows.append(metrics)
            continue

        metrics = metric_bundle(g_valid["y_true"], g_valid["y_pred"])
        metrics["source_name"] = model_name
        metrics["n_total"] = n_total
        metrics["n_used_metrics"] = n_used_metrics
        metrics["n_excluded_undefined"] = n_excluded
        summary_rows.append(metrics)

        ci = bootstrap_metrics_for_model(g_valid)
        ci.insert(0, "source_name", model_name)
        ci_frames.append(ci)

    summary_df = pd.DataFrame(summary_rows)
    ci_df = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    return summary_df, ci_df


# ============================================================
# Dados pareados entre modelos
# ============================================================

def prepare_pairwise_correctness(df_all: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df_all[["idx", "source_name", "is_correct"]]
        .drop_duplicates(["idx", "source_name"])
        .pivot(index="idx", columns="source_name", values="is_correct")
        .sort_index()
    )
    return pivot


def prepare_pairwise_predictions(df_all: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df_all[["idx", "source_name", "y_pred"]]
        .drop_duplicates(["idx", "source_name"])
        .pivot(index="idx", columns="source_name", values="y_pred")
        .sort_index()
    )
    return pivot


# ============================================================
# McNemar
# ============================================================

def mcnemar_test_from_correctness(
    a_correct: Sequence[int],
    b_correct: Sequence[int],
) -> Dict[str, float]:
    a = np.array(a_correct, dtype=float)
    b = np.array(b_correct, dtype=float)

    valid = ~np.isnan(a) & ~np.isnan(b)
    a = a[valid].astype(int)
    b = b[valid].astype(int)

    both_correct = int(np.sum((a == 1) & (b == 1)))
    a_correct_b_wrong = int(np.sum((a == 1) & (b == 0)))
    a_wrong_b_correct = int(np.sum((a == 0) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        p_exact = 1.0
        chi2_cc = 0.0
        chi2_no_cc = 0.0
    else:
        p_exact = float(
            binomtest(
                a_correct_b_wrong,
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        chi2_cc = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / discordant
        chi2_no_cc = (a_correct_b_wrong - a_wrong_b_correct) ** 2 / discordant

    return {
        "n_common": int(len(a)),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "mcnemar_pvalue_exact": float(p_exact),
        "mcnemar_chi2_cc": float(chi2_cc),
        "mcnemar_chi2_no_cc": float(chi2_no_cc),
    }


def pairwise_model_comparisons(df_all: pd.DataFrame) -> pd.DataFrame:
    corr_wide = prepare_pairwise_correctness(df_all)
    rows = []

    for a_name, b_name in itertools.combinations(corr_wide.columns.tolist(), 2):
        test_res = mcnemar_test_from_correctness(
            corr_wide[a_name].values,
            corr_wide[b_name].values,
        )
        rows.append(
            {
                "model_a": a_name,
                "model_b": b_name,
                **test_res,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Pearson
# ============================================================

def pairwise_pearson_on_correctness(df_all: pd.DataFrame) -> pd.DataFrame:
    corr_wide = prepare_pairwise_correctness(df_all)
    rows = []

    for a_name, b_name in itertools.combinations(corr_wide.columns.tolist(), 2):
        sub = corr_wide[[a_name, b_name]].dropna()

        if len(sub) < 2:
            r, pval = np.nan, np.nan
        elif sub[a_name].nunique() == 1 or sub[b_name].nunique() == 1:
            r, pval = np.nan, np.nan
        else:
            r, pval = pearsonr(sub[a_name].astype(float), sub[b_name].astype(float))

        rows.append(
            {
                "model_a": a_name,
                "model_b": b_name,
                "n_common": int(len(sub)),
                "pearson_r_correctness": float(r) if pd.notna(r) else np.nan,
                "pearson_pvalue": float(pval) if pd.notna(pval) else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Markov-like
# ============================================================

def markov_like_transitions(
    df_all: pd.DataFrame,
    ordered_models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    corr_wide = prepare_pairwise_correctness(df_all)

    if ordered_models is None:
        ordered_models = df_all["source_name"].drop_duplicates().tolist()
    else:
        ordered_models = list(ordered_models)

    existing = [m for m in ordered_models if m in corr_wide.columns]
    rows = []

    for prev_model, next_model in zip(existing[:-1], existing[1:]):
        sub = corr_wide[[prev_model, next_model]].dropna()
        if len(sub) == 0:
            continue

        prev_c = sub[prev_model].astype(int).values
        next_c = sub[next_model].astype(int).values

        cc = int(np.sum((prev_c == 1) & (next_c == 1)))
        ci = int(np.sum((prev_c == 1) & (next_c == 0)))
        ic = int(np.sum((prev_c == 0) & (next_c == 1)))
        ii = int(np.sum((prev_c == 0) & (next_c == 0)))

        total_prev_c = cc + ci
        total_prev_i = ic + ii

        rows.extend(
            [
                {
                    "from_model": prev_model,
                    "to_model": next_model,
                    "from_state": "C",
                    "to_state": "C",
                    "count": cc,
                    "probability": cc / total_prev_c if total_prev_c > 0 else np.nan,
                    "n_common": int(len(sub)),
                },
                {
                    "from_model": prev_model,
                    "to_model": next_model,
                    "from_state": "C",
                    "to_state": "I",
                    "count": ci,
                    "probability": ci / total_prev_c if total_prev_c > 0 else np.nan,
                    "n_common": int(len(sub)),
                },
                {
                    "from_model": prev_model,
                    "to_model": next_model,
                    "from_state": "I",
                    "to_state": "C",
                    "count": ic,
                    "probability": ic / total_prev_i if total_prev_i > 0 else np.nan,
                    "n_common": int(len(sub)),
                },
                {
                    "from_model": prev_model,
                    "to_model": next_model,
                    "from_state": "I",
                    "to_state": "I",
                    "count": ii,
                    "probability": ii / total_prev_i if total_prev_i > 0 else np.nan,
                    "n_common": int(len(sub)),
                },
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# Distribuições acumuladas
# ============================================================

def cumulative_distribution_tables(
    df_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_wide = prepare_pairwise_correctness(df_all)
    pred_wide = prepare_pairwise_predictions(df_all)

    correct_count = corr_wide.sum(axis=1, skipna=True)
    available_models = corr_wide.notna().sum(axis=1)
    acc_frac = correct_count / available_models.replace(0, np.nan)

    by_sample = pd.DataFrame(
        {
            "idx": corr_wide.index,
            "n_models_available": available_models.values,
            "n_models_correct": correct_count.values,
            "fraction_models_correct": acc_frac.values,
        }
    )

    value_counts = by_sample["n_models_correct"].value_counts().sort_index()
    ecdf = value_counts.cumsum() / value_counts.sum()

    cdf_table = pd.DataFrame(
        {
            "n_models_correct": value_counts.index,
            "count": value_counts.values,
            "cdf": ecdf.values,
        }
    )

    pred_fake_count = pred_wide.apply(lambda row: np.sum(row == POSITIVE_LABEL), axis=1)
    pred_fake_counts = pred_fake_count.value_counts().sort_index()
    pred_fake_ecdf = pred_fake_counts.cumsum() / pred_fake_counts.sum()

    fake_vote_cdf = pd.DataFrame(
        {
            "n_models_predicting_fake": pred_fake_counts.index,
            "count": pred_fake_counts.values,
            "cdf": pred_fake_ecdf.values,
        }
    )

    return by_sample, cdf_table.merge(
        fake_vote_cdf,
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("_correct", "_fake_votes"),
    )


# ============================================================
# Gráficos
# ============================================================

def plot_ecdf_correct_models(by_sample: pd.DataFrame, outpath: Path) -> None:
    vc = by_sample["n_models_correct"].value_counts().sort_index()
    x = vc.index.to_numpy()
    y = vc.cumsum().to_numpy() / vc.sum()

    plt.figure(figsize=(8, 5))
    plt.step(x, y, where="post")
    plt.xlabel("Número de modelos corretos por amostra")
    plt.ylabel("CDF empírica")
    plt.title("Distribuição acumulada do número de modelos corretos")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_model_accuracy_with_ci(
    summary_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    outpath: Path,
) -> None:
    acc_ci = ci_df[ci_df["metric"] == "accuracy"].copy()

    merged = summary_df.merge(
        acc_ci[["source_name", "ci_low", "ci_high"]],
        on="source_name",
        how="left",
    )

    merged = merged.sort_values("accuracy", ascending=False).reset_index(drop=True)

    x = np.arange(len(merged))
    y = merged["accuracy"].to_numpy()
    yerr_lower = y - merged["ci_low"].to_numpy()
    yerr_upper = merged["ci_high"].to_numpy() - y

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt="none", capsize=4)
    plt.xticks(x, merged["source_name"], rotation=35, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy com IC bootstrap")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ============================================================
# Consistência do ground truth
# ============================================================

def validate_ground_truth_consistency(df_all: pd.DataFrame) -> pd.DataFrame:
    gt_check = (
        df_all.groupby("idx")["y_true"]
        .nunique(dropna=True)
        .reset_index(name="n_unique_true_labels")
    )
    return gt_check[gt_check["n_unique_true_labels"] > 1].copy()


# ============================================================
# Alinhamento de idx entre fontes
# ============================================================

def _clean_idx_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out[~out.isna()]
    out = out[(out != "") & (out.str.lower() != "nan") & (out.str.lower() != "<na>")]
    return out


def extract_reference_idx_set(
    dfs: Dict[str, pd.DataFrame],
    reference_source_key: str = "classic_models",
) -> set[str]:
    if reference_source_key not in dfs:
        raise KeyError(
            f"Fonte de referência '{reference_source_key}' não encontrada em dfs. "
            f"Disponíveis: {list(dfs.keys())}"
        )

    ref_df = dfs[reference_source_key].copy()
    if "idx" not in ref_df.columns:
        raise KeyError(
            f"A fonte de referência '{reference_source_key}' não possui coluna 'idx'."
        )

    ref_idx = _clean_idx_series(ref_df["idx"])
    ref_idx_set = set(ref_idx.unique())

    if len(ref_idx_set) == 0:
        raise ValueError(
            f"A fonte de referência '{reference_source_key}' não possui idx válidos."
        )

    return ref_idx_set


def align_sources_to_reference_idx(
    dfs: Dict[str, pd.DataFrame],
    reference_source_key: str = "classic_models",
    target_source_keys: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    dfs = {k: v.copy() for k, v in dfs.items()}
    reference_idx_set = extract_reference_idx_set(
        dfs, reference_source_key=reference_source_key
    )

    if target_source_keys is None:
        target_source_keys = [k for k in dfs.keys() if k != reference_source_key]

    for source_key in target_source_keys:
        if source_key not in dfs:
            warnings.warn(
                f"[align_sources_to_reference_idx] Fonte-alvo '{source_key}' não encontrada. Ignorando."
            )
            continue

        df = dfs[source_key].copy()
        if "idx" not in df.columns:
            warnings.warn(
                f"[{source_key}] Não possui coluna 'idx'. Não foi possível alinhar com a referência."
            )
            dfs[source_key] = df
            continue

        before_rows = len(df)
        before_unique_idx = _clean_idx_series(df["idx"]).nunique()

        mask = df["idx"].astype(str).str.strip().isin(reference_idx_set)
        df = df[mask].copy().reset_index(drop=True)

        after_rows = len(df)
        after_unique_idx = _clean_idx_series(df["idx"]).nunique()

        warnings.warn(
            f"[{source_key}] Alinhado ao conjunto de idx de '{reference_source_key}'. "
            f"Linhas: {before_rows} -> {after_rows}. "
            f"Idx únicos: {before_unique_idx} -> {after_unique_idx}. "
            f"Idx de referência: {len(reference_idx_set)}."
        )

        dfs[source_key] = df

    return dfs


# ============================================================
# Relatórios auxiliares
# ============================================================

def make_source_cfgs(phase_name: str) -> Dict[str, SourceConfig]:
    phase_inputs = PHASE_INPUTS[phase_name]
    return {
        name: SourceConfig(
            name=name,
            path=Path(conf["path"]),
            filters=conf["filters"],
            true_col=conf.get("true_col"),
            pred_col=conf.get("pred_col"),
            model_col=conf.get("model_col"),
            idx_prefix_filters=conf.get("idx_prefix_filters"),
        )
        for name, conf in phase_inputs.items()
    }


def make_detailed_model_report(
    summary_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    phase_name: str,
) -> pd.DataFrame:
    out = summary_df.copy()
    out["split"] = phase_name

    if ci_df.empty:
        return out

    for metric in ci_df["metric"].dropna().unique():
        sub = ci_df[ci_df["metric"] == metric][["source_name", "ci_low", "ci_high"]].copy()
        sub = sub.rename(
            columns={
                "ci_low": f"{metric}_ci_low",
                "ci_high": f"{metric}_ci_high",
            }
        )
        out = out.merge(sub, on="source_name", how="left")

    return out


def normalize_cross_family_mcnemar(
    df: pd.DataFrame,
    left_prefix: str = "classic_models::",
    right_prefix: str = "zero_shot::",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows = []
    for _, row in df.iterrows():
        a = row["model_a"]
        b = row["model_b"]

        cond_direct = str(a).startswith(left_prefix) and str(b).startswith(right_prefix)
        cond_swap = str(a).startswith(right_prefix) and str(b).startswith(left_prefix)

        if not (cond_direct or cond_swap):
            continue

        row_dict = row.to_dict()

        if cond_swap:
            row_dict["model_a"], row_dict["model_b"] = row_dict["model_b"], row_dict["model_a"]
            row_dict["a_correct_b_wrong"], row_dict["a_wrong_b_correct"] = (
                row_dict["a_wrong_b_correct"],
                row_dict["a_correct_b_wrong"],
            )

        row_dict["comparison_family"] = "classic_vs_zero_shot"
        rows.append(row_dict)

    return pd.DataFrame(rows)


def normalize_cross_family_symmetric(
    df: pd.DataFrame,
    left_prefix: str = "classic_models::",
    right_prefix: str = "zero_shot::",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows = []
    for _, row in df.iterrows():
        a = row["model_a"]
        b = row["model_b"]

        cond_direct = str(a).startswith(left_prefix) and str(b).startswith(right_prefix)
        cond_swap = str(a).startswith(right_prefix) and str(b).startswith(left_prefix)

        if not (cond_direct or cond_swap):
            continue

        row_dict = row.to_dict()

        if cond_swap:
            row_dict["model_a"], row_dict["model_b"] = row_dict["model_b"], row_dict["model_a"]

        row_dict["comparison_family"] = "classic_vs_zero_shot"
        rows.append(row_dict)

    return pd.DataFrame(rows)


# ============================================================
# Execução por fase
# ============================================================

def run_phase_analysis(phase_name: str, base_output_dir: Path) -> Dict[str, pd.DataFrame]:
    phase_output_dir = base_output_dir / phase_name
    phase_output_dir.mkdir(parents=True, exist_ok=True)

    source_cfgs = make_source_cfgs(phase_name)

    standardized = {}
    for name, cfg in source_cfgs.items():
        standardized[name] = standardize_source(cfg)

    standardized = resolve_missing_idx(standardized)

    standardized = align_sources_to_reference_idx(
        standardized,
        reference_source_key="classic_models",
        target_source_keys=["zero_shot"],
    )

    df_all = pd.concat(standardized.values(), ignore_index=True)
    df_all["idx"] = df_all["idx"].astype(str)

    df_all = df_all.drop_duplicates(
        ["source_name", "idx", "y_true", "y_pred"]
    ).reset_index(drop=True)

    df_all = add_confusion_columns(df_all)

    undefined_report = df_all[df_all["excluded_from_metrics"]].copy()
    undefined_report.to_csv(
        phase_output_dir / f"{phase_name}_zero_shot_undefined_exclusions.csv",
        index=False,
    )

    inconsistent_gt = validate_ground_truth_consistency(df_all)
    inconsistent_gt.to_csv(
        phase_output_dir / f"{phase_name}_ground_truth_inconsistencies.csv",
        index=False,
    )

    df_all.to_csv(
        phase_output_dir / f"{phase_name}_standardized_predictions_per_sample.csv",
        index=False,
    )

    summary_df, ci_df = evaluate_all_models(df_all)
    summary_df = summary_df.sort_values(
        ["macro_f1", "fake_f1", "mcc"],
        ascending=False,
    ).reset_index(drop=True)

    detailed_df = make_detailed_model_report(summary_df, ci_df, phase_name)

    summary_df.to_csv(
        phase_output_dir / f"{phase_name}_metrics_summary_by_model.csv",
        index=False,
    )
    ci_df.to_csv(
        phase_output_dir / f"{phase_name}_bootstrap_confidence_intervals.csv",
        index=False,
    )
    detailed_df.to_csv(
        phase_output_dir / f"{phase_name}_detailed_model_report.csv",
        index=False,
    )

    mcnemar_df = pairwise_model_comparisons(df_all)
    mcnemar_df.to_csv(
        phase_output_dir / f"{phase_name}_pairwise_mcnemar_results.csv",
        index=False,
    )

    pearson_df = pairwise_pearson_on_correctness(df_all)
    pearson_df.to_csv(
        phase_output_dir / f"{phase_name}_pairwise_pearson_correctness.csv",
        index=False,
    )

    classical_vs_zero_shot_mcnemar = normalize_cross_family_mcnemar(mcnemar_df)
    classical_vs_zero_shot_mcnemar.to_csv(
        phase_output_dir / f"{phase_name}_classical_vs_zero_shot_mcnemar.csv",
        index=False,
    )

    classical_vs_zero_shot_pearson = normalize_cross_family_symmetric(pearson_df)
    classical_vs_zero_shot_pearson.to_csv(
        phase_output_dir / f"{phase_name}_classical_vs_zero_shot_pearson.csv",
        index=False,
    )

    markov_df = markov_like_transitions(df_all, ordered_models=DEFAULT_MODEL_ORDER)
    markov_df.to_csv(
        phase_output_dir / f"{phase_name}_markov_like_transitions_between_models.csv",
        index=False,
    )

    by_sample_df, cdf_df = cumulative_distribution_tables(df_all)
    by_sample_df.to_csv(
        phase_output_dir / f"{phase_name}_sample_level_agreement_distribution.csv",
        index=False,
    )
    cdf_df.to_csv(
        phase_output_dir / f"{phase_name}_cumulative_distribution_tables.csv",
        index=False,
    )

    plot_ecdf_correct_models(
        by_sample_df,
        phase_output_dir / f"{phase_name}_ecdf_num_correct_models.png",
    )
    plot_model_accuracy_with_ci(
        summary_df,
        ci_df,
        phase_output_dir / f"{phase_name}_accuracy_with_bootstrap_ci.png",
    )

    return {
        "summary_df": summary_df,
        "ci_df": ci_df,
        "detailed_df": detailed_df,
        "mcnemar_df": mcnemar_df,
        "pearson_df": pearson_df,
        "classical_vs_zero_shot_mcnemar": classical_vs_zero_shot_mcnemar,
        "classical_vs_zero_shot_pearson": classical_vs_zero_shot_pearson,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise estatística de resultados de modelos de desinformação."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"results\analysis_outputs_models",
        help="Diretório de saída para CSVs e figuras.",
    )
    parser.add_argument(
        "--disable-shared-order-idx",
        action="store_true",
        help="Desabilita a herança de idx via ordem das linhas quando o idx estiver ausente.",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    global ASSUME_SHARED_TEST_ORDER_WHEN_IDX_MISSING

    args = parse_args()
    if args.disable_shared_order_idx:
        ASSUME_SHARED_TEST_ORDER_WHEN_IDX_MISSING = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_phase_reports = []

    for phase_name in ["val", "test"]:
        print("\n" + "=" * 100)
        print(f"PROCESSANDO SPLIT: {phase_name.upper()}")
        print("=" * 100)

        reports = run_phase_analysis(phase_name, output_dir)

        detailed_df = reports["detailed_df"].copy()
        all_phase_reports.append(detailed_df)

        print(f"\n=== Resumo das métricas por modelo [{phase_name}] ===")
        cols = [
            "source_name",
            "n_total",
            "n_used_metrics",
            "n_excluded_undefined",
            "accuracy",
            "macro_f1",
            "fake_f1",
            "real_f1",
            "mcc",
            "TP",
            "TN",
            "FP",
            "FN",
        ]
        print(detailed_df[cols].to_string(index=False))

    if all_phase_reports:
        combined_detailed_df = pd.concat(all_phase_reports, ignore_index=True)
        combined_detailed_df.to_csv(
            output_dir / "combined_detailed_model_report_all_splits.csv",
            index=False,
        )

    print("\nArquivos salvos em:", output_dir)
    print("Subpastas geradas:")
    print(" - val/")
    print(" - test/")
    print("\nArquivos consolidados:")
    print(" - combined_detailed_model_report_all_splits.csv")


if __name__ == "__main__":
    main()