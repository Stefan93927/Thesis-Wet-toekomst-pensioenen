"""data_pipeline.py — Load, clean, resample, and feature-engineer all data.

Data sources
------------
- Financial_Data_30Y_English.xlsx  : equity prices, swap rates, gold, oil, FX,
                                     DE/NL/US yields (CSV stored in single column).
- Extra_Indicators.csv             : clean Euribor_3M, VSTOXX, Yield_US_10Y.
- Agal_Indicators.csv              : VSTOXX backup, Yield_US_10Y/3M backup.
- LSEG_Clean_Global_Indices.csv    : global equity indices (SPX, DAX, etc.).
- CPI nederland.csv                : Dutch CPI, semicolon-delimited, monthly.
- DNB_RTS_Constructed.csv          : DNB rentetermijnstructuur (bootstrapped yield
                                     curve), liability returns, RTS slopes.

Outputs
-------
- z_train / z_val / z_test   : scaled 31-dim feature DataFrames (StandardScaler).
- z_*_raw                    : unscaled feature DataFrames.
- raw_train / raw_val / test : full merged monthly DataFrames.
- cpi                        : monthly CPI DataFrame.
- scaler                     : fitted StandardScaler (also saved via joblib).
- r_L_blended                : full-history Series of blended RTS liability returns.

Feature set (31 dimensions)
---------------------------
Equity momentum (12) : 1M/3M/12M log-returns for MSCI World, AEX, Stoxx50, EM.
RTS slopes      (2)  : RTS_10Y-RTS_2Y, RTS_30Y-RTS_10Y  (DNB term structure).
Volatility      (3)  : VSTOXX level, 1M change, 3M change.
Rates           (4)  : Euribor_3M, Euribor_1Y proxy (Swap_1Y), Yield_US_10Y,
                       Yield_DE_10Y.
Liability return(1)  : Liability_Return_MtM from DNB RTS (replaces swap proxy).
RTS 20Y level   (1)  : RTS_20Y -- official DNB 20Y discount rate.
Swap levels     (4)  : Swap_2Y, Swap_5Y, Swap_10Y, Swap_30Y.
Rate changes    (2)  : Delta_Swap_10Y, Delta_RTS_20Y.
Commodities     (2)  : Gold log-return, Oil log-return.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SRC_DIR  = Path(__file__).parent          # …/data/src/
_DATA_DIR = _SRC_DIR.parent               # …/data/  (project root)


@dataclass
class PipelineConfig:
    """All tunable constants in one place — never hard-coded in logic."""

    data_dir:    Path = _DATA_DIR
    scaler_path:    Path = _SRC_DIR / "scaler.joblib"
    artifacts_dir:  Path = _DATA_DIR / "processed"

    # Temporal splits (inclusive on both ends, ISO date strings)
    train_start: str = "1999-01-01"
    train_end:   str = "2015-12-31"
    val_start:   str = "2016-01-01"
    val_end:     str = "2017-12-31"
    test_start:  str = "2018-01-01"
    test_end:    str = "2025-12-31"

    # Liability modelling — mid-point of 17–20 year range
    duration: float = 18.0

    # Ordered list of 31 feature names (must match compute_features output exactly)
    feature_names: list = field(default_factory=lambda: [
        # Equity momentum (12)
        "mom_msci_1m",  "mom_msci_3m",  "mom_msci_12m",
        "mom_aex_1m",   "mom_aex_3m",   "mom_aex_12m",
        "mom_stoxx_1m", "mom_stoxx_3m", "mom_stoxx_12m",
        "mom_em_1m",    "mom_em_3m",    "mom_em_12m",
        # RTS slopes (2) — DNB rentetermijnstructuur, replaces swap slopes
        "rts_slope_10y_2y", "rts_slope_30y_10y",
        # Volatility (3)  — 3M change replaces TED Spread (full coverage from 1999)
        "vstoxx_level", "d_vstoxx_1m", "d_vstoxx_3m",
        # Rates (4)
        "euribor_3m", "euribor_1y_proxy",
        "yield_us_10y", "yield_de_10y",
        # Liability return (1) — DNB RTS MtM, replaces swap-based proxy
        "liab_return_mkt",
        # RTS 20Y level (1) — official DNB discount rate, replaces swap_20y
        "rts_20y",
        # Swap levels (4) — swap_20y removed (replaced by rts_20y above)
        "swap_2y", "swap_5y", "swap_10y", "swap_30y",
        # Rate changes (2) — d_swap_20y replaced by d_rts_20y
        "d_swap_10y", "d_rts_20y",
        # Commodities (2)
        "gold_log_ret", "oil_log_ret",
    ])


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_financial_excel(data_dir: Path) -> pd.DataFrame:
    """Load Financial_Data_30Y_English.xlsx.

    The file stores all data as comma-separated text inside a single Excel
    column.  Row 0 is the header; subsequent rows are daily observations.

    Args:
        data_dir: Directory containing the Excel file.

    Returns:
        DataFrame with parsed columns and a DatetimeIndex sorted ascending.
    """
    path = data_dir / "Financial_Data_30Y_English.xlsx"
    raw  = pd.read_excel(path, header=None, engine="openpyxl")
    csv_text = "\n".join(raw[0].astype(str).tolist())
    df = pd.read_csv(StringIO(csv_text), parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_extra_indicators(data_dir: Path) -> pd.DataFrame:
    """Load Extra_Indicators.csv (clean Euribor_3M, VSTOXX, US/DE yields).

    Args:
        data_dir: Directory containing the file.

    Returns:
        DataFrame with a DatetimeIndex sorted ascending.
    """
    path = data_dir / "Extra_Indicators.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.set_index("Date").sort_index()
    return df


def load_global_indices(data_dir: Path) -> pd.DataFrame:
    """Load LSEG_Clean_Global_Indices.csv (SPX, NDX, DAX, CAC, FTSE, N225, HSI).

    Args:
        data_dir: Directory containing the file.

    Returns:
        DataFrame with a DatetimeIndex sorted ascending.
    """
    path = data_dir / "LSEG_Clean_Global_Indices.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.set_index("Date").sort_index()
    return df


def load_agal_indicators(data_dir: Path) -> pd.DataFrame:
    """Load Agal_Indicators.csv (VSTOXX, TED Spread, US yields).

    Args:
        data_dir: Directory containing the file.

    Returns:
        DataFrame with standardised column names and a DatetimeIndex.
    """
    path = data_dir / "Agal_Indicators.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.set_index("Date").sort_index()
    df   = df.rename(columns={
        "VIX_Index europe": "VIX_EU",
        "VSTOXX_Index":     "VSTOXX_agal",
        "vix us":           "VIX_US",
        "TED_Spread":       "TED_Spread",
        "Yield_US_10Y":     "Yield_US_10Y_agal",
        "Yield_US_3M":      "Yield_US_3M_agal",
    })
    return df


def load_cpi(data_dir: Path) -> pd.DataFrame:
    """Load CPI nederland.csv (Dutch CPI, CBS Statline).

    Date format is ``YYYYMMnn``; annual aggregate rows (``YYYYJJnn``) are
    silently dropped.  Returns month-end indexed monthly data.

    Args:
        data_dir: Directory containing the file.

    Returns:
        DataFrame with columns ``cpi_annual_pct`` and ``pi_monthly``
        on a month-end DatetimeIndex.
    """
    path = data_dir / "CPI nederland.csv"
    df   = pd.read_csv(path, sep=";")

    # Keep only monthly rows (skip annual aggregates like 2025JJ00)
    monthly_mask = df["Perioden"].str.contains("MM", na=False)
    df = df[monthly_mask].copy()

    # Parse date: "YYYYMMnn" → first day of that month → snap to month-end
    df["date"] = pd.to_datetime(
        df["Perioden"].str[:4] + "-" + df["Perioden"].str[6:8] + "-01"
    )
    df = df.set_index("date").sort_index()
    df.index = df.index + pd.offsets.MonthEnd(0)

    df["cpi_annual_pct"] = pd.to_numeric(df["JaarmutatieCPI_1"], errors="coerce")
    # YoY % → monthly geometric equivalent
    df["pi_monthly"] = (1.0 + df["cpi_annual_pct"] / 100.0) ** (1.0 / 12.0) - 1.0

    return df[["cpi_annual_pct", "pi_monthly"]]


def load_rts(data_dir: Path) -> pd.DataFrame:
    """Load DNB_RTS_Constructed.csv (bootstrapped Dutch yield curve).

    Contains the DNB rentetermijnstructuur computed using the Smith-Wilson
    method, matching the official DNB methodology for pension liability
    discounting.  Key columns used downstream:

    - ``RTS_20Y``                : 20Y RTS rate (% p.a.) for liability level signal.
    - ``Delta_RTS_20Y``          : monthly change in RTS_20Y (bps driver).
    - ``Liability_Return_MtM``   : MtM liability return = -Duration * Delta_RTS_20Y/100.
    - ``Liability_Return_Blended``: 70% MtM + 30% UFR blend (environment transition).
    - ``RTS_Slope_10Y_2Y``       : 10Y - 2Y RTS slope.
    - ``RTS_Slope_30Y_10Y``      : 30Y - 10Y RTS slope.

    Args:
        data_dir: Directory containing the file.

    Returns:
        Monthly DataFrame with DatetimeIndex sorted ascending.
    """
    path = data_dir / "DNB_RTS_Constructed.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.set_index("Date").sort_index()
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df


# ---------------------------------------------------------------------------
# Merging & resampling
# ---------------------------------------------------------------------------

def _resample_to_monthly(df: pd.DataFrame, ffill_limit: int = 10) -> pd.DataFrame:
    """Forward-fill intra-month gaps then take the last observation per month.

    Args:
        df:          Daily DataFrame with DatetimeIndex.
        ffill_limit: Maximum consecutive days to forward-fill (covers
                     weekends and public holidays).

    Returns:
        Monthly DataFrame snapped to month-end (pandas ``"ME"`` offset).
    """
    df = df.ffill(limit=ffill_limit)
    return df.resample("ME").last()


def merge_all_sources(
    excel_df:  pd.DataFrame,
    extra_df:  pd.DataFrame,
    lseg_df:   pd.DataFrame,
    agal_df:   pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join all daily sources, resolve conflicts, and resample monthly.

    Resolution priority for duplicated series:
    - VSTOXX  : Extra_Indicators > Excel Volatility_EU_V2TX > Agal VSTOXX_agal
    - Yield_US_10Y : Extra_Indicators > Agal backup
    - Euribor_3M   : Extra_Indicators (Excel values are in a non-standard scale)

    Args:
        excel_df: Financial_Data_30Y data (primary source).
        extra_df: Extra_Indicators (clean rates and volatility).
        lseg_df:  LSEG global equity indices.
        agal_df:  Agal macro indicators.

    Returns:
        Monthly merged DataFrame with month-end DatetimeIndex.
    """
    # Merge daily frames on a union index
    merged = excel_df.join(extra_df,  how="outer", rsuffix="_extra")
    merged = merged.join(lseg_df,     how="outer", rsuffix="_lseg")
    merged = merged.join(agal_df,     how="outer", rsuffix="_agal_join")

    # --- Resolve VSTOXX ---
    # Extra_Indicators 'VSTOXX' is clean from 1999 (0 % NaN from 2000).
    # Fall back to Excel V2TX, then Agal.
    vstoxx_sources = []
    if "VSTOXX" in extra_df.columns:
        vstoxx_sources.append(merged["VSTOXX"])
    if "Volatility_EU_V2TX" in excel_df.columns:
        vstoxx_sources.append(merged["Volatility_EU_V2TX"])
    if "VSTOXX_agal" in merged.columns:
        vstoxx_sources.append(merged["VSTOXX_agal"])
    merged["VSTOXX_clean"] = vstoxx_sources[0].copy()
    for s in vstoxx_sources[1:]:
        merged["VSTOXX_clean"] = merged["VSTOXX_clean"].combine_first(s)

    # --- Resolve Yield_US_10Y ---
    if "Yield_US_10Y" in extra_df.columns and "Yield_US_10Y_agal" in merged.columns:
        merged["Yield_US_10Y_clean"] = (
            merged["Yield_US_10Y"].combine_first(merged["Yield_US_10Y_agal"])
        )
    elif "Yield_US_10Y" in extra_df.columns:
        merged["Yield_US_10Y_clean"] = merged["Yield_US_10Y"]
    else:
        merged["Yield_US_10Y_clean"] = merged.get(
            "Yield_US_10Y_agal", pd.Series(np.nan, index=merged.index)
        )

    # --- Euribor_3M from Extra (clean); Euribor_1Y proxy = Excel Swap_1Y ---
    merged["Euribor_3M_clean"] = merged["Euribor_3M"]   # from Extra_Indicators
    merged["Euribor_1Y_proxy"] = merged["Swap_1Y"]      # 1Y EUR swap ≈ 1Y Euribor

    monthly = _resample_to_monthly(merged)
    return monthly


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _log_ret(prices: pd.Series, lag: int) -> pd.Series:
    """Compute log return over *lag* monthly periods."""
    return np.log(prices / prices.shift(lag))


def compute_features(monthly: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Compute the 31-dimensional feature vector z_t from monthly data.

    All features are either log-returns, level changes, or levels of
    rate/volatility series.  The output column order matches
    ``cfg.feature_names`` exactly.

    Args:
        monthly: Monthly merged DataFrame from :func:`merge_all_sources`.
        cfg:     :class:`PipelineConfig` instance.

    Returns:
        DataFrame with 31 columns (``cfg.feature_names``) and the same
        DatetimeIndex as *monthly* (rows containing NaN from lags are
        retained; call :meth:`~pd.DataFrame.dropna` downstream).
    """
    z = pd.DataFrame(index=monthly.index)

    # ---- Equity momentum (12) ------------------------------------------- #
    equity_map = {
        "msci":  "Equity_World_MSCI",
        "aex":   "Equity_NL_AEX",
        "stoxx": "Equity_EU_Stoxx50",
        "em":    "Equity_Emerging_MSCI",
    }
    for short, col in equity_map.items():
        prices = monthly[col]
        z[f"mom_{short}_1m"]  = _log_ret(prices, 1)
        z[f"mom_{short}_3m"]  = _log_ret(prices, 3)
        z[f"mom_{short}_12m"] = _log_ret(prices, 12)

    # ---- RTS slopes (2) — DNB rentetermijnstructuur ------------------------- #
    z["rts_slope_10y_2y"]  = monthly["RTS_Slope_10Y_2Y"]
    z["rts_slope_30y_10y"] = monthly["RTS_Slope_30Y_10Y"]

    # ---- Volatility (3) -------------------------------------------------- #
    # 3M change replaces TED Spread (TED only available from Nov 2005).
    vstoxx = monthly["VSTOXX_clean"]
    z["vstoxx_level"] = vstoxx
    z["d_vstoxx_1m"]  = vstoxx.diff(1)
    z["d_vstoxx_3m"]  = vstoxx.diff(3)

    # ---- Rates (4) ------------------------------------------------------- #
    z["euribor_3m"]      = monthly["Euribor_3M_clean"]
    z["euribor_1y_proxy"] = monthly["Euribor_1Y_proxy"]   # Swap_1Y
    z["yield_us_10y"]    = monthly["Yield_US_10Y_clean"]
    z["yield_de_10y"]    = monthly["Yield_DE_10Y"]

    # ---- Liability return (1): DNB RTS MtM --------------------------------- #
    # Replaces the swap-based proxy; computed directly from the bootstrapped
    # RTS curve matching the DNB Smith-Wilson methodology.
    z["liab_return_mkt"] = monthly["Liability_Return_MtM"]

    # ---- RTS 20Y level (1) ---------------------------------------------- #
    z["rts_20y"] = monthly["RTS_20Y"]

    # ---- Swap levels (4) — swap_20y replaced by rts_20y above ------------ #
    z["swap_2y"]  = monthly["Swap_2Y"]
    z["swap_5y"]  = monthly["Swap_5Y"]
    z["swap_10y"] = monthly["Swap_10Y"]
    z["swap_30y"] = monthly["Swap_30Y"]

    # ---- Rate changes (2) ----------------------------------------------- #
    z["d_swap_10y"] = monthly["Swap_10Y"].diff(1)
    z["d_rts_20y"]  = monthly["Delta_RTS_20Y"]

    # ---- Commodities (2) ------------------------------------------------- #
    z["gold_log_ret"] = _log_ret(monthly["Gold_Spot"],  1)
    z["oil_log_ret"]  = _log_ret(monthly["Oil_Brent"],  1)

    # Enforce column order defined in config
    return z[cfg.feature_names]


# ---------------------------------------------------------------------------
# Splitting & scaling
# ---------------------------------------------------------------------------

def split_periods(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / validation / test by date.

    Args:
        df:  Any DataFrame with a DatetimeIndex.
        cfg: :class:`PipelineConfig` with period boundary strings.

    Returns:
        ``(train, val, test)`` DataFrames (date-inclusive slices).
    """
    train = df.loc[cfg.train_start : cfg.train_end]
    val   = df.loc[cfg.val_start   : cfg.val_end]
    test  = df.loc[cfg.test_start  : cfg.test_end]
    return train, val, test


def fit_scaler(
    z_train: pd.DataFrame,
    cfg: PipelineConfig,
) -> StandardScaler:
    """Fit a StandardScaler on training features and persist to disk.

    The scaler is fitted **only** on training data to prevent look-ahead bias.

    Args:
        z_train: Training-period feature DataFrame (must be NaN-free).
        cfg:     :class:`PipelineConfig` with ``scaler_path``.

    Returns:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    """
    scaler = StandardScaler()
    scaler.fit(z_train.values)
    cfg.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, cfg.scaler_path)
    return scaler


def apply_scaler(
    z: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Transform a feature DataFrame with a pre-fitted scaler.

    Args:
        z:      Feature DataFrame (index and columns are preserved).
        scaler: Fitted :class:`~sklearn.preprocessing.StandardScaler`.

    Returns:
        Scaled DataFrame with the same index and columns as *z*.
    """
    scaled = scaler.transform(z.values)
    return pd.DataFrame(scaled, index=z.index, columns=z.columns)


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(results: dict, cfg: PipelineConfig) -> None:
    """Save processed splits to CSV files in ``cfg.artifacts_dir``.

    Files written
    -------------
    - ``z_train.csv``     : scaled 31-dim features, training period.
    - ``z_val.csv``       : scaled 31-dim features, validation period.
    - ``z_test.csv``      : scaled 31-dim features, test period.
    - ``raw_train.csv``   : full merged monthly data, training period.
    - ``raw_val.csv``     : full merged monthly data, validation period.
    - ``raw_test.csv``    : full merged monthly data, test period.

    The index (month-end date) is written as the first column in every file.

    Args:
        results: Dictionary returned by :func:`run_pipeline`.
        cfg:     :class:`PipelineConfig` with ``artifacts_dir``.
    """
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    exports = {
        "z_train":   results["z_train"],
        "z_val":     results["z_val"],
        "z_test":    results["z_test"],
        "raw_train": results["raw_train"],
        "raw_val":   results["raw_val"],
        "raw_test":  results["raw_test"],
    }

    for name, df in exports.items():
        path = cfg.artifacts_dir / f"{name}.csv"
        df.to_csv(path, index=True)

    print(f"Artifacts saved to: {cfg.artifacts_dir}")
    for name in exports:
        path = cfg.artifacts_dir / f"{name}.csv"
        print(f"  {path.name:<18}  {exports[name].shape}")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig | None = None) -> dict:
    """Run the full data pipeline end-to-end.

    Steps
    -----
    1. Load all data sources.
    2. Merge daily data and resample to monthly frequency.
    3. Attach monthly CPI.
    4. Compute 31-dimensional feature vector z_t.
    5. Drop rows with any NaN (removes leading history and differencing lags).
    6. Split into train / validation / test periods.
    7. Fit StandardScaler on training features only; save to disk with joblib.
    8. Apply scaler to all three splits.

    Args:
        cfg: Optional :class:`PipelineConfig`; defaults to ``PipelineConfig()``.

    Returns:
        Dictionary with the following keys:

        - ``monthly_raw``        : full merged monthly DataFrame (unscaled).
        - ``z_train`` / ``z_val`` / ``z_test``         : scaled feature DFs.
        - ``z_train_raw`` / ``z_val_raw`` / ``z_test_raw`` : unscaled feature DFs.
        - ``raw_train`` / ``raw_val`` / ``raw_test``   : raw monthly splits.
        - ``cpi``                : monthly CPI DataFrame (full history).
        - ``scaler``             : fitted StandardScaler.
        - ``feature_names``      : ordered list of 31 feature names.
        - ``config``             : the :class:`PipelineConfig` used.
    """
    if cfg is None:
        cfg = PipelineConfig()

    # 1. Load --------------------------------------------------------------- #
    excel_df = load_financial_excel(cfg.data_dir)
    extra_df = load_extra_indicators(cfg.data_dir)
    lseg_df  = load_global_indices(cfg.data_dir)
    agal_df  = load_agal_indicators(cfg.data_dir)
    cpi_df   = load_cpi(cfg.data_dir)
    rts_df   = load_rts(cfg.data_dir)

    # 2. Merge daily sources → monthly -------------------------------------- #
    monthly = merge_all_sources(excel_df, extra_df, lseg_df, agal_df)

    # 3. Attach CPI and RTS (both already monthly) -------------------------- #
    monthly = monthly.join(cpi_df, how="left")
    rts_cols = [
        "RTS_20Y", "Delta_RTS_20Y", "Liability_Return_MtM",
        "Liability_Return_Blended", "RTS_Slope_10Y_2Y", "RTS_Slope_30Y_10Y",
    ]
    monthly = monthly.join(rts_df[rts_cols], how="left")

    # 4. Compute 31-dim features -------------------------------------------- #
    z_full = compute_features(monthly, cfg)

    # 5. Drop NaN rows (max lag = 12 months from momentum; leading history) -- #
    valid_idx = z_full.dropna().index
    z_full    = z_full.loc[valid_idx]
    monthly   = monthly.loc[valid_idx]

    # 6. Temporal split ------------------------------------------------------ #
    z_train_raw, z_val_raw, z_test_raw = split_periods(z_full,   cfg)
    raw_train,   raw_val,   raw_test   = split_periods(monthly,  cfg)

    # 7 & 8. Fit scaler on training data; apply to all splits --------------- #
    scaler  = fit_scaler(z_train_raw, cfg)
    z_train = apply_scaler(z_train_raw, scaler)
    z_val   = apply_scaler(z_val_raw,   scaler)
    z_test  = apply_scaler(z_test_raw,  scaler)

    # Extract blended RTS liability return series for each split
    # (used by environment to replace swap-based r_L computation)
    r_L_blended = monthly["Liability_Return_Blended"].fillna(0.0)

    return {
        "monthly_raw":   monthly,
        "z_train":       z_train,
        "z_val":         z_val,
        "z_test":        z_test,
        "z_train_raw":   z_train_raw,
        "z_val_raw":     z_val_raw,
        "z_test_raw":    z_test_raw,
        "raw_train":     raw_train,
        "raw_val":       raw_val,
        "raw_test":      raw_test,
        "cpi":           cpi_df,
        "scaler":        scaler,
        "feature_names": cfg.feature_names,
        "config":        cfg,
        "r_L_blended":   r_L_blended,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("Wtp DRL Pension Fund — Data Pipeline")
    print("=" * 64)

    cfg     = PipelineConfig()
    results = run_pipeline(cfg)

    save_artifacts(results, cfg)
    print(f"Scaler saved  : {cfg.scaler_path}\n")

    # Feature splits
    header = f"{'Dataset':<18} {'Shape':>12}  {'Start':>12}  {'End':>12}"
    print(header)
    print("-" * len(header))
    for key in ("z_train", "z_val", "z_test"):
        df = results[key]
        print(
            f"{key:<18} {str(df.shape):>12}  "
            f"{str(df.index[0].date()):>12}  "
            f"{str(df.index[-1].date()):>12}"
        )

    print()

    # Raw monthly splits
    print(header)
    print("-" * len(header))
    for key in ("raw_train", "raw_val", "raw_test"):
        df = results[key]
        print(
            f"{key:<18} {str(df.shape):>12}  "
            f"{str(df.index[0].date()):>12}  "
            f"{str(df.index[-1].date()):>12}"
        )

    print()
    cpi = results["cpi"]
    print(
        f"CPI monthly        : {cpi.shape}  "
        f"{cpi.index[0].date()} -> {cpi.index[-1].date()}"
    )

    # Feature names
    print("\nFeature names (31):")
    for i, name in enumerate(results["feature_names"], 1):
        print(f"  {i:2d}. {name}")

    # Scaler sanity check
    scaler  = results["scaler"]
    z_tr    = results["z_train"]
    print(f"\nScaler statistics (fitted on z_train_raw):")
    print(f"  mean  range : [{scaler.mean_.min():.4f},  {scaler.mean_.max():.4f}]")
    print(f"  scale range : [{scaler.scale_.min():.4f},  {scaler.scale_.max():.4f}]")
    print(f"\nScaled z_train:")
    print(f"  column means  - min {z_tr.values.mean(axis=0).min():.6f},"
          f" max {z_tr.values.mean(axis=0).max():.6f}  (all should be ~0)")
    print(f"  column stds   - min {z_tr.values.std(axis=0).min():.6f},"
          f" max {z_tr.values.std(axis=0).max():.6f}  (all should be ~1)")

    # NaN check
    print("\nNaN counts in scaled splits:")
    for key in ("z_train", "z_val", "z_test"):
        n = results[key].isna().sum().sum()
        status = "OK" if n == 0 else f"WARNING: {n} NaN(s)"
        print(f"  {key:<12}: {status}")

    print("\nDone.")
    sys.exit(0)
