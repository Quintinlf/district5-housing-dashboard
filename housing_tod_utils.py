"""
Utilities for integrating District 5 housing trend data with SB 79 transit-stop outputs.

This module intentionally mirrors the data conventions in the existing
city_council workflow so notebook analyses can stay reproducible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import requests


CENSUS_ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"

DISTRICT_5_ZIPS: Dict[str, str] = {
    "90210": "Beverly Hills / Bel Air",
    "90212": "Beverly Hills / Century City",
    "90035": "Beverly Grove / Beverlywood",
    "90048": "Fairfax / West Hollywood adj.",
    "90064": "Rancho Park / West LA",
    "90024": "Westwood / UCLA area",
}

# Approximate ZIP centroids used as a proxy for district-level stop tagging.
DISTRICT_5_ZIP_CENTROIDS: Dict[str, tuple[float, float]] = {
    "90210": (34.1030, -118.4105),
    "90212": (34.0630, -118.4020),
    "90035": (34.0525, -118.3850),
    "90048": (34.0735, -118.3720),
    "90064": (34.0355, -118.4280),
    "90024": (34.0622, -118.4437),
}

CENSUS_VARS: Dict[str, str] = {
    "median_rent": "B25031_001E",
    "median_income": "B19013_001E",
}


@dataclass(frozen=True)
class HousingSummary:
    """Compact District 5 housing trend summary."""

    start_year: int
    end_year: int
    rent_start: float
    rent_end: float
    income_start: float
    income_end: float
    ratio_start: float
    ratio_end: float
    rent_growth_pct: float
    income_growth_pct: float
    ratio_change_pp: float


def _read_acs_zip_batch(
    year: int,
    variables: Dict[str, str],
    zip_codes: Sequence[str],
    api_key: str = "",
) -> pd.DataFrame:
    base_url = CENSUS_ACS_URL.format(year=year)
    var_codes = list(variables.values())

    params = {
        "get": "NAME," + ",".join(var_codes),
        "for": f"zip code tabulation area:{','.join(zip_codes)}",
    }
    if api_key:
        params["key"] = api_key

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if len(payload) < 2:
        return pd.DataFrame()

    headers, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=headers)
    rename_map = {"zip code tabulation area": "zip_code"}
    for friendly, code in variables.items():
        rename_map[code] = friendly

    df = df.rename(columns=rename_map)
    df["year"] = year

    for col in variables.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-666666666, np.nan)

    return df[["year", "zip_code", *variables.keys()]]


def _read_acs_zip_single(
    year: int,
    variables: Dict[str, str],
    zip_code: str,
    api_key: str = "",
) -> pd.DataFrame:
    base_url = CENSUS_ACS_URL.format(year=year)
    var_codes = list(variables.values())

    params = {
        "get": "NAME," + ",".join(var_codes),
        "for": f"zip code tabulation area:{zip_code}",
    }
    if year < 2020:
        params["in"] = "state:06"
    if api_key:
        params["key"] = api_key

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if len(payload) < 2:
        return pd.DataFrame()

    headers, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=headers)

    rename_map = {"zip code tabulation area": "zip_code"}
    for friendly, code in variables.items():
        rename_map[code] = friendly

    df = df.rename(columns=rename_map)
    df["year"] = year

    for col in variables.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-666666666, np.nan)

    return df[["year", "zip_code", *variables.keys()]]


def fetch_d5_housing_data(
    years: Iterable[int],
    zip_codes: Sequence[str] | None = None,
    api_key: str = "",
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch District 5 ZIP-level rent/income ACS data for all provided years.

    This function attempts a batch ZIP query first, then falls back to one ZIP
    at a time for years where ACS API behavior differs.
    """
    if zip_codes is None:
        zip_codes = list(DISTRICT_5_ZIPS.keys())

    frames: list[pd.DataFrame] = []

    for year in sorted(set(years)):
        try:
            yearly = _read_acs_zip_batch(year, CENSUS_VARS, zip_codes, api_key=api_key)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 400:
                raise

            parts: list[pd.DataFrame] = []
            for zip_code in zip_codes:
                try:
                    df_zip = _read_acs_zip_single(year, CENSUS_VARS, zip_code, api_key=api_key)
                    if not df_zip.empty:
                        parts.append(df_zip)
                except requests.HTTPError:
                    continue
                time.sleep(sleep_seconds)

            yearly = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if not yearly.empty:
            frames.append(yearly)

        time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(columns=["year", "zip_code", "median_rent", "median_income"])

    return pd.concat(frames, ignore_index=True).sort_values(["year", "zip_code"]).reset_index(drop=True)


def aggregate_d5_housing(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ZIP-level housing data to District 5 annual mean values."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=["year", "median_rent", "median_income", "monthly_income", "rent_to_income_ratio"]
        )

    grouped = (
        raw_df.groupby("year", as_index=False)[["median_rent", "median_income"]]
        .mean()
        .sort_values("year")
        .reset_index(drop=True)
    )

    grouped["monthly_income"] = grouped["median_income"] / 12.0
    grouped["rent_to_income_ratio"] = (grouped["median_rent"] / grouped["monthly_income"]) * 100.0

    return grouped


def summarize_housing_trends(d5_df: pd.DataFrame) -> HousingSummary:
    """Compute summary trend metrics from District 5 annual aggregate data."""
    if d5_df.empty:
        raise ValueError("District 5 housing DataFrame is empty.")

    first = d5_df.iloc[0]
    last = d5_df.iloc[-1]

    rent_growth = ((last["median_rent"] / first["median_rent"]) - 1.0) * 100.0
    income_growth = ((last["median_income"] / first["median_income"]) - 1.0) * 100.0

    return HousingSummary(
        start_year=int(first["year"]),
        end_year=int(last["year"]),
        rent_start=float(first["median_rent"]),
        rent_end=float(last["median_rent"]),
        income_start=float(first["median_income"]),
        income_end=float(last["median_income"]),
        ratio_start=float(first["rent_to_income_ratio"]),
        ratio_end=float(last["rent_to_income_ratio"]),
        rent_growth_pct=float(rent_growth),
        income_growth_pct=float(income_growth),
        ratio_change_pp=float(last["rent_to_income_ratio"] - first["rent_to_income_ratio"]),
    )


def _haversine_miles(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in miles from one point to many points."""
    radius_miles = 3958.756

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))

    return radius_miles * c


def tag_d5_proxy_stops(
    stop_df: pd.DataFrame,
    zip_centroids: Dict[str, tuple[float, float]] | None = None,
    max_miles: float = 2.5,
) -> pd.DataFrame:
    """
    Tag each GTFS stop as inside/outside a District 5 proxy geography.

    The proxy is defined as being within max_miles of at least one selected
    District 5 ZIP centroid.
    """
    if stop_df.empty:
        return stop_df.copy()

    if zip_centroids is None:
        zip_centroids = DISTRICT_5_ZIP_CENTROIDS

    out = stop_df.copy()
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")

    all_distances = []
    zip_labels = []

    lats = out["stop_lat"].to_numpy(dtype=float)
    lons = out["stop_lon"].to_numpy(dtype=float)

    for zip_code, (zip_lat, zip_lon) in zip_centroids.items():
        dist = _haversine_miles(zip_lat, zip_lon, lats, lons)
        all_distances.append(dist)
        zip_labels.append(zip_code)

    distance_matrix = np.vstack(all_distances)
    min_idx = np.argmin(distance_matrix, axis=0)
    min_dist = np.min(distance_matrix, axis=0)

    out["nearest_d5_zip"] = [zip_labels[i] for i in min_idx]
    out["distance_to_d5_zip_miles"] = min_dist
    out["is_d5_proxy_stop"] = out["distance_to_d5_zip_miles"] <= max_miles

    return out


def summarize_d5_proxy_tiers(tagged_stops_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize tier counts for stops tagged inside District 5 proxy area."""
    if tagged_stops_df.empty:
        return pd.DataFrame(columns=["prelim_sb79_tier", "stop_count"])

    scoped = tagged_stops_df[tagged_stops_df["is_d5_proxy_stop"]].copy()

    summary = (
        scoped.groupby("prelim_sb79_tier", as_index=False)
        .agg(stop_count=("stop_id", "count"))
        .sort_values("stop_count", ascending=False)
        .reset_index(drop=True)
    )

    return summary
