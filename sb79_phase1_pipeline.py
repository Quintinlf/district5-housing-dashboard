"""
SB 79 Phase 1 Pipeline
======================

This script starts implementation of the SB 79 TOD workflow by:
1. Downloading the latest LA Metro GTFS schedule feed
2. Building stop-level service intensity metrics from GTFS tables
3. Assigning preliminary SB 79 transit tiers using transparent heuristics
4. Exporting clean CSV outputs for downstream parcel/zoning spatial joins

Note
----
These tier assignments are intentionally conservative and should be validated
against final SCAG/City Planning stop designations before publication.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
import requests


BUS_GTFS_URL = "https://gitlab.com/LACMTA/gtfs_bus/-/raw/master/gtfs_bus.zip"
RAIL_GTFS_URL = "https://gitlab.com/LACMTA/gtfs_rail/-/raw/master/gtfs_rail.zip"

# GTFS route_type values from the official GTFS reference.
ROUTE_TYPE_TRAM = 0
ROUTE_TYPE_SUBWAY = 1
ROUTE_TYPE_RAIL = 2
ROUTE_TYPE_BUS = 3


@dataclass(frozen=True)
class Thresholds:
    """Service thresholds used to assign preliminary SB 79 tiers."""

    tier1_daily_trips: int = 72
    tier2_daily_trips: int = 48


@dataclass(frozen=True)
class Paths:
    """Filesystem paths for raw and processed data outputs."""

    raw_dir: Path
    interim_dir: Path

    @property
    def bus_gtfs_zip(self) -> Path:
        return self.raw_dir / "la_metro_gtfs_bus.zip"

    @property
    def rail_gtfs_zip(self) -> Path:
        return self.raw_dir / "la_metro_gtfs_rail.zip"

    @property
    def stop_summary_csv(self) -> Path:
        return self.interim_dir / "sb79_stop_service_summary.csv"

    @property
    def tier_summary_csv(self) -> Path:
        return self.interim_dir / "sb79_tier_counts.csv"


def ensure_dirs(paths: Paths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)


def download_gtfs(
    gtfs_url: str,
    output_path: Path,
    timeout: int = 60,
    allow_insecure_fallback: bool = False,
) -> None:
    """
    Download GTFS schedule feed to disk.

    By default SSL certificate verification is enforced. If local certificate
    trust stores are incomplete (common in locked-down environments), pass
    allow_insecure_fallback=True to retry without SSL verification.
    """
    try:
        response = requests.get(gtfs_url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        if not allow_insecure_fallback:
            raise RuntimeError(
                "SSL certificate validation failed while downloading GTFS. "
                "Retry with --allow-insecure-download if you are on a trusted network."
            ) from exc

        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
        response = requests.get(gtfs_url, timeout=timeout, verify=False)
        response.raise_for_status()

    output_path.write_bytes(response.content)


def _read_csv_from_zip(zf: zipfile.ZipFile, file_name: str, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a single GTFS text file from a zip archive into a DataFrame."""
    with zf.open(file_name) as f:
        cols_arg = list(usecols) if usecols is not None else None
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=cols_arg)


def load_gtfs_tables(gtfs_zip_path: Path, id_prefix: str) -> Dict[str, pd.DataFrame]:
    """Load required GTFS tables for service and stop calculations."""
    required = {
        "stops": ("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"]),
        "routes": ("routes.txt", ["route_id", "route_short_name", "route_long_name", "route_type"]),
        "trips": ("trips.txt", ["route_id", "service_id", "trip_id"]),
        "stop_times": ("stop_times.txt", ["trip_id", "stop_id"]),
        "calendar": (
            "calendar.txt",
            [
                "service_id",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "start_date",
                "end_date",
            ],
        ),
    }

    tables: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(gtfs_zip_path, "r") as zf:
        for key, (name, cols) in required.items():
            tables[key] = _read_csv_from_zip(zf, name, usecols=cols)

    # Namespace IDs to avoid collisions across bus and rail feeds.
    tables["stops"]["stop_id"] = id_prefix + ":" + tables["stops"]["stop_id"].astype(str)
    tables["routes"]["route_id"] = id_prefix + ":" + tables["routes"]["route_id"].astype(str)

    tables["trips"]["trip_id"] = id_prefix + ":" + tables["trips"]["trip_id"].astype(str)
    tables["trips"]["route_id"] = id_prefix + ":" + tables["trips"]["route_id"].astype(str)
    tables["trips"]["service_id"] = id_prefix + ":" + tables["trips"]["service_id"].astype(str)

    tables["stop_times"]["trip_id"] = id_prefix + ":" + tables["stop_times"]["trip_id"].astype(str)
    tables["stop_times"]["stop_id"] = id_prefix + ":" + tables["stop_times"]["stop_id"].astype(str)

    tables["calendar"]["service_id"] = id_prefix + ":" + tables["calendar"]["service_id"].astype(str)

    return tables


def merge_gtfs_tables(table_sets: Iterable[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Merge normalized GTFS table dictionaries from multiple feeds."""
    keys = ["stops", "routes", "trips", "stop_times", "calendar"]
    merged: Dict[str, pd.DataFrame] = {}

    for key in keys:
        parts = [tables[key] for tables in table_sets]
        merged_df = pd.concat(parts, ignore_index=True)
        subset = ["stop_id"] if key == "stops" else None
        merged[key] = merged_df.drop_duplicates(subset=subset).reset_index(drop=True)

    return merged


def _active_weekday_services(calendar_df: pd.DataFrame) -> pd.Series:
    """Return service IDs that run on weekdays and are currently in date range."""
    today = dt.date.today().strftime("%Y%m%d")

    cal = calendar_df.copy()
    cal["start_date"] = cal["start_date"].astype(str)
    cal["end_date"] = cal["end_date"].astype(str)

    in_range = (cal["start_date"] <= today) & (cal["end_date"] >= today)
    weekday = (cal[["monday", "tuesday", "wednesday", "thursday", "friday"]].sum(axis=1) > 0)

    active = cal.loc[in_range & weekday, "service_id"]

    # Fall back if feed date windows do not include today's date.
    if active.empty:
        active = cal.loc[weekday, "service_id"]

    return active


def compute_stop_service_summary(
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build stop-level daily trip counts by mode type for a representative weekday.

    The output includes conservative proxy metrics for:
    - rail_subway_daily_trips (route_type 1)
    - rail_commuter_daily_trips (route_type 2)
    - rail_light_daily_trips (route_type 0)
    - bus_daily_trips (route_type 3)
    """
    active_services = _active_weekday_services(calendar)

    active_trips = trips[trips["service_id"].isin(active_services)].copy()
    if active_trips.empty:
        raise RuntimeError("No active GTFS trips found after service filtering.")

    trip_modes = active_trips.merge(routes[["route_id", "route_type"]], on="route_id", how="left")
    stop_trip_modes = stop_times.merge(trip_modes[["trip_id", "route_type"]], on="trip_id", how="inner")

    grouped = (
        stop_trip_modes.groupby(["stop_id", "route_type"], dropna=False)["trip_id"]
        .nunique()
        .reset_index(name="daily_trips")
    )

    pivot = grouped.pivot_table(
        index="stop_id",
        columns="route_type",
        values="daily_trips",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for route_type in (ROUTE_TYPE_TRAM, ROUTE_TYPE_SUBWAY, ROUTE_TYPE_RAIL, ROUTE_TYPE_BUS):
        if route_type not in pivot.columns:
            pivot[route_type] = 0

    pivot = pivot.rename(
        columns={
            ROUTE_TYPE_TRAM: "rail_light_daily_trips",
            ROUTE_TYPE_SUBWAY: "rail_subway_daily_trips",
            ROUTE_TYPE_RAIL: "rail_commuter_daily_trips",
            ROUTE_TYPE_BUS: "bus_daily_trips",
        }
    )

    pivot["rail_total_daily_trips"] = (
        pivot["rail_light_daily_trips"]
        + pivot["rail_subway_daily_trips"]
        + pivot["rail_commuter_daily_trips"]
    )

    summary = stops.merge(pivot, on="stop_id", how="left").fillna(0)

    numeric_cols = [
        "rail_light_daily_trips",
        "rail_subway_daily_trips",
        "rail_commuter_daily_trips",
        "bus_daily_trips",
        "rail_total_daily_trips",
    ]
    for col in numeric_cols:
        summary[col] = summary[col].astype(int)

    return summary


def assign_preliminary_sb79_tier(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    """
    Assign a preliminary SB 79 tier label using service-intensity heuristics.

    Heuristic logic:
    - Tier 1: subway/metro or commuter rail service at/above tier1 threshold
    - Tier 2: light rail or bus service at/above tier2 threshold
    - Ineligible: below both thresholds
    """
    out = df.copy()

    tier1_mask = (
        (out["rail_subway_daily_trips"] >= thresholds.tier1_daily_trips)
        | (out["rail_commuter_daily_trips"] >= thresholds.tier1_daily_trips)
    )
    tier2_mask = (
        (out["rail_light_daily_trips"] >= thresholds.tier2_daily_trips)
        | (out["bus_daily_trips"] >= thresholds.tier2_daily_trips)
    )

    out["prelim_sb79_tier"] = "ineligible"
    out.loc[tier2_mask, "prelim_sb79_tier"] = "tier2"
    out.loc[tier1_mask, "prelim_sb79_tier"] = "tier1"

    return out


def summarize_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact tier summary counts and service medians."""
    summary = (
        df.groupby("prelim_sb79_tier", dropna=False)
        .agg(
            stop_count=("stop_id", "count"),
            median_rail_trips=("rail_total_daily_trips", "median"),
            median_bus_trips=("bus_daily_trips", "median"),
        )
        .reset_index()
        .sort_values("stop_count", ascending=False)
    )
    return summary


def run_pipeline(
    paths: Paths,
    thresholds: Thresholds,
    bus_gtfs_url: str = BUS_GTFS_URL,
    rail_gtfs_url: str = RAIL_GTFS_URL,
    allow_insecure_download: bool = False,
) -> None:
    ensure_dirs(paths)

    print("[1/5] Downloading GTFS feeds (bus + rail)...")
    download_gtfs(
        bus_gtfs_url,
        paths.bus_gtfs_zip,
        allow_insecure_fallback=allow_insecure_download,
    )
    download_gtfs(
        rail_gtfs_url,
        paths.rail_gtfs_zip,
        allow_insecure_fallback=allow_insecure_download,
    )

    print("[2/5] Loading GTFS tables...")
    bus_tables = load_gtfs_tables(paths.bus_gtfs_zip, id_prefix="bus")
    rail_tables = load_gtfs_tables(paths.rail_gtfs_zip, id_prefix="rail")

    print("[3/5] Merging bus and rail tables...")
    tables = merge_gtfs_tables([bus_tables, rail_tables])

    print("[4/5] Computing stop-level service metrics...")
    stop_summary = compute_stop_service_summary(
        stops=tables["stops"],
        routes=tables["routes"],
        trips=tables["trips"],
        stop_times=tables["stop_times"],
        calendar=tables["calendar"],
    )
    stop_summary = assign_preliminary_sb79_tier(stop_summary, thresholds)

    print("[5/5] Writing outputs...")
    stop_summary.to_csv(paths.stop_summary_csv, index=False)

    tier_summary = summarize_tiers(stop_summary)
    tier_summary.to_csv(paths.tier_summary_csv, index=False)

    print("\nPhase 1 complete.")
    print(f"- Stop summary: {paths.stop_summary_csv}")
    print(f"- Tier summary: {paths.tier_summary_csv}")
    print("\nTier counts:")
    print(tier_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SB 79 Phase 1 GTFS stop-tier pipeline.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw downloads.")
    parser.add_argument("--interim-dir", default="data/interim", help="Directory for processed outputs.")
    parser.add_argument("--tier1-trips", type=int, default=72, help="Daily trip threshold for preliminary Tier 1.")
    parser.add_argument("--tier2-trips", type=int, default=48, help="Daily trip threshold for preliminary Tier 2.")
    parser.add_argument("--bus-gtfs-url", default=BUS_GTFS_URL, help="URL for Metro bus GTFS ZIP feed.")
    parser.add_argument("--rail-gtfs-url", default=RAIL_GTFS_URL, help="URL for Metro rail GTFS ZIP feed.")
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help="Retry GTFS download without SSL verification if certificate validation fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(raw_dir=Path(args.raw_dir), interim_dir=Path(args.interim_dir))
    thresholds = Thresholds(tier1_daily_trips=args.tier1_trips, tier2_daily_trips=args.tier2_trips)

    run_pipeline(
        paths=paths,
        thresholds=thresholds,
        bus_gtfs_url=args.bus_gtfs_url,
        rail_gtfs_url=args.rail_gtfs_url,
        allow_insecure_download=args.allow_insecure_download,
    )


if __name__ == "__main__":
    main()
