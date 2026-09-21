"""
data_loader.py
==============
Streaming parser for allocation_algorithm_logs CSV.

The file has ~3M rows and each row contains three JSON columns:
  - request          → order details + items
  - AVAILABLE_INVENTORY → list of stock records per facility/SKU
  - PLAN             → the production algorithm's allocation output

We parse in chunks and materialise four flat DataFrames which are cached
as Parquet files for fast re-loading on subsequent runs.

Outputs (all in data/cache/):
  orders.parquet        – one row per order
  items.parquet         – one row per (order, item)
  inventory.parquet     – one row per (order, facility, sku) stock record
  reservations.parquet  – one row per reservation in the plan
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_CSV = DATA_DIR / "allocation_algorithm_logs_20240105.csv"

CACHE_FILES = {
    "orders":       CACHE_DIR / "orders.parquet",
    "items":        CACHE_DIR / "items.parquet",
    "inventory":    CACHE_DIR / "inventory.parquet",
    "reservations": CACHE_DIR / "reservations.parquet",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal row parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_request(req: dict, order_idx: int) -> tuple[dict, list[dict]]:
    """Extract order-level and item-level records from a request dict."""
    dest = req.get("destination", {}) or {}
    order = {
        "order_idx":            order_idx,
        "order_id":             req.get("order_id"),
        "external_ref":         req.get("external_order_reference"),
        "order_created_at":     req.get("order_created_at"),
        "dry_run":              req.get("dry_run", False),
        "delivery_type":        dest.get("delivery_type"),
        "destination_postal":   dest.get("postal_code"),
        "destination_province": dest.get("province"),
        "destination_branch":   dest.get("branch_number"),
        "dest_lat":             dest.get("latitude"),
        "dest_lon":             dest.get("longitude"),
        "n_items":              len(req.get("items", [])),
    }
    items = []
    for it in req.get("items", []):
        items.append({
            "order_idx":             order_idx,
            "order_id":              req.get("order_id"),
            "order_item_id":         it.get("order_item_id"),
            "sku":                   it.get("sku"),
            "trading_company_number":it.get("trading_company_number"),
            "quantity":              it.get("quantity", 1),
            "excluded_branches":     it.get("excluded_branches") or [],   # list[str] of branch numbers
            "has_excluded_branches": bool(it.get("excluded_branches")),
            "has_alternative_sku":   it.get("alternative_sku") is not None,
            "alternative_sku":       it.get("alternative_sku"),
        })
    return order, items


def _parse_inventory(inv_list: list, order_idx: int, order_id: str) -> list[dict]:
    """Flatten a list of inventory records."""
    records = []
    for rec in inv_list:
        records.append({
            "order_idx":            order_idx,
            "order_id":             order_id,
            "facility_id":          rec.get("FacilityID"),
            "branch_number":        rec.get("BranchNumber"),
            "sku":                  rec.get("Sku"),
            "normalised_sku":       rec.get("NormalisedSku"),
            "trading_company":      rec.get("TradingCompanyNumber"),
            "facility_type":        rec.get("FacilityType"),       # Store | DC
            "facility_region":      rec.get("FacilityRegion"),     # LOCAL | MAIN | REGIONAL
            "facility_province":    rec.get("FacilityProvince"),
            "facility_active":      rec.get("FacilityActive", True),
            "facility_at_capacity": rec.get("FacilityAtCapacity", False),
            "courier_c_active":     rec.get("FacilityCourierCActive", False),
            "facility_buffer":      rec.get("FacilityBuffer", 0),
            "on_hand_qty":          rec.get("OnHandQuantity", 0),
            "qty_available":        rec.get("QtyAvailable", 0),
            "reserved_qty":         rec.get("FacilityReservedQty", 0),
            "facility_coordinates": rec.get("FacilityCoordinates"),
            "stock_updated_at":     rec.get("StockSourceUpdatedAt"),
        })
    return records


def _parse_plan(plan: dict, order_idx: int) -> list[dict]:
    """Flatten the plan's reservations list."""
    records = []
    for res in plan.get("reservations") or []:
        ofd = res.get("omni_fulfilment_data") or {}
        records.append({
            "order_idx":           order_idx,
            "order_id":            plan.get("order_id"),
            "allocation_id":       plan.get("allocation_id"),
            "order_item_id":       res.get("order_item_id"),
            "reservation_id":      res.get("reservation_id"),
            "branch_number":       res.get("branch_number"),
            "courier":             res.get("courier"),
            "sku":                 res.get("sku"),
            "quantity":            res.get("quantity", 1),
            "trading_company":     res.get("trading_company_number"),
            "fc_code":             ofd.get("fulfilment_centre_code"),
            "fc_id":               ofd.get("fulfilment_centre_id"),
            "fc_description":      ofd.get("fulfilment_centre_description"),
            "created_at":          res.get("created_at"),
        })
    # Track cancellations count
    n_cancellations = len(plan.get("cancellations") or [])
    for r in records:
        r["n_cancellations"] = n_cancellations
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Streaming builder
# ─────────────────────────────────────────────────────────────────────────────

def _stream_rows(csv_path: Path) -> Iterator[tuple[int, dict, list, dict]]:
    """Yield (order_idx, request, inventory_list, plan) for every log row."""
    # Some rows contain very large JSON blobs — raise the field size limit
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            try:
                req  = json.loads(row["request"])
                inv  = json.loads(row["AVAILABLE_INVENTORY"])
                plan = json.loads(row["PLAN"])
                yield idx, req, inv, plan
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Skipping row %d — parse error: %s", idx, exc)
                continue


def build_cache(
    csv_path: Path = LOG_CSV,
    cache_dir: Path = CACHE_DIR,
    chunk_size: int = 50_000,
    max_rows: int | None = None,
    verbose: bool = True,
) -> None:
    """
    Parse the full CSV and write four Parquet files to cache_dir.

    Parameters
    ----------
    csv_path   : path to the allocation log CSV
    cache_dir  : directory where parquet files will be written
    chunk_size : number of rows per write batch (controls memory usage)
    max_rows   : if set, stop after this many rows (useful for dev/testing)
    verbose    : print progress
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    writers = {}   # name → ParquetWriter (opened lazily on first batch)
    buffers: dict[str, list] = {k: [] for k in CACHE_FILES}
    total = 0

    def _flush(force: bool = False):
        nonlocal writers
        for name, rows in buffers.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if name not in writers:
                writers[name] = pq.ParquetWriter(
                    CACHE_FILES[name], table.schema, compression="snappy"
                )
            writers[name].write_table(table)
            buffers[name].clear()

    try:
        for order_idx, req, inv, plan in _stream_rows(csv_path):
            order_id = req.get("order_id", "")

            order_rec, item_recs = _parse_request(req, order_idx)
            inv_recs             = _parse_inventory(inv, order_idx, order_id)
            res_recs             = _parse_plan(plan, order_idx)

            buffers["orders"].append(order_rec)
            buffers["items"].extend(item_recs)
            buffers["inventory"].extend(inv_recs)
            buffers["reservations"].extend(res_recs)

            total += 1
            if total % chunk_size == 0:
                _flush()
                if verbose:
                    print(f"  Parsed {total:,} rows…", flush=True)

            if max_rows and total >= max_rows:
                break

        _flush(force=True)   # flush remainder
    finally:
        for w in writers.values():
            w.close()

    if verbose:
        print(f"✓ Done — {total:,} rows parsed → {cache_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Public load API
# ─────────────────────────────────────────────────────────────────────────────

def cache_exists() -> bool:
    """True if all four parquet cache files exist."""
    return all(p.exists() for p in CACHE_FILES.values())


def load_orders() -> pd.DataFrame:
    """Load the orders parquet (one row per order)."""
    df = pd.read_parquet(CACHE_FILES["orders"])
    df["order_created_at"] = pd.to_datetime(df["order_created_at"], utc=True, errors="coerce")
    return df


def load_items() -> pd.DataFrame:
    """Load the items parquet (one row per order-item)."""
    return pd.read_parquet(CACHE_FILES["items"])


def load_inventory() -> pd.DataFrame:
    """Load the inventory parquet (one row per order × facility × sku)."""
    return pd.read_parquet(CACHE_FILES["inventory"])


def load_reservations() -> pd.DataFrame:
    """Load the reservations parquet (one row per plan reservation)."""
    df = pd.read_parquet(CACHE_FILES["reservations"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience: return all four DataFrames in a dict."""
    return {
        "orders":       load_orders(),
        "items":        load_items(),
        "inventory":    load_inventory(),
        "reservations": load_reservations(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reference data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_courier_costs() -> pd.DataFrame:
    """
    Load courier_costs.csv.
    Columns: delivery_type, courier_id, lmr, cost
    Treats cost == 10000 as 'not available' (infinity sentinel).
    """
    df = pd.read_csv(DATA_DIR / "courier_costs.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["available"] = df["cost"] < 9_999
    return df


def load_postcode_mapping() -> pd.DataFrame:
    """
    Load courier_post_code_mapping.csv.
    Columns: PostalCode, HubID, LMR, CourierId
    """
    df = pd.read_csv(DATA_DIR / "courier_post_code_mapping.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df["PostalCode"] = df["PostalCode"].str.zfill(4)
    return df


def load_dc_branches() -> pd.DataFrame:
    """Load dc_branch_numbers.csv. Returns branch_number → dc_type mapping."""
    df = pd.read_csv(DATA_DIR / "dc_branch_numbers.csv", dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    df["branch_number"] = df["branch_number"].str.zfill(6)
    return df


def load_facilities() -> pd.DataFrame:
    """Load bash_fcm_facilities.csv. Returns facility_id → postal_code."""
    df = pd.read_csv(DATA_DIR / "bash_fcm_facilities.csv", dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    df["postal_code"] = df["postal_code"].str.zfill(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for one-time cache build
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build parquet cache from allocation logs CSV.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows (useful for quick dev builds)")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    print(f"Building cache from {LOG_CSV} …")
    build_cache(max_rows=args.max_rows, chunk_size=args.chunk_size)
