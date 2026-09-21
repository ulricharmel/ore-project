"""
cost_engine.py
==============
Courier cost lookups and LMR zone resolution.

Given a facility and an order destination we can look up:
  - The LMR zone the facility falls in relative to the destination
  - Which couriers can serve the route
  - The cheapest valid courier cost

Reference data used:
  - courier_costs.csv            (delivery_type × courier × LMR → cost)
  - courier_post_code_mapping.csv (postal_code → HubID, LMR, CourierId)
  - bash_fcm_facilities.csv      (facility_id → postal_code)
  - dc_branch_numbers.csv        (branch_number → dc_type)

The cost table uses 10 000 as a sentinel for "route not available".
We treat cost ≥ 9 999 as infinity.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    load_courier_costs,
    load_postcode_mapping,
    load_facilities,
    load_dc_branches,
    load_inventory,
)

INFINITY_COST = 10_000   # sentinel value in courier_costs.csv


class CostEngine:
    """
    Pre-loads all reference tables and exposes fast cost-lookup methods.

    Usage
    -----
    engine = CostEngine()

    # Cost for a single parcel from a facility to a destination
    cost = engine.parcel_cost(
        delivery_type="DoorToDoor",
        facility_postal="0699",
        dest_postal="7880",
        courier_c_active=False,
    )

    # Cheapest cost for each facility (returns Series indexed by facility_id)
    costs = engine.cheapest_facility_costs(
        delivery_type="DoorToDoor",
        dest_postal="7880",
        facility_df=inventory_slice,
    )
    """

    def __init__(self) -> None:
        self._costs      = load_courier_costs()          # delivery_type, courier_id, lmr, cost
        self._postcode   = load_postcode_mapping()        # PostalCode → HubID, LMR, CourierId
        self._facilities = load_facilities()             # facility_id → postal_code
        self._dc_branches = load_dc_branches()           # branch_number → dc_type

        # Build fast lookup dicts.
        #
        # The postcode mapping has up to 3 rows per postal code — one per
        # courier (courier_a, courier_b, courier_c).  Each row carries the
        # HubID and LMR zone for that courier's own network.
        #
        # _postcode_lmr / _postcode_hub: keyed by postal code only, using the
        # LAST value written per code (backward-compatible fallback, used only
        # when no courier-specific lookup is needed).
        self._postcode_lmr: dict[str, str] = dict(
            zip(self._postcode["PostalCode"], self._postcode["LMR"])
        )
        # postcode_hub: "0699" → hub (last entry per postal code — fallback only)
        self._postcode_hub: dict[str, str] = dict(
            zip(self._postcode["PostalCode"], self._postcode["HubID"])
        )
        # postcode_courier: preferred courier for a postal code
        self._postcode_courier: dict[str, str] = dict(
            zip(self._postcode["PostalCode"], self._postcode["CourierId"])
        )
        # Courier-specific lookup: (courier_id, postal_code) → (HubID, LMR)
        # This is the correct mapping for route LMR calculation.
        self._courier_pc: dict[tuple[str, str], tuple[str, str]] = {
            (row.CourierId, row.PostalCode): (row.HubID, row.LMR)
            for row in self._postcode.itertuples(index=False)
        }
        # facility_postal: facility_id → postal_code
        self._facility_postal: dict[str, str] = dict(
            zip(self._facilities["facility_id"], self._facilities["postal_code"])
        )
        # dc set: set of branch_numbers that are DCs
        self._dc_set: set[str] = set(self._dc_branches["branch_number"])

        # Build cost lookup dict: (delivery_type, courier_id, lmr) → cost
        self._cost_lookup: dict[tuple[str, str, str], int] = {
            (row.delivery_type, row.courier_id, row.lmr): row.cost
            for row in self._costs.itertuples(index=False)
        }

        # All known couriers
        self._couriers = self._costs["courier_id"].unique().tolist()

        # courier_c proximity: lat/lon arrays for all courier_c-active facilities
        # Used for the 20 km eligibility check (no external geocoding library needed).
        self._cc_fac_lats: np.ndarray
        self._cc_fac_lons: np.ndarray
        self._init_courier_c_facilities()

    # ── courier_c proximity initialisation ───────────────────────────────

    def _init_courier_c_facilities(self) -> None:
        """
        Pre-load WKB coordinates for all courier_c-active facilities.

        The WKB blobs are EWKB (SRID=4326), encoded as hex strings.
        Layout (little-endian):
          byte  0      : 01  (little-endian flag)
          bytes 1-4    : geometry type (Point + SRID flag)
          bytes 5-8    : SRID value (4326)
          bytes 9-16   : longitude as IEEE-754 double
          bytes 17-24  : latitude  as IEEE-754 double
        Hex offsets:  lon = chars 18-34, lat = chars 34-50.
        """
        inv = load_inventory()
        cc_inv = (
            inv[inv["courier_c_active"].fillna(False).astype(bool)]
            [["facility_id", "facility_coordinates"]]
            .drop_duplicates("facility_id")
            .dropna(subset=["facility_coordinates"])
        )

        lats, lons = [], []
        for wkb_hex in cc_inv["facility_coordinates"].astype(str):
            try:
                lon = struct.unpack("<d", bytes.fromhex(wkb_hex[18:34]))[0]
                lat = struct.unpack("<d", bytes.fromhex(wkb_hex[34:50]))[0]
                lons.append(lon)
                lats.append(lat)
            except Exception:
                continue

        self._cc_fac_lats = np.array(lats, dtype=float)
        self._cc_fac_lons = np.array(lons, dtype=float)

    def within_20km_courier_c(
        self,
        dest_lat: float | None,
        dest_lon: float | None,
        radius_km: float = 20.0,
    ) -> bool:
        """
        Return True if the destination (lat, lon) is within `radius_km` km of
        **any** courier_c-active facility.

        Uses the haversine formula (no external library required).
        Returns False when destination coordinates are missing.
        """
        if (
            dest_lat is None
            or dest_lon is None
            or np.isnan(dest_lat)
            or np.isnan(dest_lon)
            or len(self._cc_fac_lats) == 0
        ):
            return False

        R = 6_371.0  # Earth radius in km
        lat1 = np.radians(dest_lat)
        lon1 = np.radians(dest_lon)
        lat2 = np.radians(self._cc_fac_lats)
        lon2 = np.radians(self._cc_fac_lons)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distances = 2 * R * np.arcsin(np.sqrt(a))
        return bool(distances.min() <= radius_km)

    # ── Zone resolution ───────────────────────────────────────────────────

    def lmr_zone(self, postal_code: str | None) -> str | None:
        """
        Return the LMR zone for a postal code: 'Local', 'Main', or 'Regional'.
        Returns None if the postal code is unknown.
        """
        if not postal_code:
            return None
        return self._postcode_lmr.get(str(postal_code).zfill(4))

    def route_lmr(
        self,
        courier_id: str,
        facility_postal: str | None,
        dest_postal: str | None,
    ) -> str | None:
        """
        Determine the route LMR zone for a given courier.

        The postcode mapping has one row per (PostalCode, CourierId), giving
        the HubID and LMR for *that courier's own network*.  Route LMR is
        then computed from origin × destination using the courier-specific hubs:

          1. If either the collection or delivery postcode is Regional in that
             courier's network → route is Regional.
          2. If both are Local/Main but the two hubs differ → route is Main.
          3. If both are Local/Main and the same hub → route is Local.

        Returns None when a postal code has no entry for the requested courier.
        """
        if not facility_postal or not dest_postal:
            return None

        fac_p  = str(facility_postal).zfill(4)
        dest_p = str(dest_postal).zfill(4)

        fac_info  = self._courier_pc.get((courier_id, fac_p))
        dest_info = self._courier_pc.get((courier_id, dest_p))

        if fac_info is None or dest_info is None:
            return None

        hub_fac,  lmr_fac  = fac_info
        hub_dest, lmr_dest = dest_info

        # Rule 1 — either side is Regional in this courier's network
        if lmr_fac == "Regional" or lmr_dest == "Regional":
            return "Regional"

        # Rules 2 & 3 — both Local/Main; compare hubs
        return "Local" if hub_fac == hub_dest else "Main"

    def courier_cost(
        self,
        delivery_type: str,
        courier_id: str,
        lmr: str | None,
    ) -> int:
        """
        Return the raw cost (in cents) for a (delivery_type, courier, lmr) triple.
        Returns 10_000_000 (sentinel infinity) if the combination is not in the
        cost table.
        """
        return self._cost_lookup.get((delivery_type, courier_id, lmr), 10_000_000)

    def preferred_courier(self, postal_code: str | None) -> str | None:
        """Return the preferred courier for a postal code."""
        if not postal_code:
            return None
        return self._postcode_courier.get(str(postal_code).zfill(4))

    def facility_postal(self, facility_id: str) -> str | None:
        """Return the postal code for a facility_id."""
        return self._facility_postal.get(facility_id)

    def is_dc(self, branch_number: str) -> bool:
        """Return True if branch_number is a Distribution Centre."""
        return str(branch_number).zfill(6) in self._dc_set

    # ── Cost lookup ───────────────────────────────────────────────────────

    def courier_cost(
        self,
        delivery_type: str,
        courier_id: str,
        lmr: str,
    ) -> int:
        """
        Look up cost for (delivery_type, courier, LMR).
        Returns INFINITY_COST if the combination is not available.
        """
        return self._cost_lookup.get(
            (delivery_type, courier_id, lmr), INFINITY_COST
        )

    def cheapest_courier_cost(
        self,
        delivery_type: str,
        lmr: str | None,
        courier_c_active: bool = True,
    ) -> tuple[int, str | None]:
        """
        Find the cheapest available courier for a given delivery_type and LMR zone.

        Parameters
        ----------
        delivery_type   : "DoorToDoor" or "StorePickup"
        lmr             : "Local", "Main", "Regional", or None
        courier_c_active: whether courier_c is eligible for this facility

        Returns
        -------
        (cost, courier_id) — cost is INFINITY_COST if no courier available
        """
        if lmr is None:
            return INFINITY_COST, None

        best_cost = INFINITY_COST
        best_courier = None
        for courier in self._couriers:
            if courier == "courier_c" and not courier_c_active:
                continue
            c = self._cost_lookup.get((delivery_type, courier, lmr), INFINITY_COST)
            if c < best_cost:
                best_cost = c
                best_courier = courier
        return best_cost, best_courier

    def parcel_cost(
        self,
        delivery_type: str,
        facility_postal: str | None,
        dest_postal: str | None,
        courier_c_active: bool = True,
    ) -> tuple[int, str | None]:
        """
        Compute the cheapest parcel cost for shipping from a facility postal code
        to a destination postal code.

        The LMR zone used is the zone of the FACILITY (i.e. the facility's
        distance/hub classification relative to the wider network).

        Returns (cost, courier_id). Cost is INFINITY_COST if unavailable.
        """
        lmr = self.lmr_zone(facility_postal)
        return self.cheapest_courier_cost(delivery_type, lmr, courier_c_active)

    # ── Batch helpers ─────────────────────────────────────────────────────

    def annotate_inventory(self, inv_df: pd.DataFrame, delivery_type: str) -> pd.DataFrame:
        """
        Add cost columns to an inventory DataFrame.

        Expects columns: facility_id, facility_region, courier_c_active
        Adds columns:
          lmr_zone        – resolved LMR zone for the facility
          facility_postal – postal code of the facility
          cheapest_cost   – cheapest parcel cost (int)
          cheapest_courier– courier with that cost
        """
        df = inv_df.copy()

        # Resolve postal code from facility_id → postal_code mapping
        df["facility_postal"] = df["facility_id"].map(self._facility_postal)

        # Resolve LMR zone from facility postal
        df["lmr_zone"] = df["facility_postal"].map(
            lambda p: self._postcode_lmr.get(str(p).zfill(4) if p else "", None)
            if pd.notna(p) else None
        )

        # Fall back to FacilityRegion if postal lookup failed
        # FacilityRegion is stored as LOCAL/MAIN/REGIONAL — map to title case
        region_map = {"LOCAL": "Local", "MAIN": "Main", "REGIONAL": "Regional"}
        mask_missing = df["lmr_zone"].isna()
        if "facility_region" in df.columns:
            df.loc[mask_missing, "lmr_zone"] = (
                df.loc[mask_missing, "facility_region"].map(region_map)
            )

        # Compute cheapest cost per row
        def _row_cost(row):
            cost, courier = self.cheapest_courier_cost(
                delivery_type=delivery_type,
                lmr=row["lmr_zone"],
                courier_c_active=bool(row.get("courier_c_active", True)),
            )
            return pd.Series({"cheapest_cost": cost, "cheapest_courier": courier})

        cost_cols = df.apply(_row_cost, axis=1)
        df["cheapest_cost"]    = cost_cols["cheapest_cost"]
        df["cheapest_courier"] = cost_cols["cheapest_courier"]
        return df

    def cost_summary(self) -> pd.DataFrame:
        """Return the full cost table (useful for EDA)."""
        return self._costs.copy()
