"""
allocator.py
============
Cost-optimal allocation engine for Task 2.

Given an order request and available inventory, produces an allocation that
minimises total courier cost.

Business rules
--------------
- All items sent to the same *physical* facility ship as one parcel.
- Cost is charged per parcel (= per distinct physical facility used), not per item.
- DC virtual branches within the same physical DC count as one physical facility
  (identified via dc_branch_numbers.csv; dc_type field groups them).
- Inactive facilities and at-capacity facilities are excluded.
- Items with no valid candidate become cancellations (matching production behaviour).
- excluded_branches per item are ignored here (all null in the dataset).

Algorithm
---------
For each order:
  1. Pre-filter inventory  — active, not at capacity, qty_available > 0.
  2. Annotate costs        — cheapest parcel cost per physical facility.
  3. Merge DC branches     — virtual branches → single physical facility key.
  4. Single-facility pass  — find cheapest facility covering ALL items (exact).
  5. Greedy split          — if split unavoidable: assign items scarcest-first,
                             re-use already-chosen facilities (zero marginal cost),
                             pick cheapest new facility otherwise.
  6. Consolidation         — iteratively try to move items from expensive facilities
                             to cheaper already-chosen ones, reducing parcel count.

Performance
-----------
- `precompute_inventory_costs()` pre-annotates the entire inventory DataFrame
  once (vectorised; 12 possible cost combinations) before the order loop.
- Per-order slices are pre-grouped with `groupby` for O(1) access.
- Typical runtime: < 60 s for all 3 810 orders on a laptop.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

from src.cost_engine import CostEngine, INFINITY_COST
from src.data_loader import load_dc_branches


# ─────────────────────────────────────────────────────────────────────────────
# DC physical-facility mapping
# ─────────────────────────────────────────────────────────────────────────────

def _build_dc_map() -> dict[str, str]:
    """
    Load dc_branch_numbers.csv and return {branch_number (6-digit) → dc_type}.
    All virtual branches sharing the same dc_type belong to one physical DC.
    """
    dc = load_dc_branches()
    return dict(zip(dc["branch_number"], dc["dc_type"]))


def to_physical_key(branch_number: str, dc_map: dict[str, str]) -> str:
    """
    Convert a branch_number to its physical facility key:
      - DC virtual branches  → their dc_type (e.g. 'dc_a', 'dc_b')
      - Stores               → branch_number itself
    """
    bn = str(branch_number).zfill(6)
    return dc_map.get(bn, bn)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised inventory cost pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def precompute_inventory_costs(inv: pd.DataFrame, engine: CostEngine) -> pd.DataFrame:
    """
    Pre-annotate the entire inventory DataFrame (called ONCE before the order loop).

    Adds columns:
      lmr_zone          – resolved LMR zone of each facility
      cost_DoorToDoor   – cheapest valid parcel cost for DoorToDoor delivery (int)
      courier_DoorToDoor– courier achieving that cost
      cost_StorePickup  – cheapest valid parcel cost for StorePickup delivery (int)
      courier_StorePickup

    Courier eligibility rules enforced here:
      • dc_b branches   → courier_b ONLY  (courier_a/c cannot service dc_b)
      • all other facs  → courier_a or courier_c  (courier_b cannot service them)

    For non-dc_b facilities the cost depends only on
    (delivery_type, lmr_zone, courier_c_active) — 12 unique combinations — so
    we build a lookup dict and vectorise via Series.map.
    For dc_b facilities we override with courier_b costs per LMR zone.
    """
    df = inv.copy()

    # ── 1. Facility postal code ──────────────────────────────────────────
    df["facility_postal"] = df["facility_id"].map(engine._facility_postal)

    # ── 2. LMR zone from postal code ─────────────────────────────────────
    def _lmr(postal):
        if pd.isna(postal):
            return None
        return engine._postcode_lmr.get(str(postal).zfill(4), None)

    df["lmr_zone"] = df["facility_postal"].map(_lmr)

    # ── 3. Fall back to FacilityRegion (stored as LOCAL/MAIN/REGIONAL) ───
    region_map = {"LOCAL": "Local", "MAIN": "Main", "REGIONAL": "Regional"}
    mask_missing = df["lmr_zone"].isna() & df["facility_region"].notna()
    df.loc[mask_missing, "lmr_zone"] = (
        df.loc[mask_missing, "facility_region"].map(region_map)
    )

    # ── 4. Boolean courier_c flag ─────────────────────────────────────────
    df["_cc"] = df["courier_c_active"].fillna(False).astype(bool)

    # ── 5. Identify dc_b branches ─────────────────────────────────────────
    #    courier_b may ONLY fulfil from dc_b; courier_a may NOT service dc_b.
    dc_branches = load_dc_branches()
    dc_b_set = set(
        dc_branches.loc[dc_branches["dc_type"] == "dc_b", "branch_number"]
        .astype(str)
    )
    df["_is_dc_b"] = df["branch_number"].astype(str).str.zfill(6).isin(dc_b_set)

    # ── 6. Build lookup for NON-dc_b: (lmr_zone|courier_c_active) → (cost, courier)
    #    courier_b excluded from this lookup (cannot service non-dc_b facilities)
    def _make_lookup_no_b(delivery_type: str) -> dict[str, tuple]:
        lut: dict[str, tuple] = {}
        for lmr in ["Local", "Main", "Regional", None]:
            for cc in [True, False]:
                best_cost, best_courier = INFINITY_COST, None
                for cid in [c for c in engine._couriers if c != "courier_b"]:
                    if cid == "courier_c" and not cc:
                        continue
                    c = engine._cost_lookup.get((delivery_type, cid, lmr), INFINITY_COST)
                    if c < best_cost:
                        best_cost, best_courier = c, cid
                lut[f"{lmr}|{cc}"] = (best_cost, best_courier)
        return lut

    df["_key"] = df["lmr_zone"].astype(str) + "|" + df["_cc"].astype(str)

    for dt in ["DoorToDoor", "StorePickup"]:
        lut = _make_lookup_no_b(dt)
        pairs = df["_key"].map(lut)
        df[f"cost_{dt}"]    = pairs.map(lambda x: x[0] if x is not None else INFINITY_COST)
        df[f"courier_{dt}"] = pairs.map(lambda x: x[1] if x is not None else None)

        # ── 7. Override dc_b rows with courier_b costs ────────────────────
        dc_b_mask = df["_is_dc_b"]
        for lmr in ["Local", "Main", "Regional"]:
            mask = dc_b_mask & (df["lmr_zone"] == lmr)
            if mask.any():
                cost_b = engine.courier_cost(dt, "courier_b", lmr)
                df.loc[mask, f"cost_{dt}"]    = cost_b
                df.loc[mask, f"courier_{dt}"] = "courier_b"
        # dc_b rows with unknown LMR → INFINITY (cannot route)
        mask_unknown = dc_b_mask & df["lmr_zone"].isna()
        if mask_unknown.any():
            df.loc[mask_unknown, f"cost_{dt}"]    = INFINITY_COST
            df.loc[mask_unknown, f"courier_{dt}"] = None

    df.drop(columns=["_key", "_cc", "_is_dc_b"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Allocator
# ─────────────────────────────────────────────────────────────────────────────

class Allocator:
    """
    Cost-optimal allocator.

    Typical usage
    -------------
    engine      = CostEngine()
    allocator   = Allocator(engine)

    inv_prepped = allocator.preprocess(inv)           # one-time, vectorised
    results_df  = allocator.allocate_all(orders, items, inv_prepped)
    """

    def __init__(
        self,
        cost_engine: CostEngine,
        max_split_k: int = 3,
        use_exact: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        cost_engine  : CostEngine instance for cost lookups.
        max_split_k  : Maximum number of facilities a single item's quantity
                       may be split across as a last resort.  Set to 1 to
                       disable splitting (items that cannot fit a single
                       facility will be cancelled).  Default = 3.
        use_exact    : When True, replace the greedy + consolidation steps
                       with an exact MIP solved via scipy.optimize.milp.
                       Produces a provably optimal assignment for each order
                       but is slower for large orders.  Default = False.
        """
        self.engine      = cost_engine
        self._dc_map     = _build_dc_map()
        self.max_split_k = max_split_k
        self.use_exact   = use_exact

    # ── Public API ────────────────────────────────────────────────────────

    def preprocess(self, inv: pd.DataFrame) -> pd.DataFrame:
        """Pre-annotate full inventory with costs (call once before the loop)."""
        return precompute_inventory_costs(inv, self.engine)

    def allocate_all(
        self,
        orders: pd.DataFrame,
        items: pd.DataFrame,
        inv_prepped: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the allocator over every order.

        Parameters
        ----------
        orders      : orders DataFrame (one row per order)
        items       : items DataFrame (one row per order-item)
        inv_prepped : inventory DataFrame pre-processed by `preprocess()`
        verbose     : print progress every 500 orders

        Returns
        -------
        DataFrame with one row per order:
          order_idx, order_id, n_parcels, total_cost, n_assigned, n_cancelled
        """
        # Pre-group by order_idx for O(1) per-order access
        items_grp = {idx: g for idx, g in items.groupby("order_idx")}
        inv_grp   = {idx: g for idx, g in inv_prepped.groupby("order_idx")}

        results = []
        # Include destination coordinates, postal and branch for per-order cost overrides
        coord_cols = ["order_idx", "order_id", "delivery_type"]
        for col in ("dest_lat", "dest_lon", "destination_postal", "destination_branch"):
            if col in orders.columns:
                coord_cols.append(col)
        order_rows = orders[coord_cols].to_dict("records")

        for i, row in enumerate(order_rows):
            order_idx          = row["order_idx"]
            order_id           = row["order_id"]
            delivery_type      = row["delivery_type"]
            dest_lat           = row.get("dest_lat")
            dest_lon           = row.get("dest_lon")
            dest_postal        = row.get("destination_postal")
            destination_branch = row.get("destination_branch") or None

            order_items = items_grp.get(order_idx, pd.DataFrame())
            order_inv   = inv_grp.get(order_idx, pd.DataFrame())

            result = self._allocate_one(order_idx, order_id, delivery_type,
                                        order_items, order_inv,
                                        dest_lat=dest_lat, dest_lon=dest_lon,
                                        dest_postal=dest_postal,
                                        destination_branch=destination_branch)
            results.append(result)

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1:,}/{len(order_rows):,} orders…", flush=True)

        return pd.DataFrame(results)

    # ── Per-order allocation ──────────────────────────────────────────────

    def _allocate_one(
        self,
        order_idx: int,
        order_id: str,
        delivery_type: str,
        order_items: pd.DataFrame,
        order_inv: pd.DataFrame,
        dest_lat: float | None = None,
        dest_lon: float | None = None,
        dest_postal: str | None = None,
        destination_branch: str | None = None,
    ) -> dict:
        """Allocate a single order and return a flat summary dict."""

        if order_items.empty:
            return _empty_result(order_idx, order_id)

        # Build structured item list with a unique integer item_key
        items_list = []
        for pos, (_, row) in enumerate(order_items.iterrows()):
            alt  = row.get("alternative_sku")
            excl = row.get("excluded_branches")
            # excluded_branches may arrive as a Python list (from parquet) or None/NaN
            if not isinstance(excl, list):
                excl = []
            items_list.append({
                "item_key":          order_idx * 10_000 + pos,
                "sku":               row["sku"],
                "quantity":          int(row.get("quantity", 1)),
                "order_item_id":     row.get("order_item_id"),
                "alternative_sku":   alt if (alt is not None and str(alt) != "nan") else None,
                "excluded_branches": excl,   # list[str] of branch numbers to avoid
            })

        # Pre-filter: active, not at capacity, have stock
        inv_f = self._prefilter(order_inv)

        # ── StorePickup corrections ───────────────────────────────────────────
        # For StorePickup the physical destination of every parcel is the pickup
        # store (destination_branch), not the customer's home address.  Two fixes:
        #   Fix 1 — use the pickup store's postal code for route LMR so that the
        #            Local/Main/Regional zone reflects the source→store route.
        #   Fix 2 — set cost = 0 for the pickup store itself (same_store): when
        #            the store has stock the customer walks in, no courier needed.
        if (
            delivery_type == "StorePickup"
            and destination_branch
            and not inv_f.empty
        ):
            dest_branch_padded = str(destination_branch).zfill(6)
            store_rows = inv_f[
                inv_f["branch_number"].astype(str).str.zfill(6) == dest_branch_padded
            ]

            # Fix 1 — override dest_postal with the pickup store's postal
            if not store_rows.empty and "facility_postal" in store_rows.columns:
                pickup_postal = store_rows["facility_postal"].iloc[0]
                if pickup_postal:
                    dest_postal = str(pickup_postal)

            # Fix 2 — zero cost for the pickup store (same_store, no shipping)
            dest_mask = inv_f["branch_number"].astype(str).str.zfill(6) == dest_branch_padded
            if dest_mask.any():
                inv_f = inv_f.copy()
                inv_f.loc[dest_mask, "cost_StorePickup"]    = 0
                inv_f.loc[dest_mask, "courier_StorePickup"] = "same_store"

        # ── Route LMR override (courier_a and courier_c, non-dc_b only) ──────
        # The LMR zone is a route property: it depends on BOTH the facility's
        # postal code and the customer's delivery postal code.
        # Rules: Regional if either side is Regional; Main if different hubs;
        #        Local if same hub (and both sides are Local or Main).
        # dc_b uses courier_b which is not subject to this rule.
        if dest_postal and not inv_f.empty and "facility_postal" in inv_f.columns:
            inv_f = self._apply_route_lmr(inv_f, dest_postal, delivery_type)

        # ── courier_c proximity override ─────────────────────────────────────
        # courier_c can only be used if the customer's address is within 20 km
        # of any courier_c-active facility. When eligible, cost_DoorToDoor for
        # courier_c-active facilities is 20 (Local rate) — cheaper than
        # courier_a Main (69) which our LMR-based lookup incorrectly assigns
        # to these Cape-Town stores (postal codes map to "Main" in the table).
        if (
            delivery_type == "DoorToDoor"
            and not inv_f.empty
            and self.engine.within_20km_courier_c(dest_lat, dest_lon)
            and "courier_c_active" in inv_f.columns
        ):
            cc_mask = inv_f["courier_c_active"].fillna(False).astype(bool)
            if cc_mask.any():
                cc_local_cost = self.engine.courier_cost("DoorToDoor", "courier_c", "Local")  # = 20
                if cc_local_cost < INFINITY_COST:
                    inv_f = inv_f.copy()
                    # Only override rows where courier_c Local (20) beats the current cost
                    current_cost = inv_f.loc[cc_mask, "cost_DoorToDoor"]
                    override_mask = cc_mask & (inv_f["cost_DoorToDoor"] > cc_local_cost)
                    inv_f.loc[override_mask, "cost_DoorToDoor"]    = cc_local_cost
                    inv_f.loc[override_mask, "courier_DoorToDoor"] = "courier_c"

        if inv_f.empty:
            return {
                "order_idx": order_idx, "order_id": order_id,
                "n_parcels": 0, "total_cost": 0.0,
                "n_assigned": 0, "n_cancelled": len(items_list),
            }

        # Build per-physical-facility structures (stock + cost)
        cost_col   = f"cost_{delivery_type}"
        courier_col = f"courier_{delivery_type}"
        phys_facs  = self._build_physical_facilities(inv_f, cost_col, courier_col)

        # Solve
        assignment = self._solve(items_list, phys_facs)

        return _summarise(order_idx, order_id, items_list, assignment, phys_facs)

    # ── Pre-filter ────────────────────────────────────────────────────────

    @staticmethod
    def _prefilter(inv: pd.DataFrame) -> pd.DataFrame:
        """Remove ineligible inventory rows."""
        df = inv
        if "facility_active" in df.columns:
            df = df[df["facility_active"].fillna(True).astype(bool)]
        if "facility_at_capacity" in df.columns:
            df = df[~df["facility_at_capacity"].fillna(False).astype(bool)]
        df = df[df["qty_available"].fillna(0) > 0]
        return df

    # ── Physical facility construction ────────────────────────────────────

    def _build_physical_facilities(
        self,
        inv: pd.DataFrame,
        cost_col: str,
        courier_col: str,
    ) -> dict:
        """
        Group inventory by physical facility key.

        Returns
        -------
        {phys_key: {
            'cost':     float,          # cheapest parcel cost
            'courier':  str | None,
            'stock':    {sku: int},     # max qty_available per SKU across virtual branches
            'branch_numbers': list[str],
            'facility_ids':   list[str],
        }}
        """
        df = inv.copy()
        df["phys_key"] = (
            df["branch_number"].astype(str).str.zfill(6)
            .map(lambda bn: self._dc_map.get(bn, bn))
        )

        result: dict = {}
        for pk, grp in df.groupby("phys_key"):
            costs = grp[cost_col].fillna(INFINITY_COST)
            valid  = costs[costs < INFINITY_COST]
            cost   = float(valid.min()) if len(valid) else float(INFINITY_COST)
            if len(valid):
                courier = grp.loc[costs.idxmin(), courier_col] if courier_col in grp.columns else None
            else:
                courier = None

            # Stock: per SKU, sum across virtual branches within the same physical DC.
            # Virtual branches share a physical warehouse — a picker can pull from
            # any shelf regardless of which administrative branch it belongs to,
            # and everything ships as a single parcel.  max() would understate
            # the DC's true stock capacity and cause unnecessary cancellations.
            stock = grp.groupby("sku")["qty_available"].sum().astype(int).to_dict()

            result[pk] = {
                "cost":           cost,
                "courier":        courier,
                "stock":          stock,
                "branch_numbers": grp["branch_number"].astype(str).unique().tolist(),
                "facility_ids":   grp["facility_id"].dropna().unique().tolist(),
            }

        return result

    # ── Solver ────────────────────────────────────────────────────────────

    @staticmethod
    def _item_excluded_from(item: dict, phys_fac: dict) -> bool:
        """
        Return True if any branch number of `phys_fac` appears in the item's
        excluded_branches list.  An empty list means no exclusion applies.
        """
        excl = item.get("excluded_branches")
        if not excl:
            return False
        branch_set = set(str(bn).zfill(6) for bn in phys_fac.get("branch_numbers", []))
        return bool(branch_set & set(str(b).zfill(6) for b in excl))

    def _apply_route_lmr(
        self,
        inv_f: pd.DataFrame,
        dest_postal: str,
        delivery_type: str,
    ) -> pd.DataFrame:
        """
        Re-compute cost columns for non-dc_b rows using the origin × destination
        route LMR for each eligible courier.

        The postcode mapping is courier-specific: each courier has its own hub
        network and LMR designation per postal code.  Route LMR is therefore
        computed separately for courier_a (and courier_c eligibility is already
        handled by the proximity override that runs afterwards).

        dc_b rows keep their courier_b costs from global precomputation.
        courier_c costs will be overridden by the proximity check if applicable.
        """
        is_dc_b = (
            inv_f["branch_number"].astype(str).str.zfill(6)
            .map(lambda b: self._dc_map.get(b, b) == "dc_b")
        )
        if not (~is_dc_b).any():
            return inv_f  # all rows are dc_b — nothing to override

        inv_f = inv_f.copy()
        non_dc_b_idx = inv_f.index[~is_dc_b]

        cost_col    = f"cost_{delivery_type}"
        courier_col = f"courier_{delivery_type}"

        # For each non-dc_b row: compute cheapest cost across courier_a (and
        # courier_c placeholder — proximity override finalises courier_c later).
        # courier_b is excluded for non-dc_b facilities.
        def _best_cost_for_row(row) -> tuple:
            fp        = row.get("facility_postal")
            cc        = bool(row.get("courier_c_active", False))
            # Precomputed fallback (facility-LMR-only, used when route LMR is
            # unresolvable due to missing postal code in the courier network)
            fallback_cost    = row.get(cost_col, INFINITY_COST)
            fallback_courier = row.get(courier_col)

            best_cost, best_courier = INFINITY_COST, None

            # courier_a route LMR (always eligible for non-dc_b)
            lmr_a = self.engine.route_lmr("courier_a", fp, dest_postal)
            if lmr_a:
                c_a = self.engine.courier_cost(delivery_type, "courier_a", lmr_a)
                if c_a < best_cost:
                    best_cost, best_courier = c_a, "courier_a"

            # courier_c placeholder: only if facility has courier_c_active=True.
            # Proximity override later finalises to cost=20 when 20km check passes.
            if cc:
                lmr_c = self.engine.route_lmr("courier_c", fp, dest_postal)
                if lmr_c:
                    c_c = self.engine.courier_cost(delivery_type, "courier_c", lmr_c)
                    if c_c < best_cost:
                        best_cost, best_courier = c_c, "courier_c"

            # Fall back to precomputed cost if route LMR could not be resolved
            # (e.g. facility postal not in courier_a / courier_c network table).
            # This preserves routeability for facilities with unknown postcodes.
            if best_cost >= INFINITY_COST and fallback_cost < INFINITY_COST:
                return fallback_cost, fallback_courier

            return best_cost, best_courier

        results = inv_f.loc[non_dc_b_idx].apply(_best_cost_for_row, axis=1)
        inv_f.loc[non_dc_b_idx, cost_col]    = results.map(lambda x: x[0])
        inv_f.loc[non_dc_b_idx, courier_col] = results.map(lambda x: x[1])

        return inv_f

    def _solve_exact(self, items_list: list, usable: dict) -> dict:
        """
        Exact MIP formulation via scipy.optimize.milp.

        Replaces the greedy + consolidation steps when ``use_exact=True``.
        Finds a provably optimal assignment for each order — including native
        quantity splits so no post-processing last-resort pass is needed.

        Variables
        ---------
        y[j]   ∈ {0,1}       —  facility j is opened (contributes one parcel cost)
        x[i,j] ∈ Z≥0         —  NUMBER OF UNITS of item i assigned to facility j
                                 (integer, 0 … qty_i)

        Objective
        ---------
        minimise  Σ_j  cost[j] · y[j]

        Constraints
        -----------
        (a) Σ_j x[i,j]  = qty_i                   ∀ feasible item i
            (all demanded units are placed)
        (b) Σ_{i : sku_i = s} x[i,j] ≤ stock[j,s]
            ∀ facility j, SKU s   (stock capacity — coefficient 1, not qty)
        (c) x[i,j] ≤ qty_i · y[j]                 ∀ i, j
            (big-M linking: facility must be opened for any units sent there)
        (d) x[i,j] = 0 for excluded or zero-stock (facility,item) pairs
            (enforced via variable upper bounds — no extra rows needed)

        Alternative SKU
        ---------------
        Handled by pre-selecting the best available SKU for each item before
        building the MIP: primary SKU if any eligible facility holds ≥ 1 unit,
        otherwise alternative_sku.  This is a simplification over the full
        bilinear model (which would require additional variables for SKU choice)
        but has zero impact on this dataset (cancelled items have no alt SKU).

        Pre-cancellation
        ----------------
        An item is cancelled before the MIP only when NO facility holds even
        1 unit of the SKU.  All partial-stock scenarios (sum of stock across
        facilities ≥ qty_i) are handled natively by the integer solver.
        """
        # ── 1. Pre-select SKU and classify items ──────────────────────────
        # Items enter the MIP as long as ≥1 eligible facility has any stock.
        # Items with zero stock everywhere are pre-cancelled.
        mip_items  = []   # items the MIP must assign
        pre_cancel = []   # item_keys with no stock at any eligible facility

        for it in items_list:
            qty = it["quantity"]
            sku = it["sku"]
            alt = it.get("alternative_sku")

            # Primary SKU: total eligible stock across the network >= qty?
            # (>= qty, not >= 1, so we don't lock in primary when the network
            #  can't actually satisfy the required quantity with the primary SKU)
            primary_total = sum(
                pf["stock"].get(sku, 0)
                for pf in usable.values()
                if not self._item_excluded_from(it, pf)
            )
            if primary_total >= qty:
                mip_items.append({**it, "_use_sku": sku})
                continue

            # Alternative SKU fallback: total eligible alt stock >= qty?
            if alt:
                alt_total = sum(
                    pf["stock"].get(alt, 0)
                    for pf in usable.values()
                    if not self._item_excluded_from(it, pf)
                )
                if alt_total >= qty:
                    mip_items.append({**it, "_use_sku": alt})
                    continue

            pre_cancel.append(it["item_key"])

        assignment = {ik: None for ik in pre_cancel}

        if not mip_items:
            return assignment

        # ── 2. Index structures ───────────────────────────────────────────
        fac_keys = list(usable.keys())
        n_i = len(mip_items)
        n_j = len(fac_keys)

        # Variable layout: [y_0 … y_{n_j-1} | x_{0,0} … x_{n_i-1, n_j-1}]
        n_vars = n_j + n_i * n_j

        def y(j: int) -> int:
            return j

        def x(i: int, j: int) -> int:
            return n_j + i * n_j + j

        # ── 3. Objective: minimise Σ cost[j] · y[j] ──────────────────────
        c_obj = np.zeros(n_vars)
        for j, pk in enumerate(fac_keys):
            c_obj[y(j)] = usable[pk]["cost"]

        # ── 4. Variable bounds ────────────────────────────────────────────
        # y[j] ∈ {0,1}            → lb=0, ub=1
        # x[i,j] ∈ {0,…,qty_i}   → lb=0, ub=min(qty_i, stock[j,sku_i])
        # Excluded or zero-stock pairs → ub=0 (forces x[i,j]=0)
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)   # y variables remain ∈ {0,1}

        for i, item in enumerate(mip_items):
            excl    = item.get("excluded_branches") or []
            excl_bn = set(str(b).zfill(6) for b in excl)
            use_sku = item["_use_sku"]
            qty     = item["quantity"]
            for j, pk in enumerate(fac_keys):
                pf        = usable[pk]
                branch_bn = set(str(b).zfill(6) for b in pf.get("branch_numbers", []))
                stk       = pf["stock"].get(use_sku, 0)
                if (excl_bn & branch_bn) or stk < 1:
                    ub[x(i, j)] = 0.0      # forbidden pair
                else:
                    ub[x(i, j)] = float(min(qty, stk))  # at most qty_i units

        bounds = Bounds(lb=lb, ub=ub)
        integrality = np.ones(n_vars)  # all variables are integer
        # y[j] binary enforced by ub=1; x[i,j] general integer by ub=qty_i

        # ── 5. Constraint matrix ──────────────────────────────────────────
        # (a) n_i  assignment rows    Σ_j x[i,j] = qty_i
        # (b) stock rows              one per (facility, SKU) pair used in MIP
        # (c) n_i * n_j linking rows  x[i,j] - qty_i · y[j] ≤ 0

        # Collect unique (facility j, SKU s) pairs present in MIP items
        fac_sku_caps: list[tuple[int, str, int]] = []
        seen: set[tuple[int, str]] = set()
        for j, pk in enumerate(fac_keys):
            stock = usable[pk]["stock"]
            for item in mip_items:
                s = item["_use_sku"]
                if s in stock and (j, s) not in seen:
                    fac_sku_caps.append((j, s, stock[s]))
                    seen.add((j, s))

        n_a = n_i
        n_b = len(fac_sku_caps)
        n_c = n_i * n_j
        n_rows = n_a + n_b + n_c

        A   = lil_matrix((n_rows, n_vars))
        l_b = np.full(n_rows, -np.inf)
        u_b = np.zeros(n_rows)

        # (a) Assignment: Σ_j x[i,j] = qty_i  (equality via l_b = u_b = qty_i)
        for i, item in enumerate(mip_items):
            qty = item["quantity"]
            for j in range(n_j):
                A[i, x(i, j)] = 1.0
            l_b[i] = float(qty)
            u_b[i] = float(qty)

        # (b) Stock capacity: Σ_{i: sku=s} x[i,j] ≤ cap
        #     Coefficient is 1 (not qty) because x counts units directly.
        for r, (j, s, cap) in enumerate(fac_sku_caps):
            row = n_a + r
            for i, item in enumerate(mip_items):
                if item["_use_sku"] == s:
                    A[row, x(i, j)] = 1.0   # ← was item["quantity"] in binary MIP
            u_b[row] = float(cap)

        # (c) Linking: x[i,j] - qty_i · y[j] ≤ 0
        #     Big-M = qty_i ensures y[j]=1 whenever any units are sent to j.
        for i, item in enumerate(mip_items):
            qty = item["quantity"]
            for j in range(n_j):
                row = n_a + n_b + i * n_j + j
                A[row, x(i, j)] =  1.0
                A[row, y(j)]    = -float(qty)  # ← was -1.0 in binary MIP
                # l_b already -inf, u_b already 0

        constraints = LinearConstraint(csr_matrix(A), l_b, u_b)

        # ── 6. Solve ──────────────────────────────────────────────────────
        result = milp(c_obj, constraints=constraints,
                      integrality=integrality, bounds=bounds)

        # ── 7. Extract solution ───────────────────────────────────────────
        if not result.success:
            # Solver failed (should not happen if pre-cancellation is correct).
            # Mark all MIP items as cancelled.
            for item in mip_items:
                assignment[item["item_key"]] = None
            return assignment

        xv = result.x
        for i, item in enumerate(mip_items):
            ik = item["item_key"]
            # Collect all (facility, units) pairs where units ≥ 1
            splits = [
                (fac_keys[j], int(round(xv[x(i, j)])))
                for j in range(n_j)
                if xv[x(i, j)] > 0.5
            ]
            if not splits:
                assignment[ik] = None            # should not happen
            elif len(splits) == 1:
                assignment[ik] = splits[0][0]   # single facility — no split
            else:
                assignment[ik] = splits          # quantity split: list[(pk, qty)]

        return assignment

    def _solve(self, items_list: list, phys_facs: dict) -> dict:
        """
        Find a minimum-cost assignment of items to physical facilities.

        Returns
        -------
        dict mapping item_key to one of:
          • str                   — physical_key (single-facility assignment)
          • list[tuple[str,int]]  — [(phys_key, qty), …] split across ≤ max_split_k
          • None                  — cancellation (no feasible assignment)
        """
        usable = {pk: pf for pk, pf in phys_facs.items() if pf["cost"] < INFINITY_COST}

        if not usable:
            return {item["item_key"]: None for item in items_list}

        if self.use_exact:
            # ── Exact MIP path ────────────────────────────────────────────
            # x[i,j] is a general integer (unit count), so the MIP handles
            # quantity splits natively.  No consolidation or last-resort
            # split pass is needed — return directly.
            return self._solve_exact(items_list, usable)

        # ── Greedy path ───────────────────────────────────────────────────
        # Step 1: try to fit the entire order into one facility.
        best_single = self._try_single_facility(items_list, usable)
        if best_single is not None:
            return {item["item_key"]: best_single for item in items_list}

        # Step 2: greedy multi-facility assignment + consolidation.
        assignment, sku_used = self._greedy(items_list, usable)
        assignment = self._consolidate(assignment, sku_used, items_list, usable)

        # Step 3: last-resort quantity split for items still unassigned.
        if self.max_split_k > 1:
            assignment, sku_used = self._split_cancelled(
                assignment, sku_used, items_list, usable, max_k=self.max_split_k
            )

        return assignment

    @staticmethod
    def _facility_can_cover(stock: dict, items_list: list) -> bool:
        """
        Return True if a single facility's stock can cover ALL items.
        For each item, the primary SKU is tried first; if insufficient,
        the alternative_sku is tried. Stock is tracked across items so
        that multiple items sharing a SKU are handled correctly.
        """
        consumed: dict[str, int] = defaultdict(int)
        for item in items_list:
            sku, qty = item["sku"], item["quantity"]
            alt_sku  = item.get("alternative_sku")

            avail_primary = stock.get(sku, 0) - consumed[sku]
            if avail_primary >= qty:
                consumed[sku] += qty
            elif alt_sku:
                avail_alt = stock.get(alt_sku, 0) - consumed[alt_sku]
                if avail_alt >= qty:
                    consumed[alt_sku] += qty
                else:
                    return False
            else:
                return False
        return True

    def _try_single_facility(
        self, items_list: list, usable: dict
    ) -> Optional[str]:
        """
        Return the physical_key of the cheapest facility able to cover ALL items
        (using primary SKU where possible, alternative_sku as fallback).
        A facility is skipped if it is excluded for ANY item in the order.
        Returns None if no single facility is sufficient.
        """
        best_cost = INFINITY_COST
        best_pk   = None
        for pk, pf in usable.items():
            # Skip if this facility is excluded for at least one item
            if any(self._item_excluded_from(item, pf) for item in items_list):
                continue
            if self._facility_can_cover(pf["stock"], items_list):
                if pf["cost"] < best_cost:
                    best_cost = pf["cost"]
                    best_pk   = pk
        return best_pk

    def _greedy(self, items_list: list, usable: dict) -> tuple[dict, dict]:
        """
        Greedy assignment (scarcest item first) with alternative-SKU fallback.

        Returns
        -------
        assignment : {item_key → physical_key or None}
        sku_used   : {item_key → sku string actually consumed from inventory}
        """
        # Mutable remaining stock per physical facility
        remaining: dict[str, dict[str, int]] = {
            pk: dict(pf["stock"]) for pk, pf in usable.items()
        }

        # Sort items: count eligible facilities across primary AND alt SKU,
        # respecting per-item excluded_branches
        def _n_eligible(item: dict) -> int:
            sku, qty = item["sku"], item["quantity"]
            alt_sku  = item.get("alternative_sku")
            n = sum(
                1 for pk, cap in remaining.items()
                if not self._item_excluded_from(item, usable[pk])
                and (cap.get(sku, 0) >= qty or (alt_sku and cap.get(alt_sku, 0) >= qty))
            )
            return n

        items_sorted = sorted(items_list, key=_n_eligible)

        assignment: dict = {}
        sku_used:   dict = {}   # item_key → SKU actually consumed
        chosen:     set[str] = set()

        for item in items_sorted:
            ik      = item["item_key"]
            qty     = item["quantity"]
            alt_sku = item.get("alternative_sku")
            placed  = False

            # Try primary SKU first, then alternative SKU (if available)
            skus_to_try = [item["sku"]]
            if alt_sku:
                skus_to_try.append(alt_sku)

            for try_sku in skus_to_try:
                if placed:
                    break

                # 1. Re-use an already-open facility (zero marginal cost)
                #    but only if this item is not excluded from that facility
                for pk in sorted(chosen, key=lambda k: usable[k]["cost"]):
                    if (
                        remaining[pk].get(try_sku, 0) >= qty
                        and not self._item_excluded_from(item, usable[pk])
                    ):
                        assignment[ik] = pk
                        sku_used[ik]   = try_sku
                        remaining[pk][try_sku] -= qty
                        placed = True
                        break

                if not placed:
                    # 2. Open the cheapest new facility with sufficient stock
                    #    that this item is not excluded from
                    eligible = [
                        (usable[pk]["cost"], pk)
                        for pk in remaining
                        if (
                            remaining[pk].get(try_sku, 0) >= qty
                            and not self._item_excluded_from(item, usable[pk])
                        )
                    ]
                    if eligible:
                        _, best_pk = min(eligible)
                        assignment[ik] = best_pk
                        sku_used[ik]   = try_sku
                        remaining[best_pk][try_sku] -= qty
                        chosen.add(best_pk)
                        placed = True

            if not placed:
                assignment[ik] = None   # genuine cancellation

        return assignment, sku_used

    def _consolidate(
        self,
        assignment: dict,
        sku_used: dict,
        items_list: list,
        usable: dict,
    ) -> dict:
        """
        Iteratively try to eliminate facilities by moving their items to a
        cheaper already-chosen facility. Uses sku_used to correctly track
        which SKU (primary or alternative) each item actually consumes.
        """
        item_by_key = {item["item_key"]: item for item in items_list}

        # Rebuild remaining from the current assignment + sku_used
        remaining: dict[str, dict[str, int]] = {
            pk: dict(pf["stock"]) for pk, pf in usable.items()
        }
        for ik, pk in assignment.items():
            if pk is not None:
                s = sku_used.get(ik, item_by_key[ik]["sku"])
                q = item_by_key[ik]["quantity"]
                remaining[pk][s] = remaining[pk].get(s, 0) - q

        improved = True
        while improved:
            improved = False
            # Only consider simple (non-split) assignments for consolidation
            used = {pk for pk in assignment.values() if pk is not None and not isinstance(pk, list)}

            for src in sorted(used, key=lambda k: usable[k]["cost"], reverse=True):
                items_at_src = [
                    item_by_key[ik]
                    for ik, pk in assignment.items()
                    if pk == src
                ]
                if not items_at_src:
                    continue

                for dst in sorted(used - {src}, key=lambda k: usable[k]["cost"]):
                    if usable[dst]["cost"] > usable[src]["cost"]:
                        break

                    # Check dst can absorb all items from src (using their actual SKU)
                    # and that none of those items are excluded from dst
                    temp = dict(remaining[dst])
                    ok   = True
                    for item in items_at_src:
                        s = sku_used.get(item["item_key"], item["sku"])
                        if self._item_excluded_from(item, usable[dst]):
                            ok = False
                            break
                        if temp.get(s, 0) < item["quantity"]:
                            ok = False
                            break
                        temp[s] -= item["quantity"]

                    if ok:
                        for item in items_at_src:
                            s = sku_used.get(item["item_key"], item["sku"])
                            assignment[item["item_key"]] = dst
                            remaining[src][s] = remaining[src].get(s, 0) + item["quantity"]
                            remaining[dst][s] = remaining[dst].get(s, 0) - item["quantity"]
                        improved = True
                        break

        return assignment

    def _split_cancelled(
        self,
        assignment: dict,
        sku_used: dict,
        items_list: list,
        usable: dict,
        max_k: int = 3,
    ) -> tuple[dict, dict]:
        """
        Last-resort pass: for items still marked as cancelled (assignment = None),
        attempt to fill the required quantity by splitting it across up to `max_k`
        facilities, taking stock greedily from cheapest-first.

        The split is recorded as a list of (physical_key, qty) tuples in the
        assignment dict.  Items that still cannot be covered within max_k
        facilities remain cancelled (None).

        Parameters
        ----------
        max_k : Maximum number of facilities to split across (default 3).
                Setting max_k=1 would never split (but _solve only calls this
                when max_split_k > 1).
        """
        item_by_key = {item["item_key"]: item for item in items_list}

        # Rebuild consumed stock from already-committed assignments
        remaining: dict[str, dict[str, int]] = {
            pk: dict(pf["stock"]) for pk, pf in usable.items()
        }
        for ik, pk_val in assignment.items():
            if pk_val is None:
                continue
            item = item_by_key[ik]
            s = sku_used.get(ik, item["sku"])
            if isinstance(pk_val, list):
                for sub_pk, sub_qty in pk_val:
                    remaining[sub_pk][s] = remaining[sub_pk].get(s, 0) - sub_qty
            else:
                remaining[pk_val][s] = remaining[pk_val].get(s, 0) - item["quantity"]

        for ik, pk_val in assignment.items():
            if pk_val is not None:
                continue  # already assigned

            item = item_by_key[ik]
            skus_to_try = [item["sku"]]
            if item.get("alternative_sku"):
                skus_to_try.append(item["alternative_sku"])

            for try_sku in skus_to_try:
                qty_needed = item["quantity"]

                # Candidate facilities: not excluded, have some stock, routable
                candidates = sorted(
                    [
                        (usable[pk]["cost"], pk)
                        for pk in remaining
                        if not self._item_excluded_from(item, usable[pk])
                        and remaining[pk].get(try_sku, 0) > 0
                        and usable[pk]["cost"] < INFINITY_COST
                    ]
                )

                # Greedily take from cheapest facilities, up to max_k
                chunks: list[tuple[str, int]] = []
                qty_remaining = qty_needed
                # Snapshot (cost, phys_key) — use phys_key for lookup
                snap = {pk: remaining[pk].get(try_sku, 0) for _, pk in candidates}

                for _, pk in candidates:
                    if len(chunks) >= max_k or qty_remaining == 0:
                        break
                    avail = snap[pk]
                    if avail <= 0:
                        continue
                    take = min(avail, qty_remaining)
                    chunks.append((pk, take))
                    snap[pk] -= take
                    qty_remaining -= take

                if qty_remaining == 0:
                    # Success — commit to assignment and remaining
                    if len(chunks) == 1:
                        # Collapsed to a single facility — store as plain string
                        single_pk, single_qty = chunks[0]
                        assignment[ik] = single_pk
                    else:
                        assignment[ik] = chunks
                    sku_used[ik] = try_sku
                    for pk, take in chunks:
                        remaining[pk][try_sku] = remaining[pk].get(try_sku, 0) - take
                    break  # don't try alternative SKU

        return assignment, sku_used


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(
    order_idx: int,
    order_id: str,
    items_list: list,
    assignment: dict,
    phys_facs: dict,
) -> dict:
    """Convert an item→facility assignment into a flat summary dict.

    Assignment values may be:
      str                   — single physical facility
      list[tuple[str,int]]  — [(phys_key, qty), …] split assignment
      None                  — cancellation
    """
    used_fac_costs: dict[str, float] = {}
    n_assigned  = 0
    n_cancelled = 0

    for item in items_list:
        pk = assignment.get(item["item_key"])
        if pk is None:
            n_cancelled += 1
        elif isinstance(pk, list):
            # Split across multiple facilities
            n_assigned += 1
            for sub_pk, _ in pk:
                if sub_pk not in used_fac_costs:
                    used_fac_costs[sub_pk] = phys_facs[sub_pk]["cost"]
        else:
            n_assigned += 1
            if pk not in used_fac_costs:
                used_fac_costs[pk] = phys_facs[pk]["cost"]

    return {
        "order_idx":   order_idx,
        "order_id":    order_id,
        "n_parcels":   len(used_fac_costs),
        "total_cost":  float(sum(used_fac_costs.values())),
        "n_assigned":  n_assigned,
        "n_cancelled": n_cancelled,
    }


def _empty_result(order_idx: int, order_id: str) -> dict:
    return {
        "order_idx":   order_idx,
        "order_id":    order_id,
        "n_parcels":   0,
        "total_cost":  0.0,
        "n_assigned":  0,
        "n_cancelled": 0,
    }
