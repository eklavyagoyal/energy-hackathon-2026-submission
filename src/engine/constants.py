"""Canonical engine constants, frozen by docs/03-phase1-engine.md
section 3. Copy of the doc table; never tune silently (log changes in
docs/decisions.md).

Every limit is expressed in per-unit or percent, which is what keeps
them valid across voltage levels and dataset swaps (see
docs/battery-feature.md, dataset portability).
"""

# Bus voltage limits, per-unit. Apply at every voltage level.
VOLTAGE_BAND_LOW = 0.95
VOLTAGE_BAND_HIGH = 1.05
# Comparison tolerance so a bus sitting exactly on a limit (e.g. a bus
# pinned to a generator setpoint at the band edge) is not flagged a
# violation by floating-point noise. Used everywhere the band is tested.
VOLTAGE_TOL = 1e-6

# loading_percent above this = violation.
OVERLOAD_LIMIT = 100.0

# loading_percent above this = element trips in the cascade loop
# (proxy for overcurrent / emergency-rating relays; configurable).
TRIP_THRESHOLD = 120.0

# Cascade iteration cap; hitting it sets diverged = true.
MAX_CASCADE_ITERS = 20

# Tightened line limit used by the OPF tighten-and-verify loop
# (remediation is out of scope on this branch; kept for parity with the
# doc table).
SECURITY_MARGIN_PCT = 85.0

# Contingencies promoted from screener to full AC cascade analysis
# (the screener slot is out of scope on this branch; kept for parity).
FULL_AC_TOP_K = 15

# All stochastic choices deterministic.
SEED = 42

# Load-scale increment during stress ramp.
RAMP_STEP = 0.05

# Termination guarantee for the scenario ramp.
SCALE_CAP = 3.0

# FULL_BLACKOUT pins CSS to exactly this value.
CSS_MAX = 2000.0

# Synthetic emergency-rating calibration for case118 (logged as D59).
# pandapower's case118 ships 9900 MVA placeholder ratings (max base
# loading under 5 percent), so no load scale can manufacture
# congestion. Case118Loader assigns per-line ratings proportional to
# base-case flow (Motter-Lai 2002 style: capacity = base flow divided
# by the target loading), floored at a distribution quantile so
# near-zero-flow lines do not become hair triggers. Real TSO datasets
# carry real ratings and never pass through this path.
RATING_BASE_LOADING_PCT = 50.0
RATING_FLOOR_QUANTILE = 0.25
