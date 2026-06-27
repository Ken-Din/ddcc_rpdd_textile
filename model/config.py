"""
Central configuration for the DDCC/RPDD textile optimization model.
All domain parameters are defined here to avoid duplication across modules.
"""

# Re-entry positions (P) and disposal label
POSITIONS = [
    "Resale",
    "Repair",
    "Refurbishing",
    "Repackaging",
    "Recycling",
    "Donation",
    "DiscountSale",
]
DISPOSAL = "Disposal"

# Base capacities at I=100 (scaled proportionally for larger I)
BASE_CAPACITIES = {
    "Resale":       100,
    "Repair":         8,
    "Refurbishing":   6,
    "Repackaging":  100,
    "Recycling":      4,
    "Donation":       5,
    "DiscountSale": 100,
}

# Utility function parameters
AGE_DECAY_RATE = 0.001          # exp(-AGE_DECAY_RATE * age_days)
BRAND_MULTIPLIER = {
    "Premium":  1.5,
    "Standard": 1.0,
}

# Optimization parameters
COST_SCALE = 10_000             # Integer scaling for min-cost flow edge costs
DEFAULT_W1 = 0.7                # Default weight for residual value
DEFAULT_W2 = 0.3                # Default weight for recovery cost
DEFAULT_EPSILON = 0.01          # Tie-breaking penalty for non-disposal positions