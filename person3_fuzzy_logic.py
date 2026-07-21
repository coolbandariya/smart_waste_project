import numpy as np


def create_fuzzy_system():
    """Stub function to maintain full backwards compatibility with app.py imports."""
    return None


def calculate_priority(
    fill_pct, hours_elapsed, criticality_str, fuzzy_sim=None
):
    """Computes a Fuzzy-Logic priority score (0-100) using continuous membership curves.

    No external skfuzzy or networkx dependencies required!
    """
    fill_pct = min(max(float(fill_pct), 0.0), 100.0)
    hours_elapsed = min(max(float(hours_elapsed), 0.0), 72.0)

    # 1. Fill Level Weight (Sigmoidal curve for smooth transition above 60%)
    fill_weight = 1.0 / (1.0 + np.exp(-0.1 * (fill_pct - 65.0)))

    # 2. Time Elapsed Weight (Piecewise progression)
    time_weight = min(hours_elapsed / 48.0, 1.0)

    # 3. Criticality Multiplier
    crit_map = {"Low": 0.8, "Medium": 1.0, "High": 1.3}
    crit_mult = crit_map.get(criticality_str, 1.0)

    # Calculate aggregate priority score
    base_score = (fill_weight * 70.0) + (time_weight * 30.0)
    final_score = base_score * crit_mult

    # Clamp output between 0 and 100
    return round(float(np.clip(final_score, 0.0, 100.0)), 2)


if __name__ == "__main__":
    print("Testing Updated Math Priority Engine...")
    p1 = calculate_priority(88.5, 30, "High")
    p2 = calculate_priority(20.0, 5, "Low")
    print(f"Test 1 (88.5% fill, High crit) -> Priority Score: {p1}/100")
    print(f"Test 2 (20.0% fill, Low crit)  -> Priority Score: {p2}/100")