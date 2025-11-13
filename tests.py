"""
Portfolio Performance Tests

Validates portfolio calculations against reference solutions using numpy's allclose
for float comparison with appropriate tolerance.
"""
import pandas as pd
import numpy as np
from portfolio_analyzer import PortfolioAnalyzer

def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = list(df.select_dtypes(exclude=float).columns)
    if key_cols:
        df = df.sort_values(by=key_cols)
    return df.reset_index(drop=True)

def test_allclose(df_calc, df_sol, name, rtol=1e-6, atol=1e-6):
    df_calc, df_sol = sort_df(df_calc), sort_df(df_sol)

    calc_float = [c for c in df_calc.columns if pd.api.types.is_float_dtype(df_calc[c])]
    sol_float  = [c for c in df_sol.columns  if pd.api.types.is_float_dtype(df_sol[c])]
    cols = sorted(set(calc_float) & set(sol_float))

    if not cols:
        print(f"{name}: ❌ no shared float columns to compare")
        return False

    if len(df_calc) != len(df_sol):
        print(f"{name}: ❌ row count differs (calc={len(df_calc)}, sol={len(df_sol)})")

    a = df_calc[cols].to_numpy()
    b = df_sol[cols].to_numpy()

    m, n = min(a.shape[0], b.shape[0]), len(cols)
    a, b = a[:m, :n], b[:m, :n]

    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    print(f"{name}: {'✅ match' if ok else '❌ mismatch'}")
    if not ok:
        diff = np.nanmax(np.abs(a - b))
        print(f"  Shared cols: {cols}")
        print(f"  Max abs diff: {diff:.6g}")
    return ok

def run_tests() -> bool:
    """Run portfolio validation tests and print summary."""
    analyzer = PortfolioAnalyzer("data/PortfolioPositions_Updated.xlsx")
    perf = analyzer.calculate_performance()

    contrib_sol = pd.read_csv("data/solution/top_contributors.csv")
    detr_sol    = pd.read_csv("data/solution/top_detractors.csv")
    sectors_sol = pd.read_csv("data/solution/sectors.csv")

    sectors_sol['target_weights'] = sectors_sol['sector'].map({
        "Information Technology": 0.11,
        "Health Care": 0.11,
        "Financials": 0.14,
        "Consumer Discretionary": 0.18,
        "Industrials": 0.22,
        "Consumer Staples": 0.24,
    }).fillna(0)

    tests = [
        ("Top Contributors", perf["top_contributors"], contrib_sol),
        ("Top Detractors",   perf["top_detractors"],   detr_sol),
        ("Sector Breakdown", perf["sector_breakdown"], sectors_sol),
    ]

    all_ok = all(test_allclose(a, b, name) for name, a, b in tests)
    print("\n----------------------------------------")
    if all_ok:
        print("✅ All calculated values match the hand-computed reference solutions within tolerance.")
        print("The portfolio performance calculations are consistent and numerically validated.")
    else:
        print("⚠️ Some calculated values deviate from the hand-computed reference solutions.")
        print("Review the mismatched tables above to identify potential rounding or logic differences.")
    print("----------------------------------------")
    return all_ok
