"""
Financial calculation engine for the Fee & Wealth Drag Retirement Analyzer.
Calculates nominal and inflation-adjusted balances, direct fee deductions,
lost compounding (opportunity cost), wealth drag, and break-even alpha.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    key: str
    name: str
    fee_rate: float  # e.g., 0.0003 for 0.03%
    description: str
    category: str


@dataclass
class ScenarioResult:
    key: str
    name: str
    fee_rate: float
    description: str
    monthly_df: pd.DataFrame
    yearly_df: pd.DataFrame
    ending_balance_nominal: float
    ending_balance_real: float
    total_contributions_nominal: float
    total_contributions_real: float
    total_fees_nominal: float
    total_fees_real: float
    total_net_growth_nominal: float
    total_net_growth_real: float
    gross_ending_balance_nominal: float
    gross_ending_balance_real: float
    wealth_drag_vs_gross_nominal: float
    wealth_drag_vs_gross_real: float
    lost_compounding_nominal: float
    lost_compounding_real: float
    # Relative to low-cost baseline (Scenario 1)
    wealth_drag_vs_baseline_nominal: float = 0.0
    wealth_drag_vs_baseline_real: float = 0.0
    drag_pct_vs_baseline: float = 0.0
    extra_fees_vs_baseline_nominal: float = 0.0
    extra_lost_compounding_vs_baseline_nominal: float = 0.0
    break_even_alpha_pct: float = 0.0
    break_even_alpha_bps: float = 0.0


def calculate_scenario_trajectory(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    gross_annual_return: float,
    inflation_rate: float,
    fee_rate: float,
    timing: str = "end",
) -> pd.DataFrame:
    """
    Simulates monthly portfolio evolution for a given fee rate.

    Parameters:
    -----------
    initial_investment: Starting principal ($)
    monthly_contribution: Regular monthly addition ($)
    years: Investment horizon (years)
    gross_annual_return: Annualized gross asset return (e.g., 0.08 for 8%)
    inflation_rate: Annualized inflation rate (e.g., 0.025 for 2.5%)
    fee_rate: Total annualized fee rate / expense ratio (e.g., 0.0075 for 0.75%)
    timing: 'end' (default) or 'start' of monthly deposit

    Returns:
    --------
    pd.DataFrame with monthly records: balance, fees, compounding, real values.
    """
    months = int(years * 12)
    r_monthly_gross = gross_annual_return / 12.0
    fee_monthly = fee_rate / 12.0
    # Effective net monthly rate
    r_monthly_net = r_monthly_gross - fee_monthly

    # Arrays to store values for maximum speed and numeric stability
    balance_nominal = np.zeros(months + 1)
    cum_contributions = np.zeros(months + 1)
    monthly_fees = np.zeros(months + 1)
    cum_fees_nominal = np.zeros(months + 1)
    gross_balance_nominal = np.zeros(months + 1)

    # Initial state at month 0
    balance_nominal[0] = initial_investment
    cum_contributions[0] = initial_investment
    gross_balance_nominal[0] = initial_investment
    monthly_fees[0] = 0.0
    cum_fees_nominal[0] = 0.0

    curr_bal = float(initial_investment)
    curr_gross_bal = float(initial_investment)
    curr_cum_contrib = float(initial_investment)
    curr_cum_fees = 0.0

    for m in range(1, months + 1):
        if timing == "start":
            # Deposit made at start of month, participates in growth
            curr_bal += monthly_contribution
            curr_gross_bal += monthly_contribution
            curr_cum_contrib += monthly_contribution

            gross_growth = curr_bal * r_monthly_gross
            # Direct fee deducted on the portfolio balance
            fee = curr_bal * fee_monthly
            curr_bal = curr_bal + gross_growth - fee

            # Gross benchmark (0% fee)
            curr_gross_bal = curr_gross_bal * (1.0 + r_monthly_gross)
        else:
            # End of month deposit (standard paycheck 401k/IRA model)
            gross_growth = curr_bal * r_monthly_gross
            fee = curr_bal * fee_monthly
            curr_bal = curr_bal + gross_growth - fee + monthly_contribution

            # Gross benchmark (0% fee)
            curr_gross_bal = curr_gross_bal * (1.0 + r_monthly_gross) + monthly_contribution
            curr_cum_contrib += monthly_contribution

        curr_cum_fees += fee

        balance_nominal[m] = curr_bal
        cum_contributions[m] = curr_cum_contrib
        monthly_fees[m] = fee
        cum_fees_nominal[m] = curr_cum_fees
        gross_balance_nominal[m] = curr_gross_bal

    months_arr = np.arange(months + 1)
    years_arr = months_arr / 12.0

    # Inflation factor: (1 + inflation_rate) ** (m / 12)
    inflation_factors = (1.0 + inflation_rate) ** years_arr

    balance_real = balance_nominal / inflation_factors
    gross_balance_real = gross_balance_nominal / inflation_factors
    cum_contributions_real = cum_contributions / inflation_factors

    # Monthly fee real (discounted to present purchasing power)
    monthly_fees_real = monthly_fees / inflation_factors
    cum_fees_real = np.cumsum(monthly_fees_real)

    # Opportunity cost / lost compounding vs gross benchmark
    wealth_drag_vs_gross_nom = gross_balance_nominal - balance_nominal
    lost_compounding_nom = np.maximum(0.0, wealth_drag_vs_gross_nom - cum_fees_nominal)

    wealth_drag_vs_gross_real = gross_balance_real - balance_real
    lost_compounding_real = np.maximum(0.0, wealth_drag_vs_gross_real - cum_fees_real)

    df = pd.DataFrame(
        {
            "month": months_arr,
            "year": years_arr,
            "balance_nominal": balance_nominal,
            "balance_real": balance_real,
            "cum_contributions_nominal": cum_contributions,
            "cum_contributions_real": cum_contributions_real,
            "monthly_fee_nominal": monthly_fees,
            "monthly_fee_real": monthly_fees_real,
            "cum_fees_nominal": cum_fees_nominal,
            "cum_fees_real": cum_fees_real,
            "gross_balance_nominal": gross_balance_nominal,
            "gross_balance_real": gross_balance_real,
            "wealth_drag_vs_gross_nominal": wealth_drag_vs_gross_nom,
            "wealth_drag_vs_gross_real": wealth_drag_vs_gross_real,
            "lost_compounding_nominal": lost_compounding_nom,
            "lost_compounding_real": lost_compounding_real,
        }
    )

    return df


def aggregate_yearly_data(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts annual snapshot records from monthly simulation data.
    """
    # Select month 0, 12, 24, ...
    yearly_mask = monthly_df["month"] % 12 == 0
    yearly_df = monthly_df[yearly_mask].copy()
    yearly_df["year"] = yearly_df["year"].astype(int)

    # Annual fees paid in each year
    # Using groupby on ceil(month/12)
    monthly_df_working = monthly_df.copy()
    monthly_df_working["calendar_year"] = np.ceil(monthly_df_working["month"] / 12.0).astype(int)
    
    annual_fees = (
        monthly_df_working[monthly_df_working["month"] > 0]
        .groupby("calendar_year")["monthly_fee_nominal"]
        .sum()
        .reset_index()
        .rename(columns={"calendar_year": "year", "monthly_fee_nominal": "annual_fee_nominal"})
    )

    annual_fees_real = (
        monthly_df_working[monthly_df_working["month"] > 0]
        .groupby("calendar_year")["monthly_fee_real"]
        .sum()
        .reset_index()
        .rename(columns={"calendar_year": "year", "monthly_fee_real": "annual_fee_real"})
    )

    yearly_df = yearly_df.merge(annual_fees, on="year", how="left").fillna(0.0)
    yearly_df = yearly_df.merge(annual_fees_real, on="year", how="left").fillna(0.0)

    return yearly_df


def run_comparative_simulation(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    gross_annual_return: float,
    inflation_rate: float,
    scenarios: List[ScenarioConfig],
    baseline_key: str = "index_etf",
    timing: str = "end",
) -> Dict[str, ScenarioResult]:
    """
    Simulates all scenarios and computes cross-comparison metrics against the baseline.
    """
    results: Dict[str, ScenarioResult] = {}

    # Run individual scenario simulations
    for sc in scenarios:
        m_df = calculate_scenario_trajectory(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            gross_annual_return=gross_annual_return,
            inflation_rate=inflation_rate,
            fee_rate=sc.fee_rate,
            timing=timing,
        )
        y_df = aggregate_yearly_data(m_df)

        last_row = m_df.iloc[-1]
        end_nom = float(last_row["balance_nominal"])
        end_real = float(last_row["balance_real"])
        total_contrib_nom = float(last_row["cum_contributions_nominal"])
        total_contrib_real = float(last_row["cum_contributions_real"])
        total_fees_nom = float(last_row["cum_fees_nominal"])
        total_fees_real = float(last_row["cum_fees_real"])
        gross_end_nom = float(last_row["gross_balance_nominal"])
        gross_end_real = float(last_row["gross_balance_real"])
        drag_gross_nom = float(last_row["wealth_drag_vs_gross_nominal"])
        drag_gross_real = float(last_row["wealth_drag_vs_gross_real"])
        lost_comp_nom = float(last_row["lost_compounding_nominal"])
        lost_comp_real = float(last_row["lost_compounding_real"])

        res = ScenarioResult(
            key=sc.key,
            name=sc.name,
            fee_rate=sc.fee_rate,
            description=sc.description,
            monthly_df=m_df,
            yearly_df=y_df,
            ending_balance_nominal=end_nom,
            ending_balance_real=end_real,
            total_contributions_nominal=total_contrib_nom,
            total_contributions_real=total_contrib_real,
            total_fees_nominal=total_fees_nom,
            total_fees_real=total_fees_real,
            total_net_growth_nominal=max(0.0, end_nom - total_contrib_nom),
            total_net_growth_real=max(0.0, end_real - total_contrib_real),
            gross_ending_balance_nominal=gross_end_nom,
            gross_ending_balance_real=gross_end_real,
            wealth_drag_vs_gross_nominal=drag_gross_nom,
            wealth_drag_vs_gross_real=drag_gross_real,
            lost_compounding_nominal=lost_comp_nom,
            lost_compounding_real=lost_comp_real,
        )
        results[sc.key] = res

    # Compute comparative metrics against baseline scenario
    if baseline_key in results:
        base_res = results[baseline_key]
        base_end_nom = base_res.ending_balance_nominal
        base_end_real = base_res.ending_balance_real
        base_fee_rate = base_res.fee_rate

        for key, res in results.items():
            # Wealth Drag vs Baseline ETF
            drag_nom = base_end_nom - res.ending_balance_nominal
            drag_real = base_end_real - res.ending_balance_real
            drag_pct = (drag_nom / base_end_nom * 100.0) if base_end_nom > 0 else 0.0

            extra_fees = res.total_fees_nominal - base_res.total_fees_nominal
            extra_lost_comp = drag_nom - extra_fees

            # Break-Even Alpha Required to match Low-Cost Index ETF:
            # Alpha is the excess gross return the strategy must achieve annually
            # so that (Gross + Alpha - Fee_k) == (Gross - Fee_baseline).
            # Exact identity: Alpha = Fee_k - Fee_baseline
            alpha_pct = (res.fee_rate - base_fee_rate) * 100.0
            alpha_bps = alpha_pct * 100.0

            res.wealth_drag_vs_baseline_nominal = max(0.0, drag_nom)
            res.wealth_drag_vs_baseline_real = max(0.0, drag_real)
            res.drag_pct_vs_baseline = max(0.0, drag_pct)
            res.extra_fees_vs_baseline_nominal = max(0.0, extra_fees)
            res.extra_lost_compounding_vs_baseline_nominal = max(0.0, extra_lost_comp)
            res.break_even_alpha_pct = max(0.0, alpha_pct)
            res.break_even_alpha_bps = max(0.0, alpha_bps)

    return results


def build_comparison_dataframe(
    results: Dict[str, ScenarioResult],
    baseline_key: str = "index_etf",
) -> pd.DataFrame:
    """
    Builds a consolidated year-by-year comparison DataFrame suitable for UI display and CSV export.
    """
    years = results[baseline_key].yearly_df["year"].values
    comp_df = pd.DataFrame({"Year": years})

    # Principal contributions
    comp_df["Total Principal ($)"] = results[baseline_key].yearly_df["cum_contributions_nominal"].round(2)

    for key, res in results.items():
        prefix = res.name
        y_df = res.yearly_df
        comp_df[f"{prefix} Ending Balance (Nominal)"] = y_df["balance_nominal"].round(2)
        comp_df[f"{prefix} Ending Balance (Real)"] = y_df["balance_real"].round(2)
        comp_df[f"{prefix} Annual Fee Paid"] = y_df["annual_fee_nominal"].round(2)
        comp_df[f"{prefix} Cum Fees Paid"] = y_df["cum_fees_nominal"].round(2)

        if key != baseline_key:
            # Calculate yearly drag vs baseline
            base_y_df = results[baseline_key].yearly_df
            yearly_drag = (base_y_df["balance_nominal"] - y_df["balance_nominal"]).round(2)
            comp_df[f"{prefix} Drag vs ETF ($)"] = yearly_drag

    return comp_df
