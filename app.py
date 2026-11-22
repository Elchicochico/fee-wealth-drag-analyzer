"""
Fee & Wealth Drag Retirement Analyzer
A Streamlit financial tool analyzing the true lifetime cost of investment fees,
wealth drag, lost compounding, and break-even alpha requirements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from analyzer import (
    ScenarioConfig,
    run_comparative_simulation,
    build_comparison_dataframe,
)
from charts import (
    build_trajectory_chart,
    build_wealth_breakdown_bar_chart,
    build_drag_gap_timeline_chart,
    build_alpha_hurdle_chart,
)

# Page configuration
st.set_page_config(
    page_title="Fee & Wealth Drag Retirement Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished financial dashboard styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-card.etf-card {
        border-top: 4px solid #10B981;
    }
    .metric-card.fund-card {
        border-top: 4px solid #F59E0B;
    }
    .metric-card.advisor-card {
        border-top: 4px solid #EF4444;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 16px;
    }
    .badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
    }
    .badge-green { background: #D1FAE5; color: #065F46; }
    .badge-amber { background: #FEF3C7; color: #92400E; }
    .badge-red { background: #FEE2E2; color: #991B1B; }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 7px 0;
        border-bottom: 1px dashed #F3F4F6;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #4B5563;
    }
    .stat-val {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
    }
    .stat-val-highlight-red {
        color: #DC2626;
        font-weight: 700;
    }
    .stat-val-highlight-green {
        color: #059669;
        font-weight: 700;
    }
    .hero-stat {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 12px;
        text-align: center;
    }
    .hero-val {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280;
        font-weight: 600;
    }
    .takeaway-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .takeaway-banner h2 {
        color: white;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .takeaway-banner p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chart and dataframe helper for Streamlit version compatibility
def render_altair(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

def render_dataframe(df_styled, height=400):
    try:
        st.dataframe(df_styled, width="stretch", height=height)
    except TypeError:
        st.dataframe(df_styled, use_container_width=True, height=height)


# Helper formatting
def fmt_curr(val: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"${val:,.0f}"
    return f"${val:,.{decimals}f}"


def fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}%"


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Simulation Inputs")

with st.sidebar.expander("💼 Portfolio Parameters", expanded=True):
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1_000.0,
        help="Starting lump-sum investment in your portfolio.",
    )

    monthly_contribution = st.number_input(
        "Monthly Contribution ($)",
        min_value=0.0,
        max_value=500_000.0,
        value=500.0,
        step=50.0,
        help="Regular deposit contributed each month.",
    )

    years = st.slider(
        "Investment Horizon (Years)",
        min_value=1,
        max_value=50,
        value=30,
        help="Total number of years until retirement or liquidation.",
    )

    gross_return = (
        st.slider(
            "Expected Gross Annual Return (%)",
            min_value=1.0,
            max_value=15.0,
            value=8.0,
            step=0.1,
            help="Pre-fee annualized return expected from the market (e.g., S&P 500 historical average ~10% nominal, ~7-8% conservative).",
        )
        / 100.0
    )

    inflation_rate = (
        st.slider(
            "Annual Inflation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Expected annual CPI inflation to compute real purchasing power.",
        )
        / 100.0
    )

with st.sidebar.expander("💸 Scenario Fee Rates", expanded=True):
    st.caption("Customize the annual expense ratios or fee stacks:")

    etf_fee_pct = st.number_input(
        "(1) Low-Cost Index ETF (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.03,
        step=0.01,
        format="%.3f",
        help="Broad market index ETF expense ratio (e.g., Vanguard VOO/VTI @ 0.03%).",
    )

    mutual_fund_fee_pct = st.number_input(
        "(2) High-Fee Mutual Fund (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.75,
        step=0.05,
        format="%.2f",
        help="Typical actively managed equity mutual fund expense ratio.",
    )

    advisor_fee_pct = st.number_input(
        "(3) Advisor AUM + Fund Stack (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.50,
        step=0.10,
        format="%.2f",
        help="Typical wealth manager stack: e.g., 1.00% AUM advisor fee + 0.50% underlying fund expense ratio.",
    )

with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
    deposit_timing = st.selectbox(
        "Monthly Contribution Timing",
        options=["End of Month (Paycheck)", "Start of Month"],
        index=0,
        help="Whether deposits are added before or after monthly compounding.",
    )
    timing_arg = "end" if "End" in deposit_timing else "start"

    show_gross_benchmark = st.checkbox(
        "Show 0% Gross Benchmark in Charts",
        value=True,
        help="Overlay a theoretical 0% fee scenario to reveal the complete gross wealth potential.",
    )

    show_principal_baseline = st.checkbox(
        "Show Cumulative Principal in Charts",
        value=True,
        help="Overlay the un-invested cash contributions baseline.",
    )

# Quick Presets
st.sidebar.markdown("---")
st.sidebar.caption("💡 Quick Scenarios")
col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("Standard", use_container_width=True, help="Index 0.03%, Fund 0.75%, Advisor 1.50%"):
    st.rerun()

# Currency / View Toggles
st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "📊 Value Display Mode",
    options=["Nominal (Future Dollars)", "Real (Inflation-Adjusted Purchasing Power)"],
    index=0,
    help="Nominal shows nominal dollar values in future terms. Real deflates values using your chosen inflation rate to show today's purchasing power.",
)
is_real = "Real" in view_mode


# ==========================================
# SIMULATION EXECUTION
# ==========================================
scenarios = [
    ScenarioConfig(
        key="index_etf",
        name="Low-Cost Index ETF",
        fee_rate=etf_fee_pct / 100.0,
        description="Passive index fund tracking broad market (e.g. VOO / VTI)",
        category="Passive",
    ),
    ScenarioConfig(
        key="mutual_fund",
        name="High-Fee Mutual Fund",
        fee_rate=mutual_fund_fee_pct / 100.0,
        description="Actively managed equity mutual fund",
        category="Active",
    ),
    ScenarioConfig(
        key="advisor_stack",
        name="Advisor AUM + Fund Stack",
        fee_rate=advisor_fee_pct / 100.0,
        description="1.0% Wealth Advisor AUM fee + 0.5% underlying fund fees",
        category="Advisor Stack",
    ),
]

simulation_results = run_comparative_simulation(
    initial_investment=initial_investment,
    monthly_contribution=monthly_contribution,
    years=years,
    gross_annual_return=gross_return,
    inflation_rate=inflation_rate,
    scenarios=scenarios,
    baseline_key="index_etf",
    timing=timing_arg,
)

res_etf = simulation_results["index_etf"]
res_fund = simulation_results["mutual_fund"]
res_adv = simulation_results["advisor_stack"]


# ==========================================
# HERO BANNER & HEADLINE INSIGHT
# ==========================================
adv_drag_nom = res_adv.wealth_drag_vs_baseline_nominal
adv_drag_real = res_adv.wealth_drag_vs_baseline_real
adv_pct_lost = res_adv.drag_pct_vs_baseline
display_drag = adv_drag_real if is_real else adv_drag_nom

st.markdown(
    f"""
    <div class="takeaway-banner">
        <h2>Fee & Wealth Drag Retirement Analyzer</h2>
        <p>
            A <strong>{advisor_fee_pct:.2f}%</strong> annual advisor fee doesn't just reduce returns by {advisor_fee_pct:.2f}%—over 
            <strong>{years} years</strong>, it drains <strong>{fmt_curr(display_drag)} ({adv_pct_lost:.1f}%)</strong> 
            of your final retirement wealth compared to a low-cost index ETF.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# SIDE-BY-SIDE SCENARIO COMPARISON CARDS
# ==========================================
st.subheader("Side-by-Side Scenario Breakdown")

col1, col2, col3 = st.columns(3)

# ----------------- SCENARIO 1 -----------------
with col1:
    end_val = res_etf.ending_balance_real if is_real else res_etf.ending_balance_nominal
    fees_paid = res_etf.total_fees_real if is_real else res_etf.total_fees_nominal
    lost_comp = res_etf.lost_compounding_real if is_real else res_etf.lost_compounding_nominal
    total_drag_gross = res_etf.wealth_drag_vs_gross_real if is_real else res_etf.wealth_drag_vs_gross_nominal

    st.markdown(
        f"""
        <div class="metric-card etf-card">
            <div class="card-title">
                <span>Low-Cost Index ETF</span>
                <span class="badge badge-green">{etf_fee_pct:.2f}% ER</span>
            </div>
            <div class="card-subtitle">Passive broad-market index tracking</div>
            
            <div class="hero-stat">
                <div class="hero-val" style="color: #059669;">{fmt_curr(end_val)}</div>
                <div class="hero-label">Ending Balance ({'Real' if is_real else 'Nominal'})</div>
            </div>

            <div class="stat-row">
                <span class="stat-label">Drag vs Baseline ETF:</span>
                <span class="stat-val stat-val-highlight-green">$0 (Baseline)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Direct Fees Paid:</span>
                <span class="stat-val">{fmt_curr(fees_paid)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Lost Compounding Cost:</span>
                <span class="stat-val">{fmt_curr(lost_comp)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Wealth Drag vs Gross:</span>
                <span class="stat-val">{fmt_curr(total_drag_gross)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Break-Even Alpha Required:</span>
                <span class="stat-val">0.00% (0 bps)</span>
            </div>
            <div class="stat-row" style="border-bottom: none;">
                <span class="stat-label">Wealth Retained vs Potential:</span>
                <span class="stat-val stat-val-highlight-green">
                    {(res_etf.ending_balance_nominal / res_etf.gross_ending_balance_nominal * 100):.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- SCENARIO 2 -----------------
with col2:
    end_val2 = res_fund.ending_balance_real if is_real else res_fund.ending_balance_nominal
    fees_paid2 = res_fund.total_fees_real if is_real else res_fund.total_fees_nominal
    lost_comp2 = res_fund.lost_compounding_real if is_real else res_fund.lost_compounding_nominal
    drag_vs_etf2 = res_fund.wealth_drag_vs_baseline_real if is_real else res_fund.wealth_drag_vs_baseline_nominal
    total_drag_gross2 = res_fund.wealth_drag_vs_gross_real if is_real else res_fund.wealth_drag_vs_gross_nominal

    st.markdown(
        f"""
        <div class="metric-card fund-card">
            <div class="card-title">
                <span>High-Fee Mutual Fund</span>
                <span class="badge badge-amber">{mutual_fund_fee_pct:.2f}% ER</span>
            </div>
            <div class="card-subtitle">Actively managed stock mutual fund</div>
            
            <div class="hero-stat">
                <div class="hero-val" style="color: #D97706;">{fmt_curr(end_val2)}</div>
                <div class="hero-label">Ending Balance ({'Real' if is_real else 'Nominal'})</div>
            </div>

            <div class="stat-row">
                <span class="stat-label">Drag vs Low-Cost ETF:</span>
                <span class="stat-val stat-val-highlight-red">-{fmt_curr(drag_vs_etf2)} (-{res_fund.drag_pct_vs_baseline:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Direct Fees Paid:</span>
                <span class="stat-val">{fmt_curr(fees_paid2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Lost Compounding Cost:</span>
                <span class="stat-val">{fmt_curr(lost_comp2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Wealth Drag vs Gross:</span>
                <span class="stat-val">{fmt_curr(total_drag_gross2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Break-Even Alpha Required:</span>
                <span class="stat-val stat-val-highlight-red">+{res_fund.break_even_alpha_pct:.2f}% (+{res_fund.break_even_alpha_bps:.0f} bps)</span>
            </div>
            <div class="stat-row" style="border-bottom: none;">
                <span class="stat-label">Wealth Retained vs Potential:</span>
                <span class="stat-val">
                    {(res_fund.ending_balance_nominal / res_fund.gross_ending_balance_nominal * 100):.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- SCENARIO 3 -----------------
with col3:
    end_val3 = res_adv.ending_balance_real if is_real else res_adv.ending_balance_nominal
    fees_paid3 = res_adv.total_fees_real if is_real else res_adv.total_fees_nominal
    lost_comp3 = res_adv.lost_compounding_real if is_real else res_adv.lost_compounding_nominal
    drag_vs_etf3 = res_adv.wealth_drag_vs_baseline_real if is_real else res_adv.wealth_drag_vs_baseline_nominal
    total_drag_gross3 = res_adv.wealth_drag_vs_gross_real if is_real else res_adv.wealth_drag_vs_gross_nominal

    st.markdown(
        f"""
        <div class="metric-card advisor-card">
            <div class="card-title">
                <span>Advisor + Fund Stack</span>
                <span class="badge badge-red">{advisor_fee_pct:.2f}% Total</span>
            </div>
            <div class="card-subtitle">1.0% Advisor fee + 0.5% fund expenses</div>
            
            <div class="hero-stat">
                <div class="hero-val" style="color: #DC2626;">{fmt_curr(end_val3)}</div>
                <div class="hero-label">Ending Balance ({'Real' if is_real else 'Nominal'})</div>
            </div>

            <div class="stat-row">
                <span class="stat-label">Drag vs Low-Cost ETF:</span>
                <span class="stat-val stat-val-highlight-red">-{fmt_curr(drag_vs_etf3)} (-{res_adv.drag_pct_vs_baseline:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Direct Fees Paid:</span>
                <span class="stat-val">{fmt_curr(fees_paid3)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Lost Compounding Cost:</span>
                <span class="stat-val">{fmt_curr(lost_comp3)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Wealth Drag vs Gross:</span>
                <span class="stat-val">{fmt_curr(total_drag_gross3)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Break-Even Alpha Required:</span>
                <span class="stat-val stat-val-highlight-red">+{res_adv.break_even_alpha_pct:.2f}% (+{res_adv.break_even_alpha_bps:.0f} bps)</span>
            </div>
            <div class="stat-row" style="border-bottom: none;">
                <span class="stat-label">Wealth Retained vs Potential:</span>
                <span class="stat-val">
                    {(res_adv.ending_balance_nominal / res_adv.gross_ending_balance_nominal * 100):.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# INTERACTIVE DATA VISUALIZATIONS
# ==========================================
st.markdown("---")
st.subheader("Interactive Visual Analytics")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Portfolio Growth Trajectory",
        "📊 Fee Drag Decomposition",
        "📉 Wealth Drag Gap Over Time",
        "🎯 Break-Even Alpha Analysis",
    ]
)

with tab1:
    chart1 = build_trajectory_chart(
        simulation_results,
        is_real=is_real,
        show_gross=show_gross_benchmark,
        show_principal=show_principal_baseline,
    )
    render_altair(chart1)
    st.info(
        f"**Trajectory Insights:** Over {years} years, the Low-Cost Index ETF compounds to **{fmt_curr(end_val)}**, "
        f"outperforming the High-Fee Mutual Fund by **{fmt_curr(drag_vs_etf2)}** and the Advisor Stack by **{fmt_curr(drag_vs_etf3)}**."
    )

with tab2:
    chart2 = build_wealth_breakdown_bar_chart(simulation_results, is_real=is_real)
    render_altair(chart2)
    st.caption(
        "💡 **Key Insight:** Notice the red portion (**Lost Compounding**). Direct fees are only the tip of the iceberg; "
        "the compounding returns you forfeit on those fee dollars comprise the majority of your total wealth drag."
    )

with tab3:
    chart3 = build_drag_gap_timeline_chart(
        simulation_results, baseline_key="index_etf", is_real=is_real
    )
    render_altair(chart3)
    st.caption(
        "💡 In the first 5-10 years, fee drag seems modest. But by years 20-30, the exponential divergence of compound growth causes the wealth gap to surge exponentially."
    )

with tab4:
    chart4 = build_alpha_hurdle_chart(simulation_results)
    render_altair(chart4)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(
            f"""
            #### 🎯 Break-Even Alpha Required
            To merely match the ending balance of a **{etf_fee_pct:.2f}%** index fund:
            - **High-Fee Fund ({mutual_fund_fee_pct:.2f}%):** Must beat the index by **+{res_fund.break_even_alpha_pct:.2f}% (+{res_fund.break_even_alpha_bps:.0f} bps)** every year.
            - **Advisor Stack ({advisor_fee_pct:.2f}%):** Must beat the index by **+{res_adv.break_even_alpha_pct:.2f}% (+{res_adv.break_even_alpha_bps:.0f} bps)** every year.
            """
        )
    with col_a2:
        st.markdown(
            """
            #### 📉 What Empirical Research Shows (SPIVA Data)
            - **85% to 95%** of active equity fund managers fail to beat their benchmark index over 15 to 20-year horizons.
            - Outperforming by 147 basis points persistently every year after fees is nearly statistically impossible over multi-decade periods.
            - Past manager outperformance shows almost zero persistence in future market cycles.
            """
        )


# ==========================================
# YEAR-BY-YEAR BREAKDOWN TABLE & CSV DOWNLOAD
# ==========================================
st.markdown("---")
st.subheader("📋 Year-by-Year Schedule & Export")

comp_df = build_comparison_dataframe(simulation_results, baseline_key="index_etf")

col_tbl_ctl1, col_tbl_ctl2 = st.columns([3, 1])
with col_tbl_ctl1:
    view_filter = st.radio(
        "Table Granularity:",
        options=["5-Year Milestones", "Complete Year-by-Year Schedule"],
        horizontal=True,
    )

if view_filter == "5-Year Milestones":
    filtered_df = comp_df[comp_df["Year"] % 5 == 0].reset_index(drop=True)
else:
    filtered_df = comp_df

# Display formatted table
render_dataframe(
    filtered_df.style.format(
        {
            col: "${:,.2f}" if col != "Year" else "{:d}"
            for col in filtered_df.columns
        }
    ),
    height=400,
)

# CSV Export Button
csv_data = comp_df.to_csv(index=False).encode("utf-8")
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    label="📥 Download Complete Schedule as CSV",
    data=csv_data,
    file_name=f"fee_wealth_drag_schedule_{timestamp_str}.csv",
    mime="text/csv",
    help="Download full year-by-year portfolio simulation and drag comparison in CSV format.",
)


# ==========================================
# EDUCATIONAL GUIDE & FINANCIAL MATH
# ==========================================
st.markdown("---")
with st.expander("📚 Educational Guide: The Mathematical Reality of Fee Drag", expanded=False):
    st.markdown(
        """
        ### 1. The "1% Fallacy"
        Investors often believe that a 1.00% or 1.50% annual advisory fee means they keep 98.5% or 99.0% of their money.
        **This is fundamentally mistaken.**
        
        The fee is charged against your **entire account balance (Assets Under Management)** every single year, regardless of whether the market goes up or down.
        
        Over a 30-year horizon with an 8% gross return:
        - A **1.50% fee** consumes **~26% to 30%** of your entire lifetime accumulated wealth.
        - That is over **one-fourth** of your entire retirement nest egg forfeited to financial intermediaries.

        ---

        ### 2. Direct Fees vs. Lost Compounding (Opportunity Cost)
        Every dollar removed from your portfolio as a fee today can never compound tomorrow.
        
        $$\\text{Total Wealth Drag} = \\text{Gross Potential Wealth} - \\text{Net Ending Wealth}$$
        $$\\text{Total Wealth Drag} = \\text{Direct Fees Paid} + \\text{Lost Compounding}$$

        In multi-decade simulations, **Lost Compounding** is typically **1.3x to 1.8x larger** than all direct fees paid combined!

        ---

        ### 3. Break-Even Alpha Mathematical Proof
        Let $r_g$ be the gross market return, $f_1$ be the low-cost ETF fee (e.g. 0.03%), and $f_k$ be the high-fee fund or advisor fee (e.g. 1.50%).
        
        To produce the identical net ending balance, the active portfolio's net return must equal the index portfolio's net return:
        $$(r_g + \\alpha) - f_k = r_g - f_1$$
        $$\\alpha = f_k - f_1$$
        
        For an advisor charging 1.50% versus a 0.03% ETF:
        $$\\alpha = 1.50\\% - 0.03\\% = +1.47\\% \\text{ per year (147 bps)}$$
        
        Because active management fees are guaranteed deductions, an active manager must generate **+147 bps of pure excess skill every single year** just to leave you with the same balance as a simple index fund.
        """
    )

# Footer
st.markdown(
    """
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem; margin-top: 40px; padding: 20px 0;">
        Fee & Wealth Drag Retirement Analyzer &bull; Built with Streamlit, Pandas, NumPy & Altair &bull; Open Source Financial Planning
    </div>
    """,
    unsafe_allow_html=True,
)
