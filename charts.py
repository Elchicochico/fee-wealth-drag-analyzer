"""
Visualization components using Altair for the Fee & Wealth Drag Retirement Analyzer.
Generates responsive, interactive charts for portfolio trajectories, fee drag decomposition,
compounding loss gaps, and alpha hurdle comparisons.
"""

from typing import Dict
import altair as alt
import pandas as pd
from analyzer import ScenarioResult


# Palette: Green for low-cost, Amber for mutual fund, Red for advisor, Slate for principal/benchmark
COLOR_PALETTE = {
    "Low-Cost Index ETF": "#10B981",       # Emerald green
    "High-Fee Mutual Fund": "#F59E0B",     # Amber orange
    "Advisor AUM + Fund Stack": "#EF4444", # Crimson red
    "Gross Benchmark (0% Fee)": "#9CA3AF", # Cool gray
    "Invested Principal": "#6366F1",       # Indigo
}


def build_trajectory_chart(
    results: Dict[str, ScenarioResult],
    is_real: bool = False,
    show_gross: bool = True,
    show_principal: bool = True,
) -> alt.Chart:
    """
    Creates an interactive multi-line growth chart comparing scenario balances over time.
    """
    records = []
    val_col = "balance_real" if is_real else "balance_nominal"
    unit_label = "Real Value ($)" if is_real else "Nominal Value ($)"

    # Add scenario lines
    for key, res in results.items():
        y_df = res.yearly_df
        for _, row in y_df.iterrows():
            records.append(
                {
                    "Year": int(row["year"]),
                    "Portfolio Value": float(row[val_col]),
                    "Scenario": res.name,
                    "Type": "Scenario",
                }
            )

    # Optional: Add Gross Benchmark
    if show_gross and "index_etf" in results:
        ref_df = results["index_etf"].yearly_df
        gross_col = "gross_balance_real" if is_real else "gross_balance_nominal"
        for _, row in ref_df.iterrows():
            records.append(
                {
                    "Year": int(row["year"]),
                    "Portfolio Value": float(row[gross_col]),
                    "Scenario": "Gross Benchmark (0% Fee)",
                    "Type": "Benchmark",
                }
            )

    # Optional: Add Principal Contributed
    if show_principal and "index_etf" in results:
        ref_df = results["index_etf"].yearly_df
        princ_col = "cum_contributions_real" if is_real else "cum_contributions_nominal"
        for _, row in ref_df.iterrows():
            records.append(
                {
                    "Year": int(row["year"]),
                    "Portfolio Value": float(row[princ_col]),
                    "Scenario": "Invested Principal",
                    "Type": "Principal",
                }
            )

    df_plot = pd.DataFrame(records)

    # Build color scale mapping
    domain = list(COLOR_PALETTE.keys())
    range_colors = [COLOR_PALETTE[k] for k in domain]

    # Nearest selection for interactive hover tooltip
    nearest = alt.selection_point(
        nearest=True,
        on="mouseover",
        fields=["Year"],
        empty=False,
    )

    base = alt.Chart(df_plot).encode(
        x=alt.X(
            "Year:Q",
            title="Investment Horizon (Years)",
            axis=alt.Axis(grid=True, format="d"),
        ),
        y=alt.Y(
            "Portfolio Value:Q",
            title=unit_label,
            axis=alt.Axis(grid=True, format="$,.0f"),
        ),
        color=alt.Color(
            "Scenario:N",
            title="Portfolio Scenario",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(orient="bottom", columns=3),
        ),
    )

    lines = base.mark_line(strokeWidth=3).encode(
        strokeDash=alt.condition(
            alt.datum.Type == "Benchmark",
            alt.value([6, 4]),
            alt.value([0]),
        )
    )

    # Invisible point selectors for smooth mouseover
    selectors = (
        alt.Chart(df_plot)
        .mark_point()
        .encode(x="Year:Q", opacity=alt.value(0))
        .add_params(nearest)
    )

    # Points on hover
    points = base.mark_circle(size=60).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Scenario:N", title="Scenario"),
            alt.Tooltip("Year:Q", title="Year"),
            alt.Tooltip("Portfolio Value:Q", title="Value", format="$,.0f"),
        ],
    )

    # Vertical hover rule
    rules = (
        alt.Chart(df_plot)
        .mark_rule(color="#9CA3AF", strokeDash=[3, 3])
        .encode(x="Year:Q")
        .transform_filter(nearest)
    )

    chart = (
        alt.layer(lines, selectors, points, rules)
        .properties(
            title=alt.TitleParams(
                text=f"Portfolio Growth Trajectory ({'Inflation-Adjusted Real' if is_real else 'Nominal'})",
                subtitle="Compare cumulative balance accumulation over your investment horizon",
                fontSize=16,
                subtitleFontSize=12,
            ),
            height=420,
        )
        .interactive()
    )

    return chart


def build_wealth_breakdown_bar_chart(
    results: Dict[str, ScenarioResult],
    is_real: bool = False,
) -> alt.Chart:
    """
    Stacked bar chart showing breakdown of:
    - Principal Contributions
    - Net Growth Kept
    - Direct Fees Paid
    - Lost Compounding (Opportunity Cost)
    """
    records = []

    for sc_key, res in results.items():
        if is_real:
            principal = res.total_contributions_real
            net_growth = res.total_net_growth_real
            direct_fees = res.total_fees_real
            lost_comp = res.lost_compounding_real
        else:
            principal = res.total_contributions_nominal
            net_growth = res.total_net_growth_nominal
            direct_fees = res.total_fees_nominal
            lost_comp = res.lost_compounding_nominal

        records.extend(
            [
                {
                    "Scenario": res.name,
                    "Component": "1. Invested Principal",
                    "Amount ($)": principal,
                    "Category": "Retained Wealth",
                },
                {
                    "Scenario": res.name,
                    "Component": "2. Net Investment Growth",
                    "Amount ($)": net_growth,
                    "Category": "Retained Wealth",
                },
                {
                    "Scenario": res.name,
                    "Component": "3. Direct Fees Paid",
                    "Amount ($)": direct_fees,
                    "Category": "Fee Drag",
                },
                {
                    "Scenario": res.name,
                    "Component": "4. Lost Compounding (Opp. Cost)",
                    "Amount ($)": lost_comp,
                    "Category": "Fee Drag",
                },
            ]
        )

    df_bar = pd.DataFrame(records)

    # Component colors
    color_scale = alt.Scale(
        domain=[
            "1. Invested Principal",
            "2. Net Investment Growth",
            "3. Direct Fees Paid",
            "4. Lost Compounding (Opp. Cost)",
        ],
        range=["#6366F1", "#10B981", "#F59E0B", "#EF4444"],
    )

    chart = (
        alt.Chart(df_bar)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Scenario:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Amount ($):Q",
                title="Total Potential Wealth ($)",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "Component:N",
                scale=color_scale,
                legend=alt.Legend(orient="bottom", columns=2, title="Wealth Allocation"),
            ),
            order=alt.Order("Component:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Component:N"),
                alt.Tooltip("Amount ($):Q", format="$,.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Decomposition of Total Potential Wealth",
                subtitle="Retained wealth (Principal + Net Growth) vs. Lost to fees & missed compounding",
                fontSize=16,
                subtitleFontSize=12,
            ),
            height=380,
        )
    )

    return chart


def build_drag_gap_timeline_chart(
    results: Dict[str, ScenarioResult],
    baseline_key: str = "index_etf",
    is_real: bool = False,
) -> alt.Chart:
    """
    Shows the cumulative wealth gap (loss) over time compared to the Low-Cost Index ETF.
    Highlights how fee drag explodes exponentially in later years.
    """
    base_res = results.get(baseline_key)
    if not base_res:
        return alt.Chart(pd.DataFrame()).mark_line()

    val_col = "balance_real" if is_real else "balance_nominal"
    base_values = base_res.yearly_df[val_col].values
    years = base_res.yearly_df["year"].values

    records = []
    for key, res in results.items():
        if key == baseline_key:
            continue
        sc_values = res.yearly_df[val_col].values
        drag_values = base_values - sc_values

        for y, drag in zip(years, drag_values):
            records.append(
                {
                    "Year": int(y),
                    "Wealth Drag ($)": max(0.0, float(drag)),
                    "Scenario": res.name,
                }
            )

    df_gap = pd.DataFrame(records)

    color_scale = alt.Scale(
        domain=["High-Fee Mutual Fund", "Advisor AUM + Fund Stack"],
        range=["#F59E0B", "#EF4444"],
    )

    chart = (
        alt.Chart(df_gap)
        .mark_area(opacity=0.4, line={"strokeWidth": 2.5})
        .encode(
            x=alt.X("Year:Q", title="Year", axis=alt.Axis(grid=True, format="d")),
            y=alt.Y(
                "Wealth Drag ($):Q",
                title="Lost Wealth vs Index ETF ($)",
                axis=alt.Axis(grid=True, format="$,.0f"),
            ),
            color=alt.Color(
                "Scenario:N",
                scale=color_scale,
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Year:Q"),
                alt.Tooltip("Wealth Drag ($):Q", format="$,.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="The Wealth Drag Gap Over Time",
                subtitle="Notice how fee drag starts tiny and accelerates into hundreds of thousands of dollars",
                fontSize=16,
                subtitleFontSize=12,
            ),
            height=360,
        )
        .interactive()
    )

    return chart


def build_alpha_hurdle_chart(results: Dict[str, ScenarioResult]) -> alt.Chart:
    """
    Bar chart showing the Break-Even Alpha hurdle required to match the Low-Cost Index ETF.
    """
    records = []
    for key, res in results.items():
        records.append(
            {
                "Scenario": res.name,
                "Alpha Hurdle (bps)": res.break_even_alpha_bps,
                "Alpha Hurdle (%)": res.break_even_alpha_pct,
                "Fee Rate (%)": res.fee_rate * 100.0,
            }
        )

    df_alpha = pd.DataFrame(records)

    color_scale = alt.Scale(
        domain=["Low-Cost Index ETF", "High-Fee Mutual Fund", "Advisor AUM + Fund Stack"],
        range=["#10B981", "#F59E0B", "#EF4444"],
    )

    bars = (
        alt.Chart(df_alpha)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=50)
        .encode(
            x=alt.X("Scenario:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Alpha Hurdle (bps):Q",
                title="Annual Break-Even Alpha Required (Basis Points)",
                axis=alt.Axis(grid=True),
            ),
            color=alt.Color("Scenario:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Fee Rate (%):Q", format=".2f"),
                alt.Tooltip("Alpha Hurdle (%):Q", format=".2f", title="Alpha Required (%)"),
                alt.Tooltip("Alpha Hurdle (bps):Q", format=".0f", title="Alpha Required (bps)"),
            ],
        )
    )

    text = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-6,
        fontSize=13,
        fontWeight="bold",
    ).encode(
        text=alt.Text("Alpha Hurdle (bps):Q", format="+.0f bps")
    )

    chart = (
        (bars + text)
        .properties(
            title=alt.TitleParams(
                text="Annual Break-Even Alpha Required to Match Index ETF",
                subtitle="The excess gross annual return a manager must generate every single year net of fees",
                fontSize=16,
                subtitleFontSize=12,
            ),
            height=340,
        )
    )

    return chart
