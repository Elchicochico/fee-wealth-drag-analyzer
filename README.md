# Fee & Wealth Drag Retirement Analyzer

An interactive retirement wealth modeling application built with Streamlit, pandas, and NumPy. It compares the long-term impact of investment expense ratios and advisory fees across three distinct portfolio scenarios.

## Overview

The application models the compounding difference between:
1. **Low-Cost Index ETF** (default: 0.03% annual expense ratio)
2. **High-Fee Mutual Fund** (default: 0.75% annual expense ratio)
3. **Advisor AUM + Fund Stack** (default: 1.50% total annual fee)

It breaks down ending portfolio values into nominal and inflation-adjusted terms, tracks total direct fees paid, computes lost compounding (opportunity cost), and calculates the required break-even alpha needed to match the index benchmark.

## Mathematical Formulation

- **Monthly Rate of Return:**
  $$r_{m} = \frac{r_{\text{gross}} - \text{fee}}{12}$$

- **Inflation Deflator:**
  $$\text{Real Balance}_m = \frac{\text{Nominal Balance}_m}{(1 + i)^{m/12}}$$

- **Total Wealth Drag:**
  $$\text{Wealth Drag} = \text{Gross Balance (0\% Fee)} - \text{Net Balance}$$

- **Lost Compounding (Opportunity Cost):**
  $$\text{Lost Compounding} = \text{Total Wealth Drag} - \text{Cumulative Direct Fees Paid}$$

- **Break-Even Alpha Required:**
  $$\alpha = \text{Fee}_{\text{scenario}} - \text{Fee}_{\text{ETF}}$$

## Local Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Repository Structure

- `app.py`: Streamlit dashboard and user interface layout
- `analyzer.py`: Financial simulation engine and cross-scenario calculations
- `charts.py`: Altair chart visual pipelines
- `requirements.txt`: Python package dependencies
- `.gitignore`: Standard project ignore definitions
