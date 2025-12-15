# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

DATA_PATH = Path("data") / "bls_monthly.csv"
st.set_page_config(
    page_title="Healthcare Labor & Medical Inflation",
    layout="wide"
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df.sort_values(["series_id", "date"], inplace=True)
    return df

raw = load_data()

# ---------- METADATA ----------
TITLES = {
    'LNS14000000': 'Unemployment Rate (%)',
    'CES0000000001': 'Total Nonfarm Employees (Thousands)',
    'CES0500000003': 'Average Hourly Earnings (Private Sector)',
    'CUUR0000SAM': 'CPI: Medical Care',
    'CUUR0000SAM1': 'CPI: Medical Care Commodities',
    'CUUR0000SAM2': 'CPI: Medical Care Services',
    'CUUR0000SA0': 'CPI: All Items',
    'CUUR0000SEMC01': 'CPI: Physician Services',
    'CES6562000001': 'All Employees: Health Care and Social Assistance (Thousands)',
    'CES6562000003': 'Average Hourly Earnings (Health Care)',
    'CUUR0000SEMC02': 'Hospital Services'
}


# ---------- PREPROCESS ----------
def preprocess(raw: pd.DataFrame):
    df = raw.copy()
    df.sort_values(["series_id", "date"], inplace=True)

    # Monthly percent change by series
    df["percent_change"] = df.groupby("series_id")["value"].pct_change()

    # Pivot to wide format for easier plotting
    wide = df.pivot(index="date", columns="series_id", values="percent_change")
    wide.rename(
        columns={sid: f"{TITLES.get(sid, sid)}_percent_change" for sid in wide.columns},
        inplace=True
    )
    return df, wide

df_long, combined_pct_change = preprocess(raw)
combined_pct_change = combined_pct_change.loc["2023-08-01":]


# ---------- KPI CALC ----------
def compute_kpis(wide: pd.DataFrame, long: pd.DataFrame):
    # Columns required for KPIs
    kpi_cols = [
        "CPI: Medical Care Services_percent_change",
        "Average Hourly Earnings (Health Care)_percent_change",
        "Average Hourly Earnings (Private Sector)_percent_change",
        "All Employees: Health Care and Social Assistance (Thousands)_percent_change",
        "CPI: Medical Care_percent_change",
        "CPI: All Items_percent_change",
    ]

    # Keep only rows where all KPI columns are non-null
    valid = wide[kpi_cols].dropna(how="any")
    latest_date = valid.index.max()
    latest_row = valid.loc[latest_date]
    k = {"date": latest_date}
    k["med_services_mom"] = latest_row["CPI: Medical Care Services_percent_change"]
    k["hc_wage_mom"] = latest_row["Average Hourly Earnings (Health Care)_percent_change"]
    k["priv_wage_mom"] = latest_row["Average Hourly Earnings (Private Sector)_percent_change"]
    k["hc_emp_mom"] = latest_row["All Employees: Health Care and Social Assistance (Thousands)_percent_change"]
    med_cpi = latest_row["CPI: Medical Care_percent_change"]
    all_cpi = latest_row["CPI: All Items_percent_change"]
    k["spread"] = med_cpi - all_cpi

    k["nonfarm_level"] = (
        long[long["series_id"] == "CES0000000001"]
        .dropna(subset=["value"])
        .sort_values("date")
        .iloc[-1]["value"]
    )

    k["unemp_rate"] = (
        long[long["series_id"] == "LNS14000000"]
        .dropna(subset=["value"])
        .sort_values("date")
        .iloc[-1]["value"]
    )

    return k

kpis = compute_kpis(combined_pct_change, df_long)

# ---------- UI: TITLE & KPIs ----------
st.title("The Cost of Healthcare: Labor Markets and Medical Inflation")

st.caption(
    "Dashboard built with monthly BLS data stored in this repository. "
    "Data is updated via a scheduled GitHub Action that appends new releases."
)


def fmt_pct(x):
    return f"{x * 100:.2f}%" if pd.notnull(x) else "NA"


latest_date_str = kpis["date"].strftime("%B %Y")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.metric(f"Medical Care Services CPI (MoM) – {latest_date_str}", fmt_pct(kpis["med_services_mom"]))
with c2:
    st.metric("Healthcare Wage Growth (MoM)", fmt_pct(kpis["hc_wage_mom"]))
with c3:
    st.metric("Private-Sector Wage Growth (MoM)", fmt_pct(kpis["priv_wage_mom"]))
with c4:
    st.metric("Healthcare Employment Growth (MoM)", fmt_pct(kpis["hc_emp_mom"]))
with c5:
    st.metric("Medical – General CPI Spread (MoM)", fmt_pct(kpis["spread"]))
st.markdown("---")
with c6:
    st.metric(
        "Total Nonfarm Employment",
        f"{kpis['nonfarm_level']:,.0f}K"
    )

with c7:
    st.metric(
        "Unemployment Rate",
        f"{kpis['unemp_rate']:.1f}%"
    )



def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------- PREP SERIES ----------
med_inf = combined_pct_change["CPI: Medical Care Services_percent_change"]
hc_wage = combined_pct_change["Average Hourly Earnings (Health Care)_percent_change"]
priv_wage = combined_pct_change["Average Hourly Earnings (Private Sector)_percent_change"]
med_cpi = combined_pct_change["CPI: Medical Care_percent_change"]
all_cpi = combined_pct_change["CPI: All Items_percent_change"]

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Med vs HC Wages",
    "HC Wages Scatter",
    "Med vs General CPI",
    "Wage Comparison",
    "CPI Subcomponents",
    "Rolling Averages",
    "Forecast"
])

# TAB 1: Medical vs HC Wages

with tab1:
    st.subheader("Medical Care Services Inflation vs Healthcare Wage Growth (MoM)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(med_inf.index, med_inf, label="Medical Care Services Inflation (MoM)", linewidth=2)
    ax.plot(hc_wage.index, hc_wage, label="Healthcare Wage Growth (MoM)", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Percent Change (decimal)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 2: Scatter

with tab2:
    st.subheader("Scatter: Healthcare Wages vs Medical Inflation")
    x = hc_wage.dropna()
    y = med_inf.loc[x.index].dropna()
    common_idx = x.index.intersection(y.index)
    x, y = x.loc[common_idx], y.loc[common_idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.7)
    m, b = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = m * line_x + b
    ax.plot(line_x, line_y, linewidth=2)
    ax.set_xlabel("Healthcare Wage Growth (MoM, decimal)")
    ax.set_ylabel("Medical Services Inflation (MoM, decimal)")
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 3: Med vs General CPI

with tab3:
    st.subheader("Medical CPI vs General CPI (All Items)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(med_cpi.index, med_cpi, label="Medical CPI (MoM)", linewidth=2)
    ax.plot(all_cpi.index, all_cpi, label="CPI: All Items (MoM)", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Percent Change (decimal)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)
    spread = med_cpi - all_cpi
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(spread.index, spread, linewidth=2)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Difference in MoM Change (decimal)")
    ax.set_title("Medical CPI – All Items CPI (Monthly Spread)")
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 4: Wages comparison

with tab4:
    st.subheader("Healthcare vs Private-Sector Wage Growth")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hc_wage.index, hc_wage, label="Healthcare Wage Growth", linewidth=2)
    ax.plot(priv_wage.index, priv_wage, label="Private-Sector Wage Growth", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Percent Change (decimal)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 5: Medical CPI subcomponents

with tab5:
    st.subheader("Medical CPI Subcomponents")
    fig, ax = plt.subplots(figsize=(10, 4))
    for sid in ["CUUR0000SAM", "CUUR0000SAM1", "CUUR0000SAM2", "CUUR0000SEMC01"]:
        sub = df_long[df_long["series_id"] == sid].copy()
        sub.sort_values("date", inplace=True)
        ax.plot(sub["date"], sub["percent_change"], label=TITLES[sid], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Percent Change (decimal)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 6: Rolling averages
with tab6:
    st.subheader("3-Month Rolling Averages")
    med_3m = med_inf.rolling(3).mean()
    hc_3m = hc_wage.rolling(3).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(med_3m.index, med_3m, label="Medical Inflation (3m avg)", linewidth=2)
    ax.plot(hc_3m.index, hc_3m, label="Healthcare Wage Growth (3m avg)", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Percent Change (decimal)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)

# TAB 7: ARIMA forecast
with tab7:
    st.subheader("ARIMA Forecast – Medical Care Services CPI (Level)")
    med_series = df_long[df_long["series_id"] == "CUUR0000SAM2"].set_index("date")["value"].sort_index()
    med_series = med_series.asfreq("MS")
    model = ARIMA(med_series, order=(1, 1, 1))
    result = model.fit()
    steps = 12
    fc = result.get_forecast(steps=steps)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int()
    last_date = med_series.index[-1]
    idx = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    mean_fc.index = idx
    ci.index = idx
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(med_series.index, med_series, label="Historical CPI: Medical Services", linewidth=2)
    ax.plot(mean_fc.index, mean_fc, label="Forecast (ARIMA)", linestyle="--", linewidth=2)
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    show_fig(fig)
st.markdown("---")
st.caption("Data updated monthly via GitHub Actions using the BLS Public API.")