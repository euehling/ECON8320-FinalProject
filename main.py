import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# =========================
# Step 1: Request setup
# =========================
headers = {'Content-type': 'application/json'}
data = json.dumps({
    "seriesid": [
        'CES0000000001',  # Total Nonfarm Employees
        'LNS14000000',    # Unemployment Rate
        'CES0500000003',  # Avg Hourly Earnings, Private Sector
        'CUUR0000SAM',    # CPI: Medical Care
        'CUUR0000SAM1',   # CPI: Medical Care Commodities
        'CUUR0000SAM2',   # CPI: Medical Care Services
        'CUUR0000SA0',    # CPI: All Items
        'CUUR0000SEMC01', # CPI: Physician Services
        'CES6562000001',  # All Employees: Health Care and Social Assistance
        'CES6562000003',  # Avg Hourly Earnings: Health Care
        'CUUR0000SEMC02'  # Hospital Services
    ],
    "startyear": "2023",
    "endyear": "2025"
})

# =========================
# Step 2: API request
# =========================
response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
json_data = response.json()

if "Results" not in json_data or "series" not in json_data["Results"]:
    raise ValueError("BLS API response missing 'Results' or 'series' – check API output.")

# =========================
# Step 3: Descriptive titles
# =========================
titles = {
    'LNS14000000': 'Unemployment Rate (%)',
    'CES0000000001': 'Total Nonfarm Employees (Thousands)',
    'CES0500000003': 'Average Hourly Earnings (Private Sector)',
    'CUUR0000SAM': 'CPI: Medical Care',
    'CUUR0000SAM1': 'CPI: Medical Care Commodities',
    'CUUR0000SAM2': 'CPI: Medical Care Services',
    'CUUR0000SEMC01': 'CPI: Physician Services',
    'CUUR0000SA0': 'CPI: All Items',
    'CES6562000001': 'All Employees: Health Care and Social Assistance (Thousands)',
    'CES6562000003': 'Average Hourly Earnings (Health Care)',
    'CUUR0000SEMC02': 'Hospital Services'
}

# =========================
# Step 4: Create dfs per series
# =========================
dfs = {}
for series in json_data['Results']['series']:
    series_id = series['seriesID']
    rows = []
    for item in series['data']:
        if 'M01' <= item['period'] <= 'M12':  # monthly data only
            rows.append({
                "series_id": series_id,
                "year": int(item['year']),
                "period": item['period'],
                "value": float(item['value']),
                "footnotes": ", ".join(
                    [f['text'] for f in item['footnotes'] if f]
                )
            })
    dfs[series_id] = pd.DataFrame(rows)

# =========================
# Step 5: Format dates
# =========================
def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['period'] = df['period'].str.replace('M', '', regex=False)
    df['period'] = df['period'].astype(int).astype(str).str.zfill(2)

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['period'], format='%Y-%m')

    df = df.sort_values('date')
    df = df.set_index('date')
    return df

for key in dfs:
    dfs[key] = format_dates(dfs[key])

print("\n===== SAMPLE: Unemployment Rate Series =====")
print(dfs['LNS14000000'].head())

# =========================
# Step 6: Filter time window (Aug 2023 – Aug 2025)
# =========================
start_date = pd.to_datetime("2023-08-01")
end_date = pd.to_datetime("2025-08-31")

for key in dfs:
    df = dfs[key]
    dfs[key] = df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()


# =========================
# Step 7: Monthly % changes
# =========================
for key in dfs:
    dfs[key]['percent_change'] = dfs[key]['value'].pct_change()            # decimal
    dfs[key]['percent_change_pct'] = dfs[key]['percent_change'] * 100      # percentage

# =========================
# Step 8: Align time series (union of all dates)
# =========================
all_indexes = sorted(set().union(*[df.index for df in dfs.values()]))
combined_pct_change = pd.DataFrame(index=all_indexes)

for key, df in dfs.items():
    if 'percent_change' not in df.columns:
        print(f"Skipping {key}, no percent_change column found.")
        continue

    col_name = f"{titles.get(key, key)}_percent_change"
    combined_pct_change[col_name] = df['percent_change']

print("\n===== COMBINED PERCENT CHANGE DATAFRAME =====")
print(combined_pct_change.head())
print("\nShape:", combined_pct_change.shape)

# =========================
# Step 9: Descriptive analytics
# =========================
summary_stats = combined_pct_change.describe()
print("\n===== SUMMARY STATISTICS (MONTHLY % CHANGE) =====")
print(summary_stats)

correlation_matrix = combined_pct_change.corr()
print("\n===== CORRELATION MATRIX =====")
print(correlation_matrix)

# KPIs for dashboard (latest row)
latest_values = combined_pct_change.iloc[-1]
print("\n===== KPIs FOR DASHBOARD (LATEST MONTH) =====")
print(latest_values)

# =========================
# Step 10: Visualizations
# =========================

#------------- Medical Inflation vs Healthcare Wages -------------
medical_inflation = combined_pct_change['CPI: Medical Care Services_percent_change']
healthcare_wage_growth = combined_pct_change['Average Hourly Earnings (Health Care)_percent_change']

plt.figure(figsize=(12, 6))
plt.plot(medical_inflation.index, medical_inflation, label='Medical Care Services Inflation (MoM)', linewidth=2)
plt.plot(healthcare_wage_growth.index, healthcare_wage_growth, label='Healthcare Wage Growth (MoM)', linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("Medical Inflation vs Healthcare Wage Growth (MoM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Scatter + simple regression: Medical inflation vs HC wages -------------
x = healthcare_wage_growth.dropna()
y = medical_inflation.loc[x.index].dropna()

common_idx = x.index.intersection(y.index)
x = x.loc[common_idx]
y = y.loc[common_idx]

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)

m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, linewidth=2)

plt.xlabel("Healthcare Wage Growth (MoM, decimal)")
plt.ylabel("Medical Care Services Inflation (MoM, decimal)")
plt.title("Relationship Between Healthcare Wages and Medical Inflation")
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Healthcare vs Private-Sector Wage Growth -------------
hc_wage_col = "Average Hourly Earnings (Health Care)_percent_change"
priv_wage_col = "Average Hourly Earnings (Private Sector)_percent_change"

hc_wage = combined_pct_change[hc_wage_col]
priv_wage = combined_pct_change[priv_wage_col]

plt.figure(figsize=(12, 6))
plt.plot(hc_wage.index, hc_wage, label="Healthcare Wage Growth (MoM)", linewidth=2)
plt.plot(priv_wage.index, priv_wage, label="Private Sector Wage Growth (MoM)", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("Healthcare vs Private-Sector Wage Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Healthcare vs Total Employment Growth -------------
print("\n===== HEALTHCARE EMPLOYMENT VS TOTAL EMPLOYMENT =====")
hc_emp_col = "All Employees: Health Care and Social Assistance (Thousands)_percent_change"
total_emp_col = "Total Nonfarm Employees (Thousands)_percent_change"

hc_emp = combined_pct_change[hc_emp_col]
total_emp = combined_pct_change[total_emp_col]

plt.figure(figsize=(12, 6))
plt.plot(hc_emp.index, hc_emp, label="Healthcare Employment Growth (MoM)", linewidth=2)
plt.plot(total_emp.index, total_emp, label="Total Nonfarm Employment Growth (MoM)", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("Healthcare vs Total Employment Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Medical CPI subcomponents -------------
med_series_ids = [
    "CUUR0000SAM",    # Medical Care
    "CUUR0000SAM1",   # Medical Care Commodities
    "CUUR0000SAM2",   # Medical Care Services
    "CUUR0000SEMC01", # Physician Services
    # "CUUR0000SEMC02", # Hospital Services (if added visually)
]

plt.figure(figsize=(12, 6))
for sid in med_series_ids:
    df = dfs[sid]
    plt.plot(df.index, df["percent_change"], label=titles[sid], linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("Medical CPI Subcomponents: Monthly Changes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- 3-Month Rolling Averages -------------
medical_inflation_3m = medical_inflation.rolling(3).mean()
healthcare_wage_3m = healthcare_wage_growth.rolling(3).mean()

plt.figure(figsize=(12, 6))
plt.plot(medical_inflation_3m.index, medical_inflation_3m, label="Medical Inflation (3-month avg)", linewidth=2)
plt.plot(healthcare_wage_3m.index, healthcare_wage_3m, label="Healthcare Wage Growth (3-month avg)", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("3-Month Rolling Averages: Medical Inflation vs Healthcare Wage Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Correlation Heatmap -------------
corr = combined_pct_change.corr()

plt.figure(figsize=(10, 8))
im = plt.imshow(corr, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)

plt.title("Correlation Matrix – Monthly Percent Changes")
plt.tight_layout()
plt.show()

# =========================
# July 2024 snapshot
# =========================
print("\n===== JULY 2024 MEDICAL CPI SUBCOMPONENTS =====")

july_date = pd.to_datetime("2024-07-01")

series_ids = [
    'CUUR0000SAM',      # Medical Care
    'CUUR0000SAM1',     # Medical Care Commodities
    'CUUR0000SAM2',     # Medical Care Services
    'CUUR0000SEMC01'    # Physicians' Services
]

for sid in series_ids:
    df = dfs[sid]
    print(f"\n--- {titles[sid]} ({sid}) ---")
    if july_date in df.index:
        print(df.loc[july_date])
    else:
        print("No data for July 2024 in this series.")

print("\n===== JULY 2024 Monthly % Change (Medical CPI Components) =====")
for sid in series_ids:
    df = dfs[sid]
    if july_date in df.index:
        val = df.loc[july_date]['percent_change']
        print(f"{titles[sid]}: {val}")
    else:
        print(f"{titles[sid]}: No July 2024 data")

# =========================
# Statistical Analysis
# =========================

#------------- Medical CPI vs General CPI -------------
med_cpi_col = "CPI: Medical Care_percent_change"
all_cpi_col = "CPI: All Items_percent_change"

medical_cpi = combined_pct_change[med_cpi_col]
all_items_cpi = combined_pct_change[all_cpi_col]

# Time series comparison
plt.figure(figsize=(12, 6))
plt.plot(medical_cpi.index, medical_cpi, label="Medical CPI (MoM)", linewidth=2)
plt.plot(all_items_cpi.index, all_items_cpi, label="CPI: All Items (MoM)", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Monthly Percent Change (decimal)")
plt.title("Medical CPI vs General CPI (All Items) – Month-over-Month Changes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Difference over time (Medical – All Items)
cpi_spread = medical_cpi - all_items_cpi

plt.figure(figsize=(12, 6))
plt.plot(cpi_spread.index, cpi_spread, linewidth=2)
plt.axhline(0, linestyle="--")

plt.xlabel("Date")
plt.ylabel("Difference in MoM Change (decimal)")
plt.title("Medical CPI – All Items CPI (Monthly Spread)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter / regression: Medical vs General CPI
df_cpi_reg = pd.concat([medical_cpi, all_items_cpi], axis=1, join="inner").dropna()
X_cpi = sm.add_constant(df_cpi_reg[all_cpi_col])
y_cpi = df_cpi_reg[med_cpi_col]

cpi_model = sm.OLS(y_cpi, X_cpi).fit()
print("\n===== REGRESSION: Medical CPI vs General CPI (MoM) =====")
print(cpi_model.summary())

x_vals = df_cpi_reg[all_cpi_col]
y_vals = df_cpi_reg[med_cpi_col]

plt.figure(figsize=(8, 6))
plt.scatter(x_vals, y_vals, alpha=0.7, label="Monthly Observations")

line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
line_X = sm.add_constant(line_x)
line_y = cpi_model.predict(line_X)

plt.plot(line_x, line_y, linewidth=2, label="Best-Fit Line")

plt.xlabel("General CPI (All Items) MoM Change (decimal)")
plt.ylabel("Medical CPI MoM Change (decimal)")
plt.title("Medical CPI vs General CPI – Monthly Relationship")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------- Multiple Regression: Medical Services Inflation Drivers -------------
med_services_col = "CPI: Medical Care Services_percent_change"
hc_wage_col = "Average Hourly Earnings (Health Care)_percent_change"
hc_emp_col = "All Employees: Health Care and Social Assistance (Thousands)_percent_change"
total_emp_col = "Total Nonfarm Employees (Thousands)_percent_change"
priv_wage_col = "Average Hourly Earnings (Private Sector)_percent_change"
all_cpi_col = "CPI: All Items_percent_change"
unemp_col = "Unemployment Rate (%)_percent_change"

reg_cols = [
    med_services_col,
    hc_wage_col,
    hc_emp_col,
    total_emp_col,
    priv_wage_col,
    all_cpi_col,
    unemp_col
]

reg_df = combined_pct_change[reg_cols].dropna()

print("\n===== REGRESSION DATAFRAME SHAPE =====")
print(reg_df.shape)

y = reg_df[med_services_col]
X = reg_df[[hc_wage_col, hc_emp_col, total_emp_col, priv_wage_col, all_cpi_col, unemp_col]]
X = sm.add_constant(X)

med_inflation_model = sm.OLS(y, X).fit()

print("\n===== MULTIPLE REGRESSION: MEDICAL SERVICES INFLATION (MoM) =====")
print(med_inflation_model.summary())

reg_results_for_dashboard = {
    "R_squared": med_inflation_model.rsquared,
    "Adj_R_squared": med_inflation_model.rsquared_adj,
    "coef": med_inflation_model.params,
}

print("\n===== KEY MODEL METRICS (for dashboard) =====")
print(reg_results_for_dashboard)

#------------- ARIMA Forecast: Medical Care Services CPI (Level) -------------
med_services_series = dfs["CUUR0000SAM2"]["value"].copy()
med_services_series = med_services_series.sort_index()

# Explicitly set monthly frequency (Month Start or Month End, whichever matches your index)
med_services_series = med_services_series.asfreq("MS")


print("\n===== MEDICAL SERVICES CPI SERIES (LEVEL) =====")
print(med_services_series.tail())

arima_order = (1, 1, 1)
arima_model = ARIMA(med_services_series, order=arima_order)
arima_result = arima_model.fit()

print("\n===== ARIMA MODEL SUMMARY: MEDICAL SERVICES CPI (LEVEL) =====")
print(arima_result.summary())

forecast_steps = 12
forecast_result = arima_result.get_forecast(steps=forecast_steps)

forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

last_date = med_services_series.index[-1]
forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq="MS")

forecast_mean.index = forecast_index
forecast_ci.index = forecast_index

plt.figure(figsize=(12, 6))
plt.plot(med_services_series.index, med_services_series, label="Historical Medical Services CPI", linewidth=2)
plt.plot(forecast_mean.index, forecast_mean, label="Forecasted CPI (ARIMA)", linestyle="--", linewidth=2)

plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.xlabel("Date")
plt.ylabel("CPI Index Level (Medical Services)")
plt.title("ARIMA Forecast – CPI: Medical Care Services")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

