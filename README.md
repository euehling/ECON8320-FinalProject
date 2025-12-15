
Healthcare Labor Markets & Medical Inflation Dashboard
Automated BLS Data Pipeline + Streamlit Dashboard

Overview
This project analyzes how U.S. healthcare labor conditions relate to medical inflation using monthly data from the Bureau of Labor Statistics (BLS).
It includes:

A Streamlit dashboard
A monthly automated data ingestion pipeline that updates BLS data in the repo
A growing historical dataset stored in data/bls_monthly.csv
Statistical analysis using regressions and forecasting
This project satisfies the course requirement to automate monthly economic data collection, store it in a GitHub repository, and visualize results using Streamlit.

Project Structure
.
├── app.py                    # Streamlit dashboard (frontend)
├── collect_bls_data.py       # Script that collects + appends monthly BLS data
├── requirements.txt
│
├── data/
│   └── bls_monthly.csv       # Automatically updated dataset
│
├── analysis/
│   └── main.py               # (Optional) Full analysis script
│
└── .github/
    └── workflows/
        └── bls_update.yml    # GitHub Action that updates data monthly

Data Sources (BLS Public API)

This project uses the following BLS series:

Total Nonfarm Employees
Unemployment Rate
Average Hourly Earnings (Private Sector)
CPI: Medical Care
CPI: Medical Care Commodities
CPI: Medical Care Services
CPI: All Items
CPI: Physician Services
Healthcare Employment
Average Hourly Earnings (Health Care)
Hospital Services
Dataset grows monthly and begins in 2023.

Automated Monthly Updates
GitHub Actions runs the workflow in:
.github/workflows/bls_update.yml

Every month it:
Runs collect_bls_data.py
Pulls new BLS data
Appends new rows to data/bls_monthly.csv
Commits the updated dataset back to GitHub
The Streamlit app never calls the API — it always uses the stored CSV file.

Running Locally

1. Install dependencies
pip install -r requirements.txt
2. Generate initial dataset (only needed once)
python collect_bls_data.py
3. Run the Streamlit app
streamlit run app.py
Dashboard will open at:
http://localhost:8501
Deployment (Streamlit Cloud)

Push this repository to GitHub
Go to https://share.streamlit.io
Select the repo and point it at app.py
Deploy — your online dashboard stays updated via GitHub Actions
Dashboard Features

KPIs:
Medical Services CPI (MoM)
Healthcare Wage Growth
Private-Sector Wage Growth
Healthcare Employment Growth
Medical CPI – All Items CPI spread

Visuals:
Medical Inflation vs Healthcare Wages
Healthcare Wages vs Medical Inflation (Scatter)
Medical CPI vs General CPI
Healthcare vs Private-Sector Wages
CPI Subcomponents
Rolling Averages (3-month)
ARIMA Forecast for Medical Services CPI
Requirements File

Ensure requirements.txt includes:
streamlit
pandas
numpy
matplotlib
statsmodels
requests

Published Dashboard:
https://econ8320-finalproject-jbf72tlr94s5ggk36cotcf.streamlit.app/

Summary of Findings:
Healthcare wages do not significantly predict medical inflation
Private-sector wages are a significant predictor
Medical CPI does not consistently exceed general inflation
Forecasting suggests medical inflation follows a persistent but non-accelerating trend
