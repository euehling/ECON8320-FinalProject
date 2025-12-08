import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "bls_monthly.csv"

BLS_SERIES = [
    "CES0000000001",  # Total Nonfarm Employees
    "LNS14000000",    # Unemployment Rate
    "CES0500000003",  # Avg Hourly Earnings, Private Sector
    "CUUR0000SAM",    # CPI: Medical Care
    "CUUR0000SAM1",   # CPI: Medical Care Commodities
    "CUUR0000SAM2",   # CPI: Medical Care Services
    "CUUR0000SA0",    # CPI: All Items
    "CUUR0000SEMC01", # CPI: Physician Services
    "CES6562000001",  # All Employees: Health Care and Social Assistance
    "CES6562000003",  # Avg Hourly Earnings: Health Care
    "CUUR0000SEMC02", # Hospital Services
]


def fetch_bls_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch BLS data for all series between start_year and end_year inclusive."""
    headers = {"Content-type": "application/json"}
    payload = json.dumps({
        "seriesid": BLS_SERIES,
        "startyear": str(start_year),
        "endyear": str(end_year),
    })

    resp = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    if "Results" not in data or "series" not in data["Results"]:
        raise ValueError("Unexpected BLS response: missing 'Results.series'")

    rows = []
    for series in data["Results"]["series"]:
        sid = series["seriesID"]
        for item in series["data"]:
            period = item["period"]

            # We only want monthly data M01–M12
            if not period.startswith("M"):
                continue

            year = int(item["year"])
            month = int(period.replace("M", ""))
            value = float(item["value"])
            dt = datetime(year=year, month=month, day=1)

            rows.append({
                "series_id": sid,
                "date": dt,
                "year": year,
                "month": month,
                "value": value,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["series_id", "date"], inplace=True)
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    current_year = now.year

    if DATA_PATH.exists():
        # Incremental update
        existing = pd.read_csv(DATA_PATH, parse_dates=["date"])
        existing.sort_values(["series_id", "date"], inplace=True)

        max_date = existing["date"].max()
        max_year = max_date.year

        start_year = max_year  # safe to re-pull this year; filter by date
        end_year = current_year

        print(f"Existing data found through {max_date.strftime('%Y-%m')}.")
        print(f"Fetching updates from {start_year} to {end_year}...")

        new_df = fetch_bls_data(start_year, end_year)
        new_df = new_df[new_df["date"] > max_date]

        if new_df.empty:
            print("No new months released yet. Nothing to append.")
            return

        updated = pd.concat([existing, new_df], ignore_index=True)
        updated.drop_duplicates(subset=["series_id", "date"], keep="last", inplace=True)
        updated.sort_values(["series_id", "date"], inplace=True)
        updated.to_csv(DATA_PATH, index=False)

        print(f"Appended {len(new_df)} new rows. Total rows: {len(updated)}")

    else:
        # First run: build a history (at least 1 year; here we do ~2 for comfort)
        start_year = current_year - 2
        end_year = current_year

        print(f"No existing data found. Fetching data from {start_year} to {end_year}...")
        df = fetch_bls_data(start_year, end_year)
        df.to_csv(DATA_PATH, index=False)
        print(f"Created new dataset with {len(df)} rows at {DATA_PATH}")


if __name__ == "__main__":
    main()