"""
wrangle() for the Hatla2ee 2026 used-car listings dataset.

Use in any notebook:
    from wrangle import wrangle
    df = wrangle("raw.csv")        # or wrangle(raw_dataframe)

Or paste the function into a notebook cell.
Only stateless cleaning happens here: nothing is learned from the data, so it is
safe to run before the train/test split. km imputation, encoding and scaling
belong in the sklearn Pipeline.
"""
import numpy as np
import pandas as pd


def wrangle(source, reference_year=2026, price_max=20_000_000, km_max=430_000,
            default_fuel="Gas", min_age_for_km_zero=2, verbose=True):
    # accept a DataFrame or a path to a CSV
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    def log(step):
        if verbose:
            print(f"{step:<40} rows = {len(df):>6}")

    def to_number(s):   # "122,000 KM" -> 122000.0, unparseable -> NaN
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s.astype(str).str.replace(r"[^\d]", "", regex=True),
                             errors="coerce").astype(float)

    log("raw")

    # description = brand + model + year, fully redundant
    df = df.drop(columns="description", errors="ignore")

    # rows with no price can't be used for training or evaluation
    df = df.dropna(subset=["price", "year"])
    log("drop missing price / year")

    df["fuel"] = df["fuel"].fillna(default_fuel)

    # "530,000 EGP" -> 530000.0
    df["price"] = to_number(df["price"])
    df["km"] = to_number(df["km"])
    df = df.dropna(subset=["price"])

    # year -> car_age, brand + model -> brand_model
    df["car_age"] = reference_year - df["year"]
    df["brand_model"] = df["brand"] + " " + df["model"]
    df = df.drop(columns=["year", "model"])

    # km == 0 on a car that isn't new = missing reading (before any km cap)
    fake_zero = (df["km"] == 0) & (df["car_age"] >= min_age_for_km_zero)
    df.loc[fake_zero, "km"] = np.nan
    if verbose:
        print(f"{'km set to NaN (fake zeros)':<40} rows = {int(fake_zero.sum()):>6}")

    # fixed caps for typos / absurd values (NaN km is kept)
    df = df[df["price"] <= price_max]
    log(f"drop price > {price_max:,}")
    df = df[(df["km"] <= km_max) | df["km"].isna()]
    log(f"drop km > {km_max:,}")

    df = df.drop_duplicates()
    log("drop duplicates")

    return df.reset_index(drop=True)
