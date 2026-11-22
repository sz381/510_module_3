"""Data preparation for the Seattle Airbnb price model.

Reads the three raw Inside Airbnb CSVs, cleans them, merges the calendar and
review aggregates back onto the listing table, engineers features and writes
the cleaned table plus the train / validation / test splits to data/processed/.

Usage:
    python src/preprocess.py

Functions provided:
- read_files:       read the three raw CSV files into DataFrames
- clean_listings:   parse currency, percent, boolean and date columns
- clean_calendar:   collapse the daily calendar into one row per listing
- clean_reviews:    collapse reviews into one row per listing
- merge_tables:     attach both aggregates and check the merge invariants
- build_features:   add tenure, distance, amenity and recency features
- select_features:  drop identifier and leakage columns, one-hot encode the rest
- split_data:       stratified train / validation / test split
- save_data:        write the interim table, the splits and the combined CSV
- main:             run the whole pipeline using config.yaml
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DEFAULT_DROP_COLS = [
    "license",
    "square_feet",
    "requires_license",
    "jurisdiction_names",
    "experiences_offered",
    "has_availability",
    "host_acceptance_rate",
    "market",
    "country",
    "country_code",
    "scrape_id",
    "last_scraped",
    "calendar_last_scraped",
    "calendar_updated",
]


def read_files(raw_dir):
    """Read the three raw CSV files.

    Args:
        raw_dir (str | Path): Directory holding listings.csv, calendar.csv and
            reviews.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The listings, calendar
        and reviews tables.
    """
    raw_dir = Path(raw_dir)
    listings = pd.read_csv(raw_dir / "listings.csv", low_memory=False)
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    reviews = pd.read_csv(raw_dir / "reviews.csv")
    return listings, calendar, reviews


def clean_listings(listings, drop_cols, max_nights):
    """Turn the raw listing table into typed, model-ready columns.

    Parses the currency, percent and boolean strings, converts the three date
    columns to timestamps, caps minimum_nights and groups the long tail of
    property_type into seven categories.

    Args:
        listings (pd.DataFrame): Raw listings table, one row per listing.
        drop_cols (list[str]): Columns that carry no signal and should go.
        max_nights (int): Upper bound applied to minimum_nights.

    Returns:
        pd.DataFrame: The cleaned listings table.
    """
    df = listings.copy()

    money_cols = [
        "price",
        "weekly_price",
        "monthly_price",
        "security_deposit",
        "cleaning_fee",
        "extra_people",
    ]
    for col in money_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[\$,]", "", regex=True),
            errors="coerce",
        )

    df["host_response_rate"] = (
        pd.to_numeric(
            df["host_response_rate"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        / 100
    )

    for col in [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
        "is_location_exact",
    ]:
        df[col] = df[col].map({"t": 1.0, "f": 0.0})

    for col in ["host_since", "first_review", "last_review"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["minimum_nights"] = df["minimum_nights"].clip(upper=max_nights)

    df["property_type"] = df["property_type"].fillna("Other")
    keep_types = [
        "House",
        "Apartment",
        "Townhouse",
        "Condominium",
        "Loft",
        "Bed & Breakfast",
    ]
    df["property_type"] = df["property_type"].where(
        df["property_type"].isin(keep_types), "Other"
    )

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def clean_calendar(calendar):
    """Collapse the daily calendar into one row per listing.

    Availability is turned into occupancy over 30, 90 and 365 days, the quoted
    daily price is summarised, and a weekend-to-weekday price ratio is added.
    Unavailable nights carry no price and are left as NaN.

    Args:
        calendar (pd.DataFrame): Raw calendar table, one row per listing per
            day.

    Returns:
        pd.DataFrame: One row per listing_id with the calendar features.
    """
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["price"] = pd.to_numeric(
        cal["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce",
    )
    cal["available"] = cal["available"].eq("t")

    start = cal["date"].min()

    agg = pd.DataFrame(index=cal["listing_id"].unique())
    agg.index.name = "listing_id"
    agg["calendar_price_median"] = cal.groupby("listing_id")["price"].median()
    agg["calendar_days_priced"] = cal.groupby("listing_id")["price"].count()

    for days in (30, 90, 365):
        window = cal[cal["date"] < start + pd.Timedelta(days=days)]
        agg[f"occupancy_{days}"] = 1 - window.groupby("listing_id")["available"].mean()

    window_30 = cal[cal["date"] < start + pd.Timedelta(days=30)]
    agg["available_days_30"] = window_30.groupby("listing_id")["available"].sum()

    cal["is_weekend"] = cal["date"].dt.dayofweek.isin([4, 5])
    weekly = cal.groupby(["listing_id", "is_weekend"])["price"].median().unstack()
    agg["weekend_price_ratio"] = weekly[True] / weekly[False]

    return agg.reset_index()


def clean_reviews(reviews, snapshot_date):
    """Collapse reviews into one row per listing.

    Duplicate (listing_id, reviewer_id) pairs are dropped before aggregation.
    Listings that were never reviewed keep NaN and are filled with 0 only for
    the rolling review count.

    Args:
        reviews (pd.DataFrame): Raw reviews table, one row per review.
        snapshot_date (pd.Timestamp): Reference date used for review recency.

    Returns:
        pd.DataFrame: One row per reviewed listing_id with the review features.
    """
    rev = reviews.copy()
    rev = rev.drop_duplicates(subset=["listing_id", "reviewer_id"], keep="first")
    rev["date"] = pd.to_datetime(rev["date"])
    rev["comment_length"] = rev["comments"].fillna("").str.len()

    agg = rev.groupby("listing_id").agg(
        n_reviews=("id", "count"),
        days_since_last_review=("date", lambda s: (snapshot_date - s.max()).days),
        comment_length_mean=("comment_length", "mean"),
    )

    last_year = rev[rev["date"] >= snapshot_date - pd.Timedelta(days=365)]
    agg["reviews_last_365d"] = last_year.groupby("listing_id").size()
    agg["reviews_last_365d"] = agg["reviews_last_365d"].fillna(0).astype(int)

    return agg.reset_index()


def merge_tables(listings, calendar_agg, reviews_agg, expected_rows):
    """Attach the calendar and review aggregates to the listing table.

    Fails loudly if the merge changes the number of rows, duplicates a listing
    id, or disagrees with the availability columns that Inside Airbnb computed.

    Args:
        listings (pd.DataFrame): Cleaned listings table.
        calendar_agg (pd.DataFrame): Output of clean_calendar.
        reviews_agg (pd.DataFrame): Output of clean_reviews.
        expected_rows (int): Row count expected after merging.

    Returns:
        pd.DataFrame: One row per listing with both aggregates attached.
    """
    merged = listings.merge(
        calendar_agg, left_on="id", right_on="listing_id", how="left"
    ).drop(columns=["listing_id"])

    merged = merged.merge(
        reviews_agg, left_on="id", right_on="listing_id", how="left"
    ).drop(columns=["listing_id"])

    assert len(merged) == expected_rows, (
        f"row count changed after merge: {len(merged)} != {expected_rows}"
    )
    assert merged["id"].is_unique, "duplicated listing id after merge"
    assert (
        merged["available_days_30"] == merged["availability_30"]
    ).all(), "calendar occupancy does not match availability_30"

    return merged


def build_features(df, snapshot_date, centre_lat, centre_lon, top_amenities):
    """Derive the engineered features used by the model.

    Args:
        df (pd.DataFrame): Merged table.
        snapshot_date (pd.Timestamp): Reference date for host tenure.
        centre_lat (float): Latitude of the city centre.
        centre_lon (float): Longitude of the city centre.
        top_amenities (int): How many amenity columns to create.

    Returns:
        pd.DataFrame: The input table plus the new features, without the raw
        amenities string.
    """
    out = df.copy()

    out["host_tenure_days"] = (snapshot_date - out["host_since"]).dt.days
    out["is_new_listing"] = out["last_review"].isna().astype(int)

    lat1 = np.radians(out["latitude"].to_numpy())
    lon1 = np.radians(out["longitude"].to_numpy())
    lat2 = np.radians(centre_lat)
    lon2 = np.radians(centre_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    out["distance_to_centre_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    def split_amenities(text):
        inner = str(text).strip("{}")
        parts = re.findall(r'"[^"]*"|[^,]+', inner)
        return [p.strip().strip('"') for p in parts if p.strip()]

    amenity_lists = out["amenities"].map(split_amenities)
    out["amenity_count"] = amenity_lists.map(len)

    counts = amenity_lists.explode().value_counts()
    for name in list(counts.index[:top_amenities]):
        column = "amenity_" + re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
        out[column] = amenity_lists.map(lambda xs, n=name: int(n in xs))

    return out.drop(columns=["amenities"])


def select_features(df, target):
    """Build the numeric feature matrix, the target and the stratification key.

    Identifiers, free text and the price columns derived from the target are
    dropped, then the remaining categorical columns are one-hot encoded.

    Args:
        df (pd.DataFrame): Featured table.
        target (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: X with numeric features only,
        y as the target, and strat as room_type before encoding.
    """
    drop = [
        "id",
        "host_id",
        "listing_url",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "thumbnail_url",
        "medium_url",
        "picture_url",
        "xl_picture_url",
        "name",
        "summary",
        "space",
        "description",
        "neighborhood_overview",
        "notes",
        "transit",
        "host_about",
        "host_location",
        "host_name",
        "host_verifications",
        "host_neighbourhood",
        "street",
        "zipcode",
        "city",
        "state",
        "smart_location",
        "neighbourhood",
        "neighbourhood_cleansed",
        "latitude",
        "longitude",
        "weekly_price",
        "monthly_price",
        "host_since",
        "first_review",
        "last_review",
        "calendar_price_median",
        "calendar_days_priced",
        "available_days_30",
    ]

    data = df.drop(columns=[c for c in drop if c in df.columns])

    y = data.pop(target)
    strat = data["room_type"].copy()

    X = pd.get_dummies(data, dummy_na=False)
    X = X.astype("float64")

    return X, y, strat


def split_data(X, y, strat, test_size, val_size, seed):
    """Split the data into train, validation and test sets.

    The split is drawn inside each room_type group so that all three room types
    appear in every set with the same proportions.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target series.
        strat (pd.Series): Column used to stratify the split.
        test_size (float): Fraction of rows for the test set.
        val_size (float): Fraction of rows for the validation set.
        seed (int): Random seed.

    Returns:
        dict[str, tuple[pd.DataFrame, pd.Series]]: The three splits keyed by
        train, val and test.
    """
    positions = np.arange(len(X))
    labels = strat.to_numpy()

    train_pos, rest_pos = train_test_split(
        positions,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=seed,
    )

    test_share = test_size / (test_size + val_size)
    val_pos, test_pos = train_test_split(
        rest_pos,
        test_size=test_share,
        stratify=labels[rest_pos],
        random_state=seed,
    )

    train = np.sort(train_pos)
    val = np.sort(val_pos)
    test = np.sort(test_pos)

    return {
        "train": (X.iloc[train], y.iloc[train]),
        "val": (X.iloc[val], y.iloc[val]),
        "test": (X.iloc[test], y.iloc[test]),
    }


def save_data(df, splits, interim_dir, processed_dir):
    """Write the intermediate table, the three splits and the combined CSV.

    The featured but not yet encoded table is written to the interim directory
    as CSV, for inspection only. The model-ready splits go to the processed
    directory, together with one combined CSV holding all three splits.

    Args:
        df (pd.DataFrame): Full cleaned and featured table.
        splits (dict[str, tuple[pd.DataFrame, pd.Series]]): Output of
            split_data.
        interim_dir (str | Path): Destination for the intermediate table.
        processed_dir (str | Path): Destination for the model-ready files.
    """
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(interim_dir / "airbnb_seattle.csv", index=False)

    parts = []
    for name, (X, y) in splits.items():
        out = X.copy()
        out["price"] = y.to_numpy()
        out["split"] = name
        out.to_parquet(processed_dir / f"{name}.parquet", index=False)
        parts.append(out)

    pd.concat(parts, ignore_index=True).to_csv(
        processed_dir / "airbnb_seattle.csv", index=False
    )


def main(config_path="config.yaml"):
    """Run the full preprocessing pipeline.

    Args:
        config_path (str): Path to config.yaml. Any missing key falls back to
            the default used here.
    """
    config_file = Path(config_path)
    cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    cfg = cfg or {}

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    interim_dir = data_cfg.get("interim_dir", "data/interim")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    target = data_cfg.get("target", "price")
    max_nights = data_cfg.get("max_nights", 30)
    snapshot_date = pd.Timestamp(data_cfg.get("snapshot_date", "2016-01-04"))
    drop_cols = data_cfg.get("drop_cols", DEFAULT_DROP_COLS)
    centre = data_cfg.get("centre", {"lat": 47.6062, "lon": -122.3321})
    top_amenities = data_cfg.get("top_amenities", 20)

    test_size = split_cfg.get("test_size", 0.15)
    val_size = split_cfg.get("val_size", 0.15)
    seed = split_cfg.get("seed", 42)

    listings, calendar, reviews = read_files(raw_dir)
    print("raw      :", listings.shape, calendar.shape, reviews.shape)

    listings = clean_listings(listings, drop_cols, max_nights)
    calendar_agg = clean_calendar(calendar)
    reviews_agg = clean_reviews(reviews, snapshot_date)
    print("aggregated:", listings.shape, calendar_agg.shape, reviews_agg.shape)

    merged = merge_tables(listings, calendar_agg, reviews_agg, len(listings))
    print("merged   :", merged.shape)

    featured = build_features(
        merged, snapshot_date, centre["lat"], centre["lon"], top_amenities
    )
    X, y, strat = select_features(featured, target)
    splits = split_data(X, y, strat, test_size, val_size, seed)

    save_data(featured, splits, interim_dir, processed_dir)

    print("features :", X.shape)
    print("target   : mean", round(y.mean(), 2), "median", round(y.median(), 2))
    for name, (X_part, y_part) in splits.items():
        print(f"{name:9s}: {X_part.shape[0]:,} rows")


if __name__ == "__main__":
    main()
