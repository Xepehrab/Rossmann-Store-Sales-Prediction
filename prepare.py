import pandas as pd
import numpy as np



def prepare_rossmann_data(train_df, store_df, is_train=True):
    """
    Cleans and engineers features for the Rossmann Store Sales dataset.
    """
    print("Merging data...")

    # Merge sales data with store metadata
    df = train_df.merge(store_df, on='Store', how='left')

    # Remove records that do not contribute to training
    if is_train:
        df = df[(df['Open'] != 0) & (df['Sales'] > 0)].copy()

    print("Handling missing values...")

    # Assume missing competition distance means no nearby competitor
    max_distance = df["CompetitionDistance"].max()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(max_distance)

    # Replace missing competition and promotion information
    fill_zero_cols = [
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'PromoInterval'
    ]

    for col in fill_zero_cols:
        df[col] = df[col].fillna(0)

    print("Extracting temporal features...")

    # Extract calendar-based features from the date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Flag weekends as a binary feature
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    print("Calculating advanced duration features...")

    # Sort data chronologically for time-series feature engineering
    df = df.sort_values(by=['Store', 'Date'], ascending=True).reset_index(drop=True)

    # Historical sales lag features
    df['Sales_lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_lag_1'] = df.groupby('Store')['Sales'].shift(1)
    df['Sales_lag_14'] = df.groupby('Store')['Sales'].shift(14)

    # Compute rolling averages to capture sales trends
    df['Sales_roll_mean_7'] = (
        df.groupby('Store')['Sales']
          .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df['Sales_roll_mean_30'] = (
        df.groupby('Store')['Sales']
          .transform(lambda x: x.shift(1).rolling(30).mean())
    )

    # Rolling sales volatility feature
    df['Sales_roll_std_7'] = (
        df.groupby('Store')['Sales']
          .transform(lambda x: x.shift(1).rolling(7).std())
    )

    # Fill NaN values created by lag and rolling features
    lag_cols = [
        'Sales_lag_1',
        'Sales_lag_7',
        'Sales_lag_14',
        'Sales_roll_mean_7',
        'Sales_roll_mean_30',
        'Sales_roll_std_7'
    ]

    df[lag_cols] = df[lag_cols].fillna(0)

    # Calculate competitor age in months
    df['CompetitionOpenMonths'] = (
        12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    )

    # Reset invalid competition ages to zero
    df.loc[df['CompetitionOpenSinceYear'] == 0, 'CompetitionOpenMonths'] = 0
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].apply(
        lambda x: x if x > 0 else 0
    )

    # Calculate Promo2 duration in months
    df['Promo2OpenMonths'] = (
        12 * (df['Year'] - df['Promo2SinceYear']) +
        (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    )

    df.loc[df['Promo2SinceYear'] == 0, 'Promo2OpenMonths'] = 0
    df['Promo2OpenMonths'] = df['Promo2OpenMonths'].apply(
        lambda x: x if x > 0 else 0
    )

    # Create a binary state holiday indicator
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)
    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)

    print("Data preparation complete!")

    return df