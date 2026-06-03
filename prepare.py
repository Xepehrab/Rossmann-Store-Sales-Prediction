import pandas as pd
import numpy as np

# train_df = pd.read_csv("data/train.csv")
# test_df = pd.read_csv("data/test.csv")
# store_df = pd.read_csv("data/store.csv")

def prepare_rossmann_data(train_df,store_df,is_train=True):
    """
    Cleans and engineers features for the Rossmann Store Sales dataset.
    """
    print("Merging data...")
    #Merge train and store datasets based on store id
    df=train_df.merge(store_df,on='Store',how='left')


    #Remove closed stores or stores with 0 sales
    if is_train:
        df = df[(df['Open'] != 0) & (df['Sales'] > 0)].copy()


    print("Handling missing values...")
    # Assume missing competition distance means no nearby competitor
    max_distance = df["CompetitionDistance"].max()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(max_distance)


    # Replace missing competition and promotion dates with 0
    fill_zero_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                      'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    for col in fill_zero_cols:
        df[col]=df[col].fillna(0)
    

    print("Extracting temporal features...")
    #Converts the string into a Pandas datetime object, and then extracts the individual numerical components into their own columns
    df['Date']=pd.to_datetime(df['Date'])
    df['Year']=df['Date'].dt.year
    df['Month']=df['Date'].dt.month
    df['Day']=df['Date'].dt.day
    df['DayOfWeek']=df['Date'].dt.dayofweek
    df['WeekOfYear']=df['Date'].dt.isocalendar().week.astype(int)
    #Finding weekends (6=Sunday , 5 = Saturday)
    df['IsWeekend']=df['DayOfWeek'].apply(lambda x:1 if x>=5 else 0)


    print("Calculating advanced duration features...")

    #Calculating each store 7 days sales 
    df = df.sort_values(by=['Store', 'Date'], ascending=True).reset_index(drop=True)
    df['Sales_lag_7']=df.groupby('Store')["Sales"].shift(7)


    # How many months has the competition been open relative to the current row's date?
    df['CompetitionOpenMonths']=12*(df['Year']-df['CompetitionOpenSinceYear'])+ (df['Month']-df['CompetitionOpenSinceMonth']) 

    #If competition hasn't opened yet (or no data exists), set age to 0
    df.loc[df['CompetitionOpenSinceYear'] == 0, 'CompetitionOpenMonths'] = 0
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].apply(lambda x: x if x > 0 else 0)

    #How many months has the continuous promotion (Promo2) been running?
    df['Promo2OpenMonths'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df ['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0

    df.loc[df['Promo2SinceYear'] == 0, 'Promo2OpenMonths'] = 0
    df['Promo2OpenMonths'] = df['Promo2OpenMonths'].apply(lambda x: x if x > 0 else 0)

    #Convert StateHoliday to numeric/string
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)
    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)
    
    
    
    print("Data preparation complete!")
    return df

