from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_FILE = Path(os.path.join(ROOT, 'data', 'raw', 'weatherAUS.csv'))
processed_dir = Path(os.path.join(Root, 'data', 'processed'))
processed_dir.mkdir(exist_ok=True)
OUT_TRAIN = Path(os.path.join(processed_dir, 'train_processed.csv'))
OUT_TEST = Path(os.path.join(processed_dir, 'processed','test_processed.csv'))
ARTIFACTS_DIR = Path(os.path.join(ROOT, 'artifacts'))
ARTIFACTS_DIR.mkdir(exist_ok=True)

def load_and_clean(df):
    # 1. Drop kolom NaN >20%
    cols = df.columns
    for col in cols:
        if df[col].isna().sum() > 28000:
            df = df.drop(columns=col)

    # 2. Drop rows (WindDir all NaN)
    wd = ['WindGustDir','WindDir9am','WindDir3pm']
    df = df.drop(df.loc[df[wd].isna().all(axis=1)].index)

    # 3. Drop rows with >10 NaN
    df = df.drop(df.loc[df.isna().sum(axis=1) > 10].index)

    # 4. FillNa pada RainTomorrow dg No
    df['RainTomorrow'] = df['RainTomorrow'].fillna('No')
    df['RainToday'] = df['RainToday'].fillna('No')


    return df


def get_wind_map():
    return {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }


def feature_engineering(df):
    wind_map = get_wind_map()

    # direction → degree
    df['windGustDir_deg'] = df['WindGustDir'].map(wind_map)
    df['windDir9am_deg'] = df['WindDir9am'].map(wind_map)
    df['windDir3pm_deg'] = df['WindDir3pm'].map(wind_map)

    # sin/cos encoding
    df['windGust_x'] = np.sin(np.deg2rad(df['windGustDir_deg']))
    df['windGust_y'] = np.cos(np.deg2rad(df['windGustDir_deg']))

    df['wind9am_x'] = np.sin(np.deg2rad(df['windDir9am_deg']))
    df['wind9am_y'] = np.cos(np.deg2rad(df['windDir9am_deg']))

    df['wind3pm_x'] = np.sin(np.deg2rad(df['windDir3pm_deg']))
    df['wind3pm_y'] = np.cos(np.deg2rad(df['windDir3pm_deg']))

    # label encoding simple
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

    # final features for v1.0 (DHT11 only)

    # extract datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    #feature generation
    df['hum_now'] = df['Humidity3pm']
    df['temp_now'] = df['Temp3pm']
    return df


def reduce_features(df):

    drop_cols = [
        'Location','WindGustDir','WindDir9am','WindDir3pm',
        'windGustDir_deg','windDir9am_deg','windDir3pm_deg','Date',
        'MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am',
        'WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am',
        'Pressure3pm','Temp9am','Temp3pm','RainToday',
        'windGust_x','windGust_y','wind9am_x','wind9am_y','wind3pm_x','wind3pm_y'
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df


def main():
    # Load raw
    df = pd.read_csv(RAW_FILE)

    # Cleaning (safe, not leaking)
    df = load_and_clean(df)

    # Train-test split FIRST (anti-leak)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, shuffle=True, 
                                   random_state=42, stratify=df['RainTomorrow'])

    # Numerical imputers
    num_cols = [
        'MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm',
        'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm'
    ]

    # compute medians using TRAIN ONLY
    medians = train[num_cols].median()
    joblib.dump(medians, ARTIFACTS_DIR/'medians.pkl')

    # apply to train
    train[num_cols] = train[num_cols].fillna(medians)
    # apply to test
    test[num_cols] = test[num_cols].fillna(medians)

    # Fill RainToday/Tomorrow
    train[['RainToday','RainTomorrow']] = train[['RainToday','RainTomorrow']].fillna('No')
    test[['RainToday','RainTomorrow']] = test[['RainToday','RainTomorrow']].fillna('No')

    # groupby(Location) mode mapping → fit on TRAIN ONLY
    wind_cols = ['WindGustDir','WindDir9am','WindDir3pm']
    mode_map = (
        train.groupby('Location')[wind_cols]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N')
    )
    joblib.dump(mode_map, ARTIFACTS_DIR/'mode_map.pkl')

    # apply to train & test
    for col in wind_cols:
        train[col] = train.groupby('Location')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'N')
        )
        # test uses TRAIN mode map
        test[col] = test.apply(
            lambda row: row[col] if pd.notna(row[col]) 
            else mode_map.loc[row['Location'], col] if row['Location'] in mode_map.index 
            else 'N',
            axis=1
        )

    # Feature engineering
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Reduce features
    train = reduce_features(train)
    test = reduce_features(test)

    # Reset index
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Save processed datasets
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Proses Berhasil... Yeyeyyy")


if __name__ == "__main__":
    main()
