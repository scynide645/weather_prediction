from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_FILE = ROOT_DIR /'data'/'raw'/'weatherAUS.csv'
PROCESSED_FILE = ROOT_DIR/'data'/'processed'/'AUS_weather_features.csv'

df = pd.read_csv(RAW_FILE)

# Dropping Column with NaN more than 20%
column = df.columns
for col in column:
    if (df[col].isna().sum() > 28000):
        df.drop(columns= col, inplace= True)

#Dropping row that columns Wind Direction when all are NaN(Dropped cause <1%)
wd = ['WindGustDir','WindDir9am', 'WindDir3pm']
df.drop(df.loc[df[wd].isna().all(axis= 1)].index, inplace= True)

#Dropping row that have more than 10 columns == NaN
df.drop(df.loc[df.isna().sum(axis= 1) > 10].index, inplace= True)

# Filling NaN value

num_cols = [
    'MinTemp','MaxTemp','Rainfall',
    'WindGustSpeed','WindSpeed9am','WindSpeed3pm',
    'Humidity9am','Humidity3pm',
    'Pressure9am','Pressure3pm',
    'Temp9am','Temp3pm'
]

for nc in num_cols:    
    df.fillna({nc: df[nc].median()}, inplace= True)

df[['RainToday', 'RainTomorrow']] = df[['RainToday', 'RainTomorrow']].fillna('No')

#Location mode imputation

wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

for col in wind_cols:
    df[col] = df.groupby('Location')[col].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'N')
    )

wind_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

#feature generation

df['windGustDir_deg'] = df['WindGustDir'].map(wind_map)
df['windDir9am_deg'] = df['WindDir9am'].map(wind_map)
df['windDir3pm_deg'] = df['WindDir3pm'].map(wind_map)

df['windGust_x'] = np.sin(np.deg2rad(df['windGustDir_deg']))
df['windGust_y'] = np.cos(np.deg2rad(df['windGustDir_deg']))

df['wind9am_x'] = np.sin(np.deg2rad(df['windDir9am_deg']))
df['wind9am_y'] = np.cos(np.deg2rad(df['windDir9am_deg']))

df['wind3pm_x'] = np.sin(np.deg2rad(df['windDir3pm_deg']))
df['wind3pm_y'] = np.cos(np.deg2rad(df['windDir3pm_deg']))

#Feature Binning
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear

#Dropping unused Columns
df.drop(columns=['Location','WindGustDir', 'WindDir9am', 'WindDir3pm', 'windGustDir_deg', 
                 'windDir9am_deg', 'windDir3pm_deg', 'Date'], inplace=True)

#Reseting index
df.reset_index(drop=True, inplace=True)

with open(PROCESSED_FILE, 'w') as f:
    df.to_csv(f, index=False)
