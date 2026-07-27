from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# ----------------------------
# FireCast AI Weather Download
# ----------------------------

# Vijayawada (change if needed)
LATITUDE = 16.5062
LONGITUDE = 80.6480

# Create output folder
OUTPUT_DIR = Path("datasets/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache + retry
cache = requests_cache.CachedSession(".cache", expire_after=3600)
session = retry(cache, retries=5, backoff_factor=0.2)

client = openmeteo_requests.Client(session=session)

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,

    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation_probability",
        "precipitation",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "soil_temperature_0cm",
        "soil_moisture_0_to_1cm"
    ],

    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "precipitation_probability_max",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "sunrise",
        "sunset",
        "uv_index_max"
    ],

    "timezone": "Asia/Kolkata"
}

response = client.weather_api(url, params=params)[0]

# ----------------------------
# Hourly
# ----------------------------

hourly = response.Hourly()

hourly_df = pd.DataFrame({
    "time": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),

    "temperature": hourly.Variables(0).ValuesAsNumpy(),
    "humidity": hourly.Variables(1).ValuesAsNumpy(),
    "dew_point": hourly.Variables(2).ValuesAsNumpy(),
    "apparent_temperature": hourly.Variables(3).ValuesAsNumpy(),
    "precip_probability": hourly.Variables(4).ValuesAsNumpy(),
    "precipitation": hourly.Variables(5).ValuesAsNumpy(),
    "pressure": hourly.Variables(6).ValuesAsNumpy(),
    "cloud_cover": hourly.Variables(7).ValuesAsNumpy(),
    "wind_speed": hourly.Variables(8).ValuesAsNumpy(),
    "wind_gust": hourly.Variables(9).ValuesAsNumpy(),
    "wind_direction": hourly.Variables(10).ValuesAsNumpy(),
    "soil_temperature": hourly.Variables(11).ValuesAsNumpy(),
    "soil_moisture": hourly.Variables(12).ValuesAsNumpy(),
})

hourly_file = OUTPUT_DIR / "hourly_weather.csv"
hourly_df.to_csv(hourly_file, index=False)

# ----------------------------
# Daily
# ----------------------------

daily = response.Daily()

daily_df = pd.DataFrame({
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s"),
        end=pd.to_datetime(daily.TimeEnd(), unit="s"),
        freq=pd.Timedelta(days=1),
        inclusive="left"
    ),

    "temperature_max": daily.Variables(0).ValuesAsNumpy(),
    "temperature_min": daily.Variables(1).ValuesAsNumpy(),
    "rainfall": daily.Variables(2).ValuesAsNumpy(),
    "rain_probability": daily.Variables(3).ValuesAsNumpy(),
    "wind_speed_max": daily.Variables(4).ValuesAsNumpy(),
    "wind_gust_max": daily.Variables(5).ValuesAsNumpy(),
    "sunrise": pd.to_datetime(daily.Variables(6).ValuesInt64AsNumpy(), unit="s"),
    "sunset": pd.to_datetime(daily.Variables(7).ValuesInt64AsNumpy(), unit="s"),
    "uv_index": daily.Variables(8).ValuesAsNumpy(),
})

daily_file = OUTPUT_DIR / "daily_weather.csv"
daily_df.to_csv(daily_file, index=False)

print("=" * 50)
print("FireCast AI Weather Download Complete")
print("=" * 50)
print(f"Hourly CSV : {hourly_file}")
print(f"Daily CSV  : {daily_file}")
print()
print(hourly_df.head())
print()
print(daily_df.head())