import pandas as pd
import numpy as np
import datetime

# Define date range
start_date = datetime.datetime(2023, 10, 1)
end_date = start_date + datetime.timedelta(days=15, hours=23)  # One week

date_range = pd.date_range(start=start_date, end=end_date, freq='h')

# Initialize lists to hold data
sea_temp = []
air_temp = []
wind_speed = []
wave_height = []
uv_index = []
activity_level = []

for current_time in date_range:
    hour = current_time.hour
    day_of_week = current_time.weekday()  # 0 = Monday, 6 = Sunday

    # Sea Surface Temperature
    sst = 27 + np.sin((hour / 24) * 2 * np.pi) * 0.5  # Slight daily variation
    sea_temp.append(round(sst, 1))

    # Air Temperature
    at = 22 + np.sin((hour / 24) * 2 * np.pi) * 12  # Cooler at night, warmer during the day
    air_temp.append(round(at, 1))

    # Wind Speed
    ws = 5 + np.random.normal(0, 2)  # Average wind speed with some randomness
    wind_speed.append(round(max(ws, 0), 1))  # Wind speed can't be negative

    # Wave Height
    wh = 0.3 + np.random.normal(0, 0.1)  # Average wave height with some randomness
    wave_height.append(round(max(wh, 0), 2))  # Wave height can't be negative

    # UV Index
    if 6 <= hour <= 18:
        uvi = max(0, (np.sin((hour - 6) / 12 * np.pi) * 10) + np.random.normal(0, 1))
    else:
        uvi = 0
    uv_index.append(round(uvi, 1))

    # Activity Level
    if day_of_week < 5:  # Weekday
        if 16 <= hour <= 19:
            al = 'High'
        elif 9 <= hour <= 15:
            al = 'Medium'
        else:
            al = 'Low'
    else:  # Weekend
        if 7 <= hour <= 19:
            al = 'High'
        else:
            al = 'Medium' if 6 <= hour <= 22 else 'Low'
    activity_level.append(al)

# Create DataFrame
data = pd.DataFrame({
    'Date & Time': date_range,
    'Sea Surface Temp (°C)': sea_temp,
    'Air Temp (°C)': air_temp,
    'Wind Speed (km/h)': wind_speed,
    'Wave Height (m)': wave_height,
    'UV Index': uv_index,
    'Activity Level': activity_level
})

# Save to CSV
data.to_csv('historical_beach_data.csv', index=False)
