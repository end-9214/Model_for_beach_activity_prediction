import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path="data/historical_beach_data.csv"):
    data = pd.read_csv(csv_path)

    # Label encoding
    le = LabelEncoder()
    data['Activity Level'] = le.fit_transform(data['Activity Level'])

    # Feature engineering
    data['Date & Time'] = pd.to_datetime(data['Date & Time'])
    data['Hour'] = data['Date & Time'].dt.hour
    data['dayOfweek'] = data['Date & Time'].dt.dayofweek

    features = ['Sea Surface Temp (°C)', 'Air Temp (°C)', 'Wind Speed (km/h)',
                'Wave Height (m)', 'UV Index', 'Hour', 'dayOfweek']
    X = data[features]
    y = data['Activity Level']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le
