# backend/model_utils.py
import pandas as pd
import numpy as np

# This helper function converts time strings (like "23:30") to minutes
def time_to_minutes(t):
    try:
        if pd.isna(t): return np.nan
        s = str(t).strip()
        if ":" in s:
            parts = s.split(":")
            h = int(parts[0])
            m = int(parts[1])
            return h * 60 + m
        return np.nan
    except:
        return np.nan

def preprocess_input_for_model(json_data):
    """
    Takes raw JSON from the API request and transforms it into a
    pandas DataFrame that matches the structure the model was trained on.
    """
    # Create a dictionary to hold the processed data
    d = {}
    
    # Extract and process each feature from the JSON input
    d['bedtime_mins'] = time_to_minutes(json_data.get("bedtime"))
    d['wake_time_mins'] = time_to_minutes(json_data.get("wake_time"))
    
    # Direct numeric inputs
    d['Age'] = float(json_data.get("age", 30))
    d['Sleep duration'] = float(json_data.get("sleep_duration", 7))
    d['Awakenings'] = int(json_data.get("awakenings", 0))
    d['Caffeine consumption'] = int(json_data.get("caffeine_consumption", 0))
    d['Alcohol consumption'] = int(json_data.get("alcohol_consumption", 0))
    d['Exercise frequency'] = int(json_data.get("exercise_frequency", 0))
    
    # Categorical inputs
    d['Gender'] = str(json_data.get("gender", "Female"))
    d['Smoking status'] = str(json_data.get("smoking_status", "No"))

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame([d])
    
    # IMPORTANT: Ensure the column order here is EXACTLY the same
    # as the order of columns used to train the model.
    expected_columns = [
        'bedtime_mins', 'wake_time_mins', 'Age', 'Sleep duration', 'Awakenings',
        'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency',
        'Gender', 'Smoking status'
    ]
    
    return df[expected_columns]