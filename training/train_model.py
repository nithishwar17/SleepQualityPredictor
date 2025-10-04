# training/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- Config ---
DATA_PATH = "sleep_data.csv"
MODEL_OUT = "../backend/model.joblib"
RANDOM_STATE = 42

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{DATA_PATH}'. Please ensure your CSV file is in the 'training' folder.")
    exit()

print("Successfully loaded sleep_data.csv. Columns found:", df.columns.tolist())

# --- 2. Feature Engineering and Selection ---
# Create a new DataFrame for the features our model will use
X = pd.DataFrame()

# The 'Bedtime' and 'Wakeup time' are full dates, we only need the time part.
# This function extracts the time as minutes from midnight.
def get_time_in_minutes(series):
    # Convert column to datetime objects, coercing errors to NaT (Not a Time)
    dt_series = pd.to_datetime(series, errors='coerce')
    # Return time as minutes from midnight
    return dt_series.dt.hour * 60 + dt_series.dt.minute

# Add time-based features to our feature DataFrame X
if 'Bedtime' in df.columns:
    X['bedtime_mins'] = get_time_in_minutes(df['Bedtime'])
if 'Wakeup time' in df.columns:
    X['wake_time_mins'] = get_time_in_minutes(df['Wakeup time'])

# Select the other relevant columns from your CSV to use as features
# We will use both numeric and categorical features
numeric_features_from_csv = [
    'Age', 'Sleep duration', 'Awakenings', 'Caffeine consumption',
    'Alcohol consumption', 'Exercise frequency'
]
categorical_features_from_csv = ['Gender', 'Smoking status']

for col in numeric_features_from_csv:
    if col in df.columns:
        X[col] = df[col]

for col in categorical_features_from_csv:
    if col in df.columns:
        X[col] = df[col]

# --- 3. Create the Target Variable (y) ---
# Your dataset doesn't have a 'sleep_quality' column, so we create one
# based on the 'Sleep efficiency' score.
if 'Sleep efficiency' not in df.columns:
    raise RuntimeError("Critical column 'Sleep efficiency' not found. Cannot create target variable.")

def map_quality_from_efficiency(efficiency_val):
    # This function works whether efficiency is a ratio (0-1) or percentage (0-100)
    val = efficiency_val
    if val <= 1.0:
        val *= 100  # Convert ratio to percentage
    
    if val >= 85:
        return 2  # Good
    elif val >= 75:
        return 1  # Average
    else:
        return 0  # Poor

y = df['Sleep efficiency'].apply(map_quality_from_efficiency)
print("\nTarget variable 'sleep_quality' created from 'Sleep efficiency'.")
print("Class distribution:\n", y.value_counts())

# --- 4. Data Cleaning and Preprocessing Setup ---
# Combine features (X) and target (y), then drop any rows with missing values
df_processed = pd.concat([X, y.rename("target")], axis=1)
df_clean = df_processed.dropna()

print(f"\nDataset contains {df.shape[0]} rows. After removing rows with missing values, {df_clean.shape[0]} rows remain for training.")
if df_clean.shape[0] == 0:
    raise ValueError("No data left after cleaning. Please check your CSV for empty cells.")

X_clean = df_clean.drop(columns=["target"])
y_clean = df_clean["target"]

# Identify column types to apply the correct transformations (scaling vs. one-hot encoding)
num_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nFeatures selected for model training:")
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

# --- 5. Build and Train the Model Pipeline ---
# Create preprocessing pipelines
num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder='passthrough'
)

# Define the final model pipeline
pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clean
)

# Train the model
print("\nTraining the RandomForest model...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate and Save the Model ---
y_pred = pipeline.predict(X_test)

print("\n--- Model Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Poor (0)", "Average (1)", "Good (2)"]))

print("\nConfusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=["True Poor", "True Avg", "True Good"], columns=["Pred Poor", "Pred Avg", "Pred Good"]))

# Save the final trained pipeline
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(pipeline, MODEL_OUT)

print(f"\n✅ Success! Model has been trained and saved to: {MODEL_OUT}")