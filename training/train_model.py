# training/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# --- IMPORTING A NEW, MORE POWERFUL MODEL ---
from sklearn.ensemble import GradientBoostingClassifier
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

# --- (The rest of the file is the same until the model pipeline section) ---

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{DATA_PATH}'.")
    exit()

def get_time_in_minutes(series):
    dt_series = pd.to_datetime(series, errors='coerce')
    return dt_series.dt.hour * 60 + dt_series.dt.minute

X = pd.DataFrame()
if 'Bedtime' in df.columns: X['bedtime_mins'] = get_time_in_minutes(df['Bedtime'])
if 'Wakeup time' in df.columns: X['wake_time_mins'] = get_time_in_minutes(df['Wakeup time'])

numeric_features_from_csv = [
    'Age', 'Sleep duration', 'Awakenings', 'Caffeine consumption',
    'Alcohol consumption', 'Exercise frequency'
]
categorical_features_from_csv = ['Gender', 'Smoking status']
for col in numeric_features_from_csv:
    if col in df.columns: X[col] = df[col]
for col in categorical_features_from_csv:
    if col in df.columns: X[col] = df[col]

if 'Sleep efficiency' not in df.columns:
    raise RuntimeError("Critical column 'Sleep efficiency' not found.")
def map_quality_from_efficiency(efficiency_val):
    val = efficiency_val * 100 if efficiency_val <= 1.0 else efficiency_val
    if val >= 85: return 2
    elif val >= 75: return 1
    else: return 0

y = df['Sleep efficiency'].apply(map_quality_from_efficiency)
df_processed = pd.concat([X, y.rename("target")], axis=1)
df_clean = df_processed.dropna()
X_clean = df_clean.drop(columns=["target"])
y_clean = df_clean["target"]

num_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()

num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder='passthrough'
)

# --- NEW: USING THE MORE POWERFUL MODEL ---
pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE))
])

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clean
)

print("\nTraining the new Gradient Boosting model...")
pipeline.fit(X_train, y_train)
print("Training complete.")

y_pred = pipeline.predict(X_test)
print("\n--- New Model Evaluation ---")
print(classification_report(y_test, y_pred, target_names=["Poor (0)", "Average (1)", "Good (2)"]))
print("\nConfusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(pipeline, MODEL_OUT)
print(f"\n✅ Success! New model has been trained and saved to: {MODEL_OUT}")
