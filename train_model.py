# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load and prepare data
df = pd.read_csv('customer_churn.csv')

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)

# Convert categoricals
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'Churn':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Train model
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
dump(model, 'churn_model.joblib')
print("Model trained and saved as 'churn_model.joblib'")