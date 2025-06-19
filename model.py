import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Create synthetic dataset with full names and IDs
data = pd.DataFrame({
    'applicant_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'full_name': ['Alice Smith', 'Bob Johnson', 'Carol Lee', 'David Kim', 'Eva Chen', 'Frank Brown', 'Grace Park', 'Henry Adams'],
    'age': [25, 45, 35, 50, 29, 42, 38, 60],
    'income': [50000, 120000, 75000, 130000, 40000, 95000, 80000, 110000],
    'job_type': ['Salaried', 'Self-Employed', 'Unemployed', 'Salaried', 'Salaried', 'Unemployed', 'Self-Employed', 'Salaried'],
    'credit_score': [0.65, 0.9, 0.55, 0.92, 0.45, 0.6, 0.75, 0.88],
    'marital_status': ['Single', 'Married', 'Divorced', 'Married', 'Single', 'Widowed', 'Divorced', 'Married'],
    'education_level': ['Bachelor', 'Master', 'High School', 'PhD', 'High School', 'Other', 'Master', 'Bachelor'],
    'approved': [0, 1, 0, 1, 0, 0, 1, 1]  # Target variable
})

# 2. Encode categorical features
le_job = LabelEncoder()
le_marital = LabelEncoder()
le_edu = LabelEncoder()

data['job_type_encoded'] = le_job.fit_transform(data['job_type'])
data['marital_status_encoded'] = le_marital.fit_transform(data['marital_status'])
data['education_level_encoded'] = le_edu.fit_transform(data['education_level'])

# 3. Define feature set and target (exclude full_name and applicant_id from features)
features = ['age', 'income', 'job_type_encoded', 'credit_score', 'marital_status_encoded', 'education_level_encoded']
X = data[features]
y = data['approved']

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save model and encoders
joblib.dump(model, 'loan_model.pkl')
joblib.dump(le_job, 'jobtype_encoder.pkl')
joblib.dump(le_marital, 'marital_encoder.pkl')
joblib.dump(le_edu, 'education_encoder.pkl')

# Optionally save the dataset with IDs and names for app use or records
data.to_csv('loan_data_with_ids.csv', index=False)

print("✅ Model and encoders saved successfully! Dataset with IDs also saved.")
