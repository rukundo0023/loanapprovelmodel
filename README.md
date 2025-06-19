# 🏦 Loan Approval Model & App

A professional, interactive machine learning web app for instant loan approval prediction. Users can enter their details and receive a real-time decision, risk score, and application summary—all powered by a trained Random Forest model.



## 🚀 Project Overview
This project demonstrates a complete ML workflow:
- Synthetic data generation for loan applications
- Model training (Random Forest Classifier)
- Interactive Streamlit app for user-friendly predictions


## ✨ Features
- Predicts loan approval for a $10,000 loan
- Calculates and displays credit risk score
- Collects user details: Age, Income, Job Type, Credit Score, Marital Status, Education Level
- Professional, modern UI with instant feedback
- Application summary table for user review
- Easily extensible for real-world data


## ⚙️ How It Works
1. Synthetic data is generated to simulate real loan applications.
2. The model is trained on features including demographic and financial info.
3. The Streamlit app collects user input, encodes features, and predicts approval and risk.


## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rukundo0023/loanapprovelmodel.git
cd loanapprovelmodel
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv ml-env
# On Windows:
ml-env\Scripts\activate
# On Mac/Linux:
source ml-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (if not already trained)
```bash
python loan_model_train.py
```

### 5. Run the Streamlit app
```bash
streamlit run loan_approval_app.py
```

---

## 💡 Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Enter applicant details in the sidebar.
- Instantly see approval status, risk score, and a summary of your application.


## 📊 Model Details
- Algorithm: Random Forest Classifier
- Features: Age, Income, Job Type, Credit Score, Marital Status, Education Level
- Encoders: LabelEncoders for all categorical features
- Data: Synthetic, for demonstration purposes only


## 🧰 Technologies Used
- Python 3
- scikit-learn
- pandas, numpy
- Streamlit
- joblib

