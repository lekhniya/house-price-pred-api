# 🏠 California House Price Prediction API

A machine learning project that predicts California house prices using a Random Forest Regressor and exposes the trained model through a FastAPI REST API.

## 🚀 Features

- House price prediction through FastAPI
- Input validation using Pydantic
- Interactive Swagger UI
- CSV file upload for batch predictions
- Automatic CSV response with predictions
- Random Forest Regression model

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn

## 📂 Project Structure

```text
house-price-pred-api/
│
├── main.py
├── train.py
├── explore.py
├── test_houses.csv
├── requirements.txt
├── .gitignore
└── README.md

## 📸 API Results

### Single House Price Prediction
![Prediction Result](screenshots/predict.png)

### Batch CSV Prediction
![File Prediction Result](screenshots/predict-file.png)
