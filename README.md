# Mentrix - API Prediksi Depresi

API sederhana berbasis Flask untuk memprediksi apakah seseorang mengalami depresi atau tidak berdasarkan input data yang diberikan.

## Fitur

- Endpoint prediksi depresi menggunakan model machine learning
- Input data berupa JSON
- Output: prediksi `"Depresi"` atau `"Tidak Depresi"`

## Instalasi

Clone This Repository

Run App : Python app.py


## Endpoint

Input : 

{
  "Gender": "Male",
  "Age": 25.0,
  "Academic Pressure": 3.0,
  "Work Pressure": 2.0,
  "CGPA": 3.8,
  "Study Satisfaction": 8.0,
  "Job Satisfaction": 7.0,
  "Sleep Duration": "7-8 hours",
  "Dietary Habits": "Healthy",
  "Have you ever had suicidal thoughts ?": "Yes",
  "Work/Study Hours": 10.0,
  "Financial Stress": 2.0,
  "Family History of Mental Illness": "Yes"
}



Output :

{
  "prediction": 1,
  "prediction_label": "Depresi",
  "probability": [
    0.47054994106292725,
    0.5294500589370728
  ],
  "status": "success"
}


3 Steps to Run Mentrix

1. Run this Repo for model prediction (ML Backend)
2. Run https://github.com/cikinodapz/Mental_Health_App.git for server (API & Logic)
3. Run https://github.com/cikinodapz/Mentrix_App.git for frontend (User Interface)
