# Mentrix - API Prediksi Depresi

API sederhana berbasis Flask untuk memprediksi apakah seseorang mengalami depresi atau tidak berdasarkan input data yang diberikan.

## Fitur

- Endpoint prediksi depresi menggunakan model machine learning
- Input data berupa JSON
- Output: prediksi `"Depresi"` atau `"Tidak Depresi"`

## Instalasi

Clone This Repository

Run App : Python app.py


Contoh Input : 

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
