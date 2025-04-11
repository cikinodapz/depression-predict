from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

app = Flask(__name__)

# Load model dan scaler yang sudah disimpan
model = joblib.load('xgboost_model_new.pkl')
scaler = joblib.load('scaler.pkl')

# Load label encoders untuk fitur kategorikal
label_encoders = {}
try:
    label_encoders = joblib.load('label_encoders.pkl')
    print("Label encoders loaded successfully")
except:
    print("Warning: Label encoders not found. Using default encoding.")
    # Jika file label_encoders.pkl tidak ada, kita perlu mendefinisikan encoding manual
    # sesuai dengan yang digunakan saat training

# Definisikan kolom-kolom yang dibutuhkan (sesuai dengan dataset training)
REQUIRED_COLUMNS = ["Gender", "Age", "Academic Pressure", "Work Pressure", "CGPA", 
                   "Study Satisfaction", "Job Satisfaction", "Sleep Duration", 
                   "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                   "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]

# Definisikan kolom-kolom kategorikal
CATEGORICAL_COLUMNS = ["Gender", "Sleep Duration", "Dietary Habits", 
                      "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi depresi berdasarkan input pengguna
    
    Request format (JSON):
    {
        "Gender": "Male",
        "Age": 33.0,
        "Academic Pressure": 5.0,
        "Work Pressure": 0.0,
        "CGPA": 8.97,
        "Study Satisfaction": 2.0,
        "Job Satisfaction": 0.0,
        "Sleep Duration": "5-6 hours",
        "Dietary Habits": "Healthy",
        "Have you ever had suicidal thoughts ?": "Yes",
        "Work/Study Hours": 3.0,
        "Financial Stress": 1.0,
        "Family History of Mental Illness": "No"
    }
    
    Response:
    {
        "prediction": 0 atau 1 (0: tidak depresi, 1: depresi),
        "probability": [prob_kelas_0, prob_kelas_1],
        "status": "success" atau "error",
        "message": pesan tambahan (khususnya untuk error)
    }
    """
    try:
        # Parse request body as JSON
        data = request.get_json()
        
        # Validasi input: periksa apakah semua kolom yang diperlukan ada
        for col in REQUIRED_COLUMNS:
            if col not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing required field: {col}"
                }), 400
        
        # Buat DataFrame dari input
        input_df = pd.DataFrame([data])
        
        # Konversi kolom kategorikal ke numerik menggunakan label encoder
        for col in CATEGORICAL_COLUMNS:
            if col in input_df.columns:
                if col in label_encoders:
                    # Gunakan label encoder yang sudah dilatih
                    try:
                        input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]
                    except ValueError:
                        # Jika kategori tidak ditemukan dalam training, berikan nilai default (0)
                        input_df[col] = 0
                        print(f"Warning: Unknown category in column {col}: {input_df[col].iloc[0]}")
                else:
                    # Jika tidak ada label encoder untuk kolom ini, gunakan konversi sederhana
                    # Ini tidak ideal dan seharusnya menggunakan encoding yang sama seperti saat training
                    if col == "Gender":
                        input_df[col] = 1 if input_df[col].iloc[0].lower() == "male" else 0
                    elif col == "Sleep Duration":
                        # Mapping sederhana berdasarkan jam
                        sleep_map = {"less than 4 hours": 0, "4-5 hours": 1, "5-6 hours": 2, 
                                    "6-7 hours": 3, "7-8 hours": 4, "more than 8 hours": 5}
                        input_df[col] = sleep_map.get(input_df[col].iloc[0], 2)  # default ke "5-6 hours"
                    elif col == "Dietary Habits":
                        diet_map = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
                        input_df[col] = diet_map.get(input_df[col].iloc[0], 1)  # default ke "Average"
                    elif col == "Have you ever had suicidal thoughts ?":
                        input_df[col] = 1 if input_df[col].iloc[0].lower() == "yes" else 0
                    elif col == "Family History of Mental Illness":
                        input_df[col] = 1 if input_df[col].iloc[0].lower() == "yes" else 0
        
        # Pastikan Financial Stress adalah numerik
        input_df["Financial Stress"] = pd.to_numeric(input_df["Financial Stress"], errors="coerce")
        
        # Ubah semua kolom menjadi tipe float
        input_df = input_df.astype(float)
        
        # Normalisasi input menggunakan scaler yang sudah dilatih
        input_scaled = scaler.transform(input_df)
        
        # Prediksi menggunakan model
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0].tolist()
        
        # Return hasil prediksi
        return jsonify({
            "prediction": int(prediction),
            "probability": probability,
            "status": "success",
            "prediction_label": "Depresi" if prediction == 1 else "Tidak Depresi"
        })
    
    except Exception as e:
        # Handle error
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk memeriksa status API"""
    return jsonify({"status": "healthy"})

@app.route('/', methods=['GET'])
def home():
    """Halaman beranda API"""
    return """
    <h1>API Prediksi Depresi Mahasiswa</h1>
    <p>Gunakan endpoint /predict dengan metode POST untuk melakukan prediksi.</p>
    """

# Fungsi untuk menyimpan label encoders (jalankan setelah training)
def save_label_encoders(encoders_dict):
    """
    Simpan label encoders yang digunakan selama training
    """
    joblib.dump(encoders_dict, 'label_encoders.pkl')
    print("Label encoders saved successfully")

if __name__ == '__main__':
    # Tambahkan kode ini jika label_encoders belum disimpan
    """
    # Contoh pembuatan dan penyimpanan label encoders
    from sklearn.preprocessing import LabelEncoder
    
    # Buat dictionary untuk menyimpan label encoders
    label_encoders = {}
    
    # Buat dan latih label encoder untuk setiap kolom kategorikal
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))  # Pastikan untuk melatih dengan data training
        label_encoders[col] = le
    
    # Simpan label encoders
    save_label_encoders(label_encoders)
    """
    
    # Jalankan aplikasi Flask
    app.run(debug=True, host='0.0.0.0', port=5000)


#BISMILLAH DONE
#tes doank