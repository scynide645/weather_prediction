weather_prediction_project/
│
├── data/                    # Semua data mentah & hasil preprocessing
│   ├── raw/
│   │   └── bmkg_raw.json    # Data original dari BMKG
│   ├── processed/
│   │   └── bmkg_clean.csv   # Data yang sudah dibersihkan
│   └── realtime/
│       └── esp32_data.csv   # Data log dari ESP32 (optional)
│
├── models/                  # Tempat simpan model akhir
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebooks/               # Eksperimen jupyter notebook
│   └── eda_and_training.ipynb
│
├── src/                     # Semua source code utama
│   ├── data_loader/
│   │   ├── download_bmkg.py       # Script ambil data BMKG
│   │   └── preprocessing.py        # Cleaning, encoding, scaling
│   │
│   ├── training/
│   │   └── train_model.py          # Training final model
│   │
│   ├── inference/
│   │   └── predict.py              # Load model → prediksi
│   │
│   ├── server/
│   │   ├── app.py                  # Flask server terima data dari ESP32
│   │   └── firebase_write.py       # (optional) tulis ke Firebase
│   │
│   └── utils/
│       └── helpers.py              # Fungsi umum
│
├── esp32/                  # Code ESP32
│   └── main.cpp
│
├── dashboard/              # Web UI (nanti)
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── style.css
│   └── dashboard.py
│
├── tests/                  # Unit tests (kalau jadi besar)
│
├── requirements.txt
└── README.md
