# 📂 Project Structure Overview

This document explains the folder and file structure of the project to help users and contributors quickly understand how everything is organized.

---

## 📁 Root Directory

```
weather_prediction/
│
├── esp32/                  # ESP32 firmware (PlatformIO)
├── models/                 # Trained ML models & evaluation results
├── notebooks/              # (Optional) Experiments & analysis notebooks
├── routes/                 # Flask blueprint routes
├── templates/              # HTML dashboard (Jinja2 templates)
│
├── run_preprocessing.py    # Data preprocessing pipeline
├── run_train.py            # Model training & evaluation
├── run_server.py           # Flask REST API & dashboard server
│
├── requirements.txt        # Python dependencies
├── README.md               # Main project documentation
└── STRUCTURE.md            # This file
```

---

## 📁 `esp32/` Folder

Contains the ESP32 firmware project managed using **PlatformIO**.

```
esp32/
├── include/
│   ├── config_template.h   # WiFi configuration (user editable)
│   ├── pins.h              # GPIO configuration for DHT11
│
├── src/
│   └── main.cpp            # ESP32 main application
│
├── platformio.ini          # PlatformIO configuration & dependencies
```

> ⚠️ Files inside `include/` must be configured before flashing the firmware.

---

## 📁 `models/` Folder

Generated after running `run_train.py`.

Contains:
- Trained Machine Learning model (.pkl)
- Model evaluation reports
- Feature importance results

These files are automatically loaded by the Flask server.

---

## 🌐 Flask Server Components

- **`routes/`**
  - Flask Blueprint routes (REST API & backend logic)

- **`templates/`**
  - HTML dashboard rendered using Jinja2

---

## 🐍 Python Entry Points

| File | Description |
|----|----|
| `run_preprocessing.py` | Prepares the dataset for training |
| `run_train.py` | Trains ML model and generates evaluation output |
| `run_server.py` | Runs Flask API and dashboard server |

---

This structure separates **IoT firmware**, **machine learning**, and **web services**, making the project easier to understand and extend.

