# 🌦️ IoT Weather Prediction System  
**ESP32 · Flask REST API · Machine Learning**

This project is an **IoT-based weather prediction system** that combines real-time sensor data, a Flask REST API, and a machine learning model to predict rain conditions.

Temperature and humidity data are collected by an ESP32 using a DHT11 sensor, sent to a Flask server via HTTP (JSON), processed by a trained ML model, and finally displayed on a simple web dashboard.

> ⚠️ This project is under active development and will continue to be improved.

---

## 🔧 Devices & Hardware Used

- **Microcontroller:** ESP32-S3 N16R8  
- **Sensor:** DHT11 (Temperature & Humidity)

---

## 💻 Tested Environment

- **Operating System:** Ubuntu Linux 24.04 LTS  
- **Python:** Python 3.x  
- **ESP32 Toolchain:** PlatformIO  

---

## 🧩 System Flow Overview

```
ESP32 (DHT11)
   ↓ HTTP POST (JSON)
Flask REST API
   ↓
Machine Learning Model
   ↓
Prediction Result
   ↓
Web Dashboard (HTML)
```

---

## 📂 ESP32 Configuration (Important)

Before uploading the firmware to ESP32, some files **must be configured manually**.

### 📁 `esp32/include/`

#### 1️⃣ `config_template.h`
Contains WiFi credentials:
- SSID
- WiFi password  

⚠️ The ESP32 **must use the same network** as the PC running the Flask server.

#### 2️⃣ `pins.h`
- Defines the GPIO pin connected to the **DHT11 sensor**
- Adjust according to your wiring

#### 3️⃣ `platformio.ini`
- Required libraries (`lib_deps`) are already provided
- The **ESP32 board type must be adjusted** if you use a different variant

---

## 🐍 Python Pipeline (Execution Order)

This project uses **three main Python runner files**, and they **must be executed in order**:

```bash
python run_preprocessing.py
python run_train.py
python run_server.py
```

### Explanation

#### 1️⃣ `run_preprocessing.py`
- Loads and cleans raw weather data
- Prepares data for machine learning training

#### 2️⃣ `run_train.py`
- Trains the machine learning model
- Saves the trained model as a `.pkl` file
- Generates:
  - Model evaluation report
  - Feature importance analysis

All outputs are stored inside the `models/` directory.

#### 3️⃣ `run_server.py`
- Starts the Flask REST API server
- Loads the trained ML model
- Waits for data sent by the ESP32
- Serves the web dashboard

---

## 🤖 Model Output

After running `run_train.py`, the `models/` folder will contain:
- ✅ Trained model (`.pkl`)
- ✅ Evaluation report
- ✅ Feature importance results

---

## 🌐 Web Dashboard Access

Once `run_server.py` is running, the dashboard can be accessed from a browser:

```
http://flask-server.local:5000/routes
```

⚠️ Requirements:
- Flask server must be running
- ESP32 and PC must be on the **same local network**
- mDNS (`.local`) support must be available on the system

---

## 📊 Dataset Source

The dataset used to train the model was downloaded from Kaggle:

🔗 https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download

---

## 🚀 Future Development

This project will continue to be developed with potential improvements such as:
- Improved prediction accuracy
- Better dashboard UI
- Additional sensor support
- Edge AI optimization
- Deployment readiness

---

Happy learning ⚡  
This project is designed as a hands-on playground for **IoT + AI integration**.
