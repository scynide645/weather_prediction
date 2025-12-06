from flask import request, jsonify, Blueprint, current_app, render_template
import pandas as pd
import datetime as dt

main = Blueprint('main', __name__, url_prefix='/routes')

# tempat aman untuk simpan data terakhir dari ESP
sensor_data = {
    "temperature": None,
    "humidity": None
}
pred_result = None
prob = None

@main.route("/")
def home():
    return render_template('index.html')

@main.route("/predict", methods = ["POST"])
def predict_weather():
    global pred_result
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Tidak ada data'}), 400

    try:
        hum = float(data['humidity'])
        temp = float(data['temperature'])
    except KeyError as e:
        return jsonify({'error': f'Missing Field ({e})'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'humidity and temperature must be numbers'}), 400

    # simpan data sensor agar bisa diambil GET /test
    sensor_data["temperature"] = round(temp, 2)
    sensor_data["humidity"] = hum

    # build dataframe untuk model
    now = dt.datetime.now()
    df = pd.DataFrame([{
        'Year': now.year,
        'Month': now.month,
        'Day': now.day,
        'DayOfYear': now.timetuple().tm_yday,
        'hum_now': hum,
        'temp_now': temp
    }])

    try:
        pred = current_app.model.predict(df)
        global prob
        prob = current_app.model.predict_proba(df)[0][1]
    except Exception:
        current_app.log_exception('prediction failed')
        return jsonify({'error': 'prediction failed'})

    pred_result = int(pred[0])
    return jsonify({'Rain': pred_result})

@main.route("/test", methods = ["GET"])
def test():
    return jsonify({
    "temperature": sensor_data["temperature"],
    "humidity": sensor_data["humidity"],
    "rain": pred_result,
    "proba": round(prob*100, 2)
})

