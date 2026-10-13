
"""
app.py
Flask API: loads the trained Random Forest model and returns congestion
predictions as JSON. Also reads the latest SIMULATED readings from Firebase.
 
Run:  py app.py
Then open:  http://127.0.0.1:5000/
"""
 
import time
from datetime import datetime
 
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify, request, send_from_directory
 
app = Flask(__name__)
 
# Load the model that train_model.py saved
model = joblib.load("traffic_model.pkl")
 
# ---------------- Firebase connection ----------------
KEY_FILE = "serviceAccountKey.json"
DATABASE_URL = "https://smart-traffic-moniter-default-rtdb.asia-southeast1.firebasedatabase.app/"
 
firebase_ready = False
try:
    firebase_admin.initialize_app(
        credentials.Certificate(KEY_FILE), {"databaseURL": DATABASE_URL}
    )
    firebase_ready = True
    print("Firebase connected.")
except Exception as error:
    # The website and ML routes still work without Firebase
    print("WARNING: Firebase not connected:", error)
 
# Same 6 roads as the website (id -> name)
ROAD_NAMES = {
    1: "MG Road (Gachibowli)",
    2: "Tank Bund Road",
    3: "Hitech City Main Road",
    4: "Begumpet Main Road",
    5: "LB Nagar Ring Road",
    6: "Kukatpally (JNTU) Road",
}
 
 
@app.after_request
def allow_website_access(response):
    """Let the website call this API even if it is opened from another address."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
 
 
def predict_one(road_id, hour, day_of_week):
    """Ask the model for one road, hour and day. Returns a small dictionary."""
    row = pd.DataFrame(
        [{"road_id": road_id, "hour": hour, "day_of_week": day_of_week}]
    )
    level = model.predict(row)[0]
    confidence = model.predict_proba(row).max()  # how sure the model is (0 to 1)
    return {
        "road_id": road_id,
        "road_name": ROAD_NAMES[road_id],
        "hour": hour,
        "day_of_week": day_of_week,
        "congestion": level,
        "confidence": round(float(confidence), 2),
    }
 
 
def read_time_from_request():
    """Use ?hour= and ?day= from the URL, otherwise use the current time."""
    now = datetime.now()
    hour = request.args.get("hour", default=now.hour, type=int)
    day = request.args.get("day", default=now.weekday(), type=int)
    return hour, day
 
 
# ------------------------------------------------------------------
# ROUTES (each route is one "door" of the API)
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Traffic prediction API is running",
        "firebase": "connected" if firebase_ready else "not connected"
    }), 200
 
 
@app.route("/")
def home():
    """Show the website itself, so everything runs from one address."""
    return send_from_directory(".", "index.html")
 
 
@app.route("/predict")
def predict():
    """Predict one road.  Example: /predict?road_id=1&hour=9&day=0"""
    road_id = request.args.get("road_id", type=int)
    if road_id not in ROAD_NAMES:
        return jsonify({"error": "road_id must be between 1 and 6"}), 400
 
    hour, day = read_time_from_request()
    if not (0 <= hour <= 23 and 0 <= day <= 6):
        return jsonify({"error": "hour must be 0-23 and day must be 0-6"}), 400
 
    return jsonify(predict_one(road_id, hour, day))
 
 
@app.route("/predict_all")
def predict_all():
    """Predict all 6 roads at once.  Example: /predict_all?hour=18&day=2"""
    hour, day = read_time_from_request()
    if not (0 <= hour <= 23 and 0 <= day <= 6):
        return jsonify({"error": "hour must be 0-23 and day must be 0-6"}), 400
 
    results = [predict_one(road_id, hour, day) for road_id in ROAD_NAMES]
    return jsonify(results)
 
 
@app.route("/latest")
def latest():
    """Newest simulated reading for each road, read from Firebase.
    Example: /latest   (needs simulator.py running to stay fresh)"""
    if not firebase_ready:
        return jsonify({"error": "Firebase is not connected"}), 503
 
    try:
        data = db.reference("latest").get()
    except Exception as error:
        return jsonify({"error": f"Could not read Firebase: {error}"}), 503
 
    if not data:
        return jsonify({"error": "No data in Firebase yet. Run simulator.py first."}), 404
 
    now = int(time.time())
    readings = []
    for reading in data.values():
        reading["age_seconds"] = now - reading.get("ts", now)  # how old the reading is
        readings.append(reading)
    readings.sort(key=lambda r: r.get("road_id", 0))
    return jsonify(readings)
 
 
@app.route("/trend")
def trend():
    """Hourly average speed for one road, from stored Firebase history (last 24 h),
    plus the ML model's predicted congestion for each hour of today.
    Example: /trend?road_id=1"""
    road_id = request.args.get("road_id", type=int)
    if road_id not in ROAD_NAMES:
        return jsonify({"error": "road_id must be between 1 and 6"}), 400
    if not firebase_ready:
        return jsonify({"error": "Firebase is not connected"}), 503
 
    cutoff = int(time.time()) - 24 * 3600   # only the last 24 hours
    try:
        data = (db.reference(f"history/road_{road_id}")
                .order_by_key().start_at(f"t{cutoff}").limit_to_last(2000).get())
    except Exception as error:
        return jsonify({"error": f"Could not read Firebase: {error}"}), 503
 
    if not data:
        return jsonify({"error": "No history yet. Run backfill_history.py first."}), 404
 
    # Average the stored speed for each hour of the day (0-23)
    sums = [0.0] * 24
    counts = [0] * 24
    for record in data.values():
        hour = record.get("hour")
        speed = record.get("speed")
        if isinstance(hour, int) and 0 <= hour <= 23 and speed is not None:
            sums[hour] += float(speed)
            counts[hour] += 1
    avg_speed = [round(sums[h] / counts[h], 1) if counts[h] else None for h in range(24)]
 
    # ML model: predicted congestion for every hour of today (road + time only)
    rows = pd.DataFrame({
        "road_id": [road_id] * 24,
        "hour": list(range(24)),
        "day_of_week": [datetime.now().weekday()] * 24,
    })
    predicted = [str(level) for level in model.predict(rows)]
 
    return jsonify({
        "road_id": road_id,
        "road_name": ROAD_NAMES[road_id],
        "hours": list(range(24)),
        "avg_speed": avg_speed,          # stored SIMULATED readings
        "samples": counts,               # how many readings were averaged per hour
        "predicted_congestion": predicted,   # from the ML model
        "window_hours": 24,
    })
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)
 