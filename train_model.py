"""
train_model.py
Step 1 of the backend: create simulated traffic data and train a
Random Forest model that predicts congestion (low / medium / high).
 
Run:  python train_model.py
"""
 
import math
import random
 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
 
random.seed(42)
 
# ------------------------------------------------------------------
# 1. ROADS: id -> normal (free-flow) speed in km/h
#    Same 6 roads as the website
# ------------------------------------------------------------------
ROADS = {1: 55, 2: 50, 3: 60, 4: 45, 5: 52, 6: 48}
 
 
# ------------------------------------------------------------------
# 2. TRAFFIC PATTERN: busy at 8-10 AM and 6-8 PM, quiet at night
# ------------------------------------------------------------------
def bell(hour, centre, width):
    """A smooth bump around 'centre' (used to make rush-hour peaks)."""
    return math.exp(-((hour - centre) ** 2) / (2 * width * width))
 
 
def traffic_factor(hour, day_of_week):
    """Higher number = more traffic. day_of_week: 0=Mon ... 6=Sun."""
    factor = 0.18                          # night baseline
    factor += 0.85 * bell(hour, 9, 1.3)    # morning rush
    factor += 0.80 * bell(hour, 19, 1.6)   # evening rush
    factor += 0.30 * bell(hour, 13, 3.0)   # mild midday traffic
    if day_of_week >= 5:                   # Sat/Sun: less rush-hour traffic
        factor *= 0.75
    return factor
 
 
def congestion_from_speed(speed):
    """Convert speed into a label (this is our 'answer' for the model)."""
    if speed >= 40:
        return "low"
    if speed >= 20:
        return "medium"
    return "high"
 
 
# ------------------------------------------------------------------
# 3. CREATE THE DATASET
#    For every road, day and hour we create many readings with a
#    little random noise so the data feels real.
# ------------------------------------------------------------------
rows = []
for road_id, base_speed in ROADS.items():
    for day in range(7):
        for hour in range(24):
            for _ in range(30):
                factor = traffic_factor(hour, day)
                speed = base_speed * (1.2 - 0.75 * factor) * random.uniform(0.9, 1.1)
                speed = max(5, min(75, speed))
                rows.append(
                    {
                        "road_id": road_id,
                        "hour": hour,
                        "day_of_week": day,
                        "speed": round(speed, 1),
                        "congestion": congestion_from_speed(speed),
                    }
                )
 
data = pd.DataFrame(rows)
data.to_csv("traffic_data.csv", index=False)
print(f"Created {len(data)} rows of traffic data -> traffic_data.csv")
print(data["congestion"].value_counts(), "\n")
 
# ------------------------------------------------------------------
# 4. TRAIN THE MODEL
#    Input (X)  : road, hour, day of week
#    Output (y) : congestion level
# ------------------------------------------------------------------
X = data[["road_id", "hour", "day_of_week"]]
y = data["congestion"]
 
# Keep 20% of the data aside to test the model on data it has not seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # this is the "learning" step
 
# ------------------------------------------------------------------
# 5. CHECK HOW GOOD IT IS
# ------------------------------------------------------------------
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%\n")
print(classification_report(y_test, predictions))
 
# ------------------------------------------------------------------
# 6. SAVE THE MODEL (Flask will load this file in the next step)
# ------------------------------------------------------------------
joblib.dump(model, "traffic_model.pkl")
print("Model saved -> traffic_model.pkl")
 
# Quick demo prediction: Road 1, 9 AM, Monday
demo = pd.DataFrame([{"road_id": 1, "hour": 9, "day_of_week": 0}])
print("Demo: Road 1 at 9 AM on Monday ->", model.predict(demo)[0])