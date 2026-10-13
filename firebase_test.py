import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# Connect to Firebase using your private key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-traffic-moniter-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Write one test reading
ref = db.reference("test_reading")
ref.set({
    "road_id": 1,
    "road_name": "MG Road (Gachibowli)",
    "speed": 42,
    "vehicle_count": 130,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "note": "simulated data"
})
print("Written to Firebase!")

# Read it back to confirm
print("Read back:", ref.get())