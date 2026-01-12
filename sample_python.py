import random
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Configuration
# -----------------------------
WIFI_APS = ["wifi_1", "wifi_2", "wifi_3"]
BLE_BEACONS = ["ble_1", "ble_2"]

LOCATIONS = {
    "Room_A": (0, 0),
    "Room_B": (5, 0),
    "Room_C": (0, 5),
    "Room_D": (5, 5),
}

NOISE_LEVEL = 2  # RSSI noise

# -----------------------------
# RSSI Simulation
# -----------------------------
def simulate_rssi(distance):
    """Log-distance path loss model (simplified)"""
    if distance == 0:
        distance = 0.1
    rssi = -30 - 20 * math.log10(distance)
    return rssi + random.gauss(0, NOISE_LEVEL)

# -----------------------------
# Generate Fingerprint
# -----------------------------
def generate_fingerprint(location_coord):
    fingerprint = []

    # Wi-Fi signals
    for _ in WIFI_APS:
        d = random.uniform(1, 10)
        fingerprint.append(simulate_rssi(d))

    # BLE signals
    for _ in BLE_BEACONS:
        d = random.uniform(0.5, 6)
        fingerprint.append(simulate_rssi(d))

    return fingerprint

# -----------------------------
# Build Fingerprint Database
# -----------------------------
X = []
y = []

SAMPLES_PER_LOCATION = 30

for location, coord in LOCATIONS.items():
    for _ in range(SAMPLES_PER_LOCATION):
        fp = generate_fingerprint(coord)
        X.append(fp)
        y.append(location)

X = np.array(X)

# -----------------------------
# Train Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# -----------------------------
# Simulate Live Scan
# -----------------------------
def live_scan():
    """Simulate real-time WiFi + BLE scan"""
    scan = []
    for _ in WIFI_APS:
        scan.append(simulate_rssi(random.uniform(1, 10)))
    for _ in BLE_BEACONS:
        scan.append(simulate_rssi(random.uniform(0.5, 6)))
    return np.array(scan).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
test_scan = live_scan()
predicted_location = model.predict(test_scan)

print("📡 Live RSSI Scan:", test_scan.flatten())
print("📍 Predicted Location:", predicted_location[0])
