# Emitter Geolocation System

Production-ready bearing-only localization using Direction of Arrival (DOA) measurements.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from emitter_geolocation import Sensor, SignalFeatures, EmitterGeolocation

# Define sensors
sensors = [
    Sensor("S1", 40.7128, -74.0060, 45.0),   # ID, lat, lon, DOA
    Sensor("S2", 40.6892, -74.0445, 120.0),
    Sensor("S3", 40.7831, -73.9712, 300.0),
]

# Define signal
signal = SignalFeatures(
    frequency=9.4e9,      # Hz
    prf=2000,             # Hz
    pulse_width=0.5e-6    # seconds
)

# Estimate location
geolocator = EmitterGeolocation()
result = geolocator.estimate_emitter_location(sensors, signal)

print(f"Latitude:  {result.latitude:.6f}°")
print(f"Longitude: {result.longitude:.6f}°")
print(f"Confidence: {result.confidence_score:.3f}")
```

## Input

- **Sensors**: List of sensor positions (lat/lon) with DOA measurements (degrees)
- **Signal Features**: Frequency (Hz), PRF (Hz), Pulse Width (seconds)

## Output

- **Latitude**: Estimated emitter latitude (degrees)
- **Longitude**: Estimated emitter longitude (degrees)
- **Confidence Score**: 0-1 (higher is better)

## Algorithm

Weighted Least Squares (WLS) triangulation with automatic coordinate transformation (UTM).
