"""
Simple usage example for emitter geolocation system
"""

from emitter_geolocation import Sensor, SignalFeatures, EmitterGeolocation

# Define sensors with their positions and DOA measurements
sensors = [
    Sensor(
        sensor_id="Sensor_1",
        latitude=40.7128,    # degrees
        longitude=-74.0060,  # degrees
        doa=45.0            # degrees (0=North, 90=East)
    ),
    Sensor(
        sensor_id="Sensor_2",
        latitude=40.6892,
        longitude=-74.0445,
        doa=120.0
    ),
    Sensor(
        sensor_id="Sensor_3",
        latitude=40.7831,
        longitude=-73.9712,
        doa=300.0
    ),
]

# Define signal characteristics
signal_features = SignalFeatures(
    frequency=9.4e9,      # 9.4 GHz
    prf=2000,             # 2 kHz
    pulse_width=0.5e-6    # 0.5 microseconds
)

# Initialize geolocation system
geolocator = EmitterGeolocation()

# Estimate emitter location
result = geolocator.estimate_emitter_location(sensors, signal_features)

# Display results
print("Emitter Geolocation Results:")
print(f"Latitude:  {result.latitude:.6f}°")
print(f"Longitude: {result.longitude:.6f}°")
print(f"Confidence: {result.confidence_score:.3f}")
print(f"Residual Error: {result.residual_error:.3f}")
