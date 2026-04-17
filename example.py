"""
Simple usage example for ultra-precision emitter geolocation system
"""

from emitter_geolocation_ultra_precision import UltraPrecisionEmitterGeolocation, Sensor, SignalFeatures

def main():
    print("Ultra-Precision Emitter Geolocation Example")
    print("=" * 50)
    
    # Example: Scenario 13 with corrected DOAs
    sensors = [
        Sensor("S1", 29.27247, 75.67879, 103.67),  # Corrected DOA
        Sensor("S2", 29.29001, 75.68673, 134.15),  # Corrected DOA  
        Sensor("S3", 29.28916, 75.70806, 159.13)   # Corrected DOA
    ]
    
    # Signal characteristics
    signal_features = SignalFeatures(
        frequency=8.5e9,      # 8.5 GHz
        prf=2400,             # 2.4 kHz
        pulse_width=0.7e-6    # 0.7 microseconds
    )
    
    # Initialize ultra-precision geolocation system
    geolocator = UltraPrecisionEmitterGeolocation()
    
    # Estimate emitter location
    result = geolocator.estimate_emitter_location(sensors, signal_features)
    
    # Display results
    print("Results:")
    print(f"  Latitude:  {result.latitude:.6f}°")
    print(f"  Longitude: {result.longitude:.6f}°")
    print(f"  Confidence: {result.confidence_score:.4f}")

if __name__ == "__main__":
    main()