import pandas as pd
import numpy as np
from emitter_geolocation_ultra_precision import UltraPrecisionEmitterGeolocation, Sensor, SignalFeatures
import sys
import os

def read_csv_input_ultra_precision(csv_file='sensors.csv'):

    df = pd.read_csv(csv_file)
    results = []
    
    for idx, row in df.iterrows():
        scenario_id = row['scenario_id']
        
        doas = [row['sensor1_doa'], row['sensor2_doa'], row['sensor3_doa']]
        sensors = [
            Sensor(f'S1_{scenario_id}', row['sensor1_lat'], row['sensor1_lon'], doas[0]),
            Sensor(f'S2_{scenario_id}', row['sensor2_lat'], row['sensor2_lon'], doas[1]),
            Sensor(f'S3_{scenario_id}', row['sensor3_lat'], row['sensor3_lon'], doas[2])
        ]
        
        # Signal features
        signal_features = SignalFeatures(
            frequency=row['frequency'],
            prf=row['prf'],
            pulse_width=row['pulse_width']
        )
        
        # Ultra-precision geolocation
        try:
            geolocator = UltraPrecisionEmitterGeolocation()
            result = geolocator.estimate_emitter_location(sensors, signal_features)
            
            results.append({
                'scenario_id': scenario_id,
                'emitter_lat': result.latitude,
                'emitter_lon': result.longitude,
                'geometry_quality': result.geometry_quality,
                'residual_error': result.residual_error,
                'iterations': result.iterations,
                'method': result.method_used,
                'cep_radius_m': result.cep_radius_m
            })
            
        except Exception as e:
            results.append({
                'scenario_id': scenario_id,
                'emitter_lat': None,
                'emitter_lon': None,
                'geometry_quality': 0.0,
                'residual_error': float('inf'),
                'iterations': 0,
                'method': 'failed',
                'cep_radius_m': float('inf'),
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def calculate_error_for_scenario_13(result_lat: float, result_lon: float) -> float:
    """Calculate error for scenario 13 with known ground truth"""
    true_lat = 29.26369
    true_lon = 75.71890
    
    # Haversine formula
    R = 6371000  # Earth radius in meters
    
    lat1_rad, lon1_rad = np.radians(result_lat), np.radians(result_lon)
    lat2_rad, lon2_rad = np.radians(true_lat), np.radians(true_lon)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

if __name__ == '__main__':
    input_csv = sys.argv[1] if len(sys.argv) > 1 else 'sensors.csv'
    output_csv = 'geolocation_results_final.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found")
        sys.exit(1)
    
    print("Ultra-Precision Batch Geolocation Processing")
    print("=" * 60)
    
    results_df = read_csv_input_ultra_precision(input_csv)
    # Prepare minimal output (scenario id + coordinates)
    output_df = results_df[[
        'scenario_id',
        'emitter_lat',
        'emitter_lon'
    ]].copy()
    output_df[['emitter_lat', 'emitter_lon']] = output_df[['emitter_lat', 'emitter_lon']].round(5)
    output_df.to_csv(output_csv, index=False)
    
    # Display results
    print(f"{'#':<3} {'Scenario ID':<20} {'Latitude':<12} {'Longitude':<12}")
    print("-" * 60)
    
    for idx, row in output_df.iterrows():
        if pd.notna(row['emitter_lat']):
            print(f"{idx+1:<3} {row['scenario_id']:<20} {row['emitter_lat']:>11.5f}  {row['emitter_lon']:>11.5f}")
        else:
            print(f"{idx+1:<3} {row['scenario_id']:<20} {'FAILED':<12} {'FAILED':<12}")
    
    print("-" * 60)
    
    print(f"Processing complete. Results saved to {output_csv}")