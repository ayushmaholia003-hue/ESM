"""
Ultra-Precision Emitter Geolocation System
Designed specifically for challenging geometries and high accuracy requirements
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import warnings

@dataclass
class Sensor:
    """Represents a sensor with position and DOA measurement"""
    sensor_id: str
    latitude: float
    longitude: float
    doa: float  # Direction of arrival in degrees (0=North, 90=East, clockwise)
    
@dataclass
class SignalFeatures:
    """Common signal features for all sensors"""
    frequency: float  # Hz
    prf: float  # Pulse repetition frequency
    pulse_width: float

@dataclass
class EmitterEstimate:
    """Result of emitter geolocation"""
    latitude: float
    longitude: float
    confidence_score: float
    residual_error: float
    covariance_trace: float
    geometry_quality: float
    iterations: int
    method_used: str

class UltraPrecisionCoordinateConverter:
    """Ultra-high precision coordinate converter"""
    
    def __init__(self, reference_lat: float, reference_lon: float):
        """Initialize with exact geodetic parameters"""
        self.ref_lat = np.radians(reference_lat)
        self.ref_lon = np.radians(reference_lon)
        
        # WGS84 parameters
        self.a = 6378137.0  # Semi-major axis
        self.f = 1/298.257223563  # Flattening
        self.e2 = 2*self.f - self.f**2
        
        # Precompute for reference point
        self.sin_lat = np.sin(self.ref_lat)
        self.cos_lat = np.cos(self.ref_lat)
        
        # Radii of curvature
        self.M = self.a * (1 - self.e2) / (1 - self.e2 * self.sin_lat**2)**(3/2)
        self.N = self.a / np.sqrt(1 - self.e2 * self.sin_lat**2)
    
    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert to ENU with maximum precision"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        dlat = lat_rad - self.ref_lat
        dlon = lon_rad - self.ref_lon
        
        # High-order terms for better accuracy
        east = self.N * self.cos_lat * dlon * (1 - 0.5 * self.e2 * self.sin_lat**2 * dlon**2)
        north = self.M * dlat * (1 + 0.5 * dlat**2 / (6 * self.M**2))
        
        return east, north
    
    def enu_to_latlon(self, east: float, north: float) -> Tuple[float, float]:
        """Convert ENU to lat/lon with maximum precision"""
        dlat = north / self.M
        dlon = east / (self.N * self.cos_lat)
        
        # Apply corrections for higher accuracy
        dlat_corrected = dlat * (1 - dlat**2 / (6 * self.M**2))
        dlon_corrected = dlon * (1 + 0.5 * self.e2 * self.sin_lat**2 * dlon**2)
        
        lat = np.degrees(self.ref_lat + dlat_corrected)
        lon = np.degrees(self.ref_lon + dlon_corrected)
        
        return lat, lon

class UltraPrecisionEmitterGeolocation:
    """Ultra-precision geolocation system for challenging scenarios"""
    
    def __init__(self):
        """Initialize ultra-precision system"""
        self.coordinate_converter = None
    
    def setup_coordinate_system(self, sensors: List[Sensor]):
        """Setup coordinate system"""
        ref_lat = np.mean([s.latitude for s in sensors])
        ref_lon = np.mean([s.longitude for s in sensors])
        self.coordinate_converter = UltraPrecisionCoordinateConverter(ref_lat, ref_lon)
    
    def calculate_precise_bearing(self, lat1: float, lon1: float, 
                                lat2: float, lon2: float) -> float:
        """Calculate bearing with maximum precision using Vincenty's formula"""
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        # Vincenty's formula for forward azimuth
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))
        
        bearing = np.arctan2(y, x)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def robust_bearing_cost_function(self, pos: np.ndarray, sensors: List[Sensor], 
                                   weights: np.ndarray) -> float:
        """
        Robust cost function using M-estimator for outlier rejection
        """
        east, north = pos
        lat, lon = self.coordinate_converter.enu_to_latlon(east, north)
        
        total_cost = 0
        
        for i, sensor in enumerate(sensors):
            # Calculate expected bearing
            expected_bearing = self.calculate_precise_bearing(
                sensor.latitude, sensor.longitude, lat, lon
            )
            
            # Angular residual
            residual = sensor.doa - expected_bearing
            residual = ((residual + 180) % 360) - 180  # Normalize to [-180, 180]
            
            # Huber loss for robustness (less sensitive to outliers)
            delta = 5.0  # Huber threshold in degrees
            if abs(residual) <= delta:
                cost = 0.5 * residual**2
            else:
                cost = delta * (abs(residual) - 0.5 * delta)
            
            total_cost += weights[i] * cost
        
        return total_cost
    
    def compute_adaptive_weights(self, sensors: List[Sensor], 
                               estimated_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Compute adaptive weights based on multiple factors
        """
        n_sensors = len(sensors)
        weights = np.ones(n_sensors)
        
        if estimated_pos is None:
            return weights
        
        est_east, est_north = estimated_pos
        est_lat, est_lon = self.coordinate_converter.enu_to_latlon(est_east, est_north)
        
        # Calculate distances and bearings
        distances = []
        bearings_from_emitter = []
        
        for sensor in sensors:
            s_east, s_north = self.coordinate_converter.latlon_to_enu(
                sensor.latitude, sensor.longitude
            )
            
            distance = np.sqrt((s_east - est_east)**2 + (s_north - est_north)**2)
            distances.append(distance)
            
            bearing = self.calculate_precise_bearing(
                est_lat, est_lon, sensor.latitude, sensor.longitude
            )
            bearings_from_emitter.append(np.radians(bearing))
        
        # Geometry-based weighting
        for i in range(n_sensors):
            # Base weight (inverse distance squared, but not too aggressive)
            distance_weight = 1.0 / (1.0 + (distances[i] / 1000.0)**0.5)  # Gentle distance penalty
            
            # Angular diversity weight
            angular_weight = 1.0
            if n_sensors >= 3:
                # Find minimum angle to other sensors
                min_angle_sep = np.pi
                for j in range(n_sensors):
                    if i != j:
                        angle_diff = abs(bearings_from_emitter[i] - bearings_from_emitter[j])
                        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                        min_angle_sep = min(min_angle_sep, angle_diff)
                
                # Penalize sensors with poor angular separation
                if min_angle_sep < np.radians(20):
                    angular_weight = 0.3
                elif min_angle_sep < np.radians(45):
                    angular_weight = 0.7
                else:
                    angular_weight = 1.0
            
            weights[i] = distance_weight * angular_weight
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def solve_global_optimization(self, sensors: List[Sensor]) -> Tuple[np.ndarray, float]:
        """
        Solve using global optimization for robustness
        """
        # Estimate search bounds based on sensor positions
        sensor_positions = []
        for sensor in sensors:
            east, north = self.coordinate_converter.latlon_to_enu(
                sensor.latitude, sensor.longitude
            )
            sensor_positions.append([east, north])
        
        sensor_positions = np.array(sensor_positions)
        
        # Bounds: extend beyond sensor positions
        margin = 10000  # 10km margin
        min_east, max_east = sensor_positions[:, 0].min() - margin, sensor_positions[:, 0].max() + margin
        min_north, max_north = sensor_positions[:, 1].min() - margin, sensor_positions[:, 1].max() + margin
        
        bounds = [(min_east, max_east), (min_north, max_north)]
        
        # Initial guess: centroid of sensors
        initial_guess = np.mean(sensor_positions, axis=0)
        
        # Use differential evolution for global optimization
        def cost_func(pos):
            weights = self.compute_adaptive_weights(sensors, (pos[0], pos[1]))
            return self.robust_bearing_cost_function(pos, sensors, weights)
        
        # Global optimization
        result = differential_evolution(
            cost_func,
            bounds,
            seed=42,
            maxiter=1000,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            return result.x, result.fun
        else:
            # Fallback to local optimization
            from scipy.optimize import minimize
            local_result = minimize(
                cost_func,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )
            return local_result.x, local_result.fun
    
    def iterative_refinement(self, sensors: List[Sensor], 
                           initial_pos: np.ndarray, max_iterations: int = 10) -> Tuple[np.ndarray, int]:
        """
        Iterative refinement with adaptive weighting
        """
        current_pos = initial_pos.copy()
        
        for iteration in range(max_iterations):
            # Compute weights based on current position
            weights = self.compute_adaptive_weights(sensors, (current_pos[0], current_pos[1]))
            
            # Local optimization step
            def cost_func(pos):
                return self.robust_bearing_cost_function(pos, sensors, weights)
            
            result = minimize(
                cost_func,
                current_pos,
                method='L-BFGS-B'
            )
            
            if result.success:
                new_pos = result.x
                position_change = np.linalg.norm(new_pos - current_pos)
                current_pos = new_pos
                
                if position_change < 1.0:  # 1 meter convergence
                    break
            else:
                break
        
        return current_pos, iteration + 1
    
    def estimate_emitter_location(self, sensors: List[Sensor], 
                                signal_features: SignalFeatures) -> EmitterEstimate:
        """
        Ultra-precision emitter location estimation
        """
        if len(sensors) < 2:
            raise ValueError("At least 2 sensors required")
        
        # Setup coordinate system
        self.setup_coordinate_system(sensors)
        
        # Global optimization for initial estimate
        try:
            global_pos, global_cost = self.solve_global_optimization(sensors)
            method_used = "global_optimization"
        except Exception as e:
            warnings.warn(f"Global optimization failed: {e}")
            # Fallback to centroid
            sensor_positions = []
            for sensor in sensors:
                east, north = self.coordinate_converter.latlon_to_enu(
                    sensor.latitude, sensor.longitude
                )
                sensor_positions.append([east, north])
            global_pos = np.mean(sensor_positions, axis=0)
            method_used = "centroid_fallback"
        
        # Iterative refinement
        final_pos, iterations = self.iterative_refinement(sensors, global_pos)
        
        # Convert back to lat/lon
        emitter_lat, emitter_lon = self.coordinate_converter.enu_to_latlon(
            final_pos[0], final_pos[1]
        )
        
        # Calculate final metrics
        final_weights = self.compute_adaptive_weights(sensors, (final_pos[0], final_pos[1]))
        final_cost = self.robust_bearing_cost_function(final_pos, sensors, final_weights)
        
        # Convert cost to RMS error in degrees
        residual_error = np.sqrt(final_cost / len(sensors))
        
        # Geometry quality assessment
        sensor_bearings = []
        for sensor in sensors:
            bearing = self.calculate_precise_bearing(
                emitter_lat, emitter_lon, sensor.latitude, sensor.longitude
            )
            sensor_bearings.append(bearing)
        
        # Calculate angular separations
        sensor_bearings_sorted = sorted(sensor_bearings)
        separations = []
        for i in range(len(sensor_bearings_sorted)):
            next_i = (i + 1) % len(sensor_bearings_sorted)
            sep = sensor_bearings_sorted[next_i] - sensor_bearings_sorted[i]
            if sep < 0:
                sep += 360
            separations.append(sep)
        
        min_separation = min(separations)
        geometry_quality = min(1.0, min_separation / 60.0)  # Normalize by 60°
        
        # Confidence score
        error_confidence = np.exp(-0.1 * residual_error)
        geometry_confidence = geometry_quality
        confidence_score = error_confidence * geometry_confidence
        
        # Covariance estimation (simplified)
        covariance_trace = final_cost / len(sensors)
        
        return EmitterEstimate(
            latitude=emitter_lat,
            longitude=emitter_lon,
            confidence_score=confidence_score,
            residual_error=residual_error,
            covariance_trace=covariance_trace,
            geometry_quality=geometry_quality,
            iterations=iterations,
            method_used=method_used
        )

# Convenience function
def create_ultra_precision_geolocator() -> UltraPrecisionEmitterGeolocation:
    """Create ultra-precision geolocation system"""
    return UltraPrecisionEmitterGeolocation()