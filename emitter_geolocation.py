"""
Emitter Geolocation System using Direction of Arrival (DOA)

Production-ready bearing-only localization using Weighted Least Squares (WLS).
Input: Sensor positions, DOA measurements, signal features
Output: Emitter latitude, longitude, confidence score
"""

import numpy as np
import pyproj
from typing import List, Tuple
from dataclasses import dataclass
import warnings

@dataclass
class Sensor:
    """Represents a sensor with position and DOA measurement"""
    sensor_id: str
    latitude: float
    longitude: float
    doa: float  # Direction of arrival in degrees
    
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

class CoordinateConverter:
    """Handles coordinate system transformations"""
    
    def __init__(self, reference_lat: float, reference_lon: float):
        """
        Initialize coordinate converter with reference point
        
        Args:
            reference_lat: Reference latitude for local coordinate system
            reference_lon: Reference longitude for local coordinate system
        """
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        
        # Create UTM projection based on reference point
        utm_zone = int((reference_lon + 180) / 6) + 1
        hemisphere = 'north' if reference_lat >= 0 else 'south'
        
        # Create transformer using modern pyproj API
        self.wgs84_crs = pyproj.CRS('EPSG:4326')  # WGS84
        self.utm_crs = pyproj.CRS(
            proj='utm',
            zone=utm_zone,
            datum='WGS84',
            units='m',
            south=(hemisphere == 'south')
        )
        
        self.to_utm = pyproj.Transformer.from_crs(
            self.wgs84_crs, self.utm_crs, always_xy=True
        )
        self.from_utm = pyproj.Transformer.from_crs(
            self.utm_crs, self.wgs84_crs, always_xy=True
        )
        
        # Store reference point in UTM
        self.ref_x, self.ref_y = self.to_utm.transform(reference_lon, reference_lat)
    
    def latlon_to_cartesian(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local Cartesian coordinates"""
        x_utm, y_utm = self.to_utm.transform(lon, lat)
        return x_utm - self.ref_x, y_utm - self.ref_y
    
    def cartesian_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local Cartesian coordinates to lat/lon"""
        x_utm = x + self.ref_x
        y_utm = y + self.ref_y
        lon, lat = self.from_utm.transform(x_utm, y_utm)
        return lat, lon

class DOAUncertaintyModel:
    """Estimates DOA uncertainty based on signal features"""
    
    def __init__(self):
        """Initialize DOA uncertainty model with heuristic parameters"""
        # Heuristic parameters (degrees)
        self.base_sigma = 2.0  # Base angular error
        self.freq_factor = 1e-9  # Frequency dependency
        self.prf_factor = 1e-6   # PRF dependency
        self.pw_factor = 1e6     # Pulse width dependency
    
    def estimate_doa_sigma(self, signal_features: SignalFeatures) -> float:
        """
        Estimate DOA uncertainty (standard deviation in degrees)
        
        Args:
            signal_features: Signal characteristics
            
        Returns:
            DOA uncertainty in degrees
        """
        # Heuristic model
        freq_term = self.freq_factor * signal_features.frequency
        prf_term = self.prf_factor * signal_features.prf
        pw_term = self.pw_factor * signal_features.pulse_width
        
        sigma = self.base_sigma + freq_term + prf_term + pw_term
        return max(0.1, min(10.0, sigma))  # Clamp between 0.1 and 10 degrees

class EmitterGeolocation:
    """Main geolocation system"""
    
    def __init__(self):
        """Initialize geolocation system"""
        self.uncertainty_model = DOAUncertaintyModel()
        self.coordinate_converter = None
        
        # Confidence scoring parameters
        self.alpha = 0.001  # Residual error weight
        self.beta = 10.0    # Geometry weight
    
    def setup_coordinate_system(self, sensors: List[Sensor]):
        """Setup coordinate system based on sensor positions"""
        # Use centroid of sensors as reference point
        ref_lat = np.mean([s.latitude for s in sensors])
        ref_lon = np.mean([s.longitude for s in sensors])
        
        self.coordinate_converter = CoordinateConverter(ref_lat, ref_lon)
    
    def build_wls_matrices(self, sensors: List[Sensor], weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build matrices for Weighted Least Squares solution
        
        Mathematical model: (x - xi) * sin(theta_i) - (y - yi) * cos(theta_i) = 0
        Linearized as: A * p = b, where p = [x, y]^T
        
        Args:
            sensors: List of sensor objects
            weights: Weight for each sensor measurement
            
        Returns:
            A matrix, b vector, W weight matrix
        """
        n_sensors = len(sensors)
        A = np.zeros((n_sensors, 2))
        b = np.zeros(n_sensors)
        
        for i, sensor in enumerate(sensors):
            # Convert sensor position to Cartesian
            xi, yi = self.coordinate_converter.latlon_to_cartesian(
                sensor.latitude, sensor.longitude
            )
            
            # Convert DOA to radians
            theta_i = np.radians(sensor.doa)
            
            # Build matrix row: sin(theta_i) * x - cos(theta_i) * y = xi * sin(theta_i) - yi * cos(theta_i)
            A[i, 0] = np.sin(theta_i)
            A[i, 1] = -np.cos(theta_i)
            b[i] = xi * np.sin(theta_i) - yi * np.cos(theta_i)
        
        # Weight matrix
        W = np.diag(weights)
        
        return A, b, W
    
    def solve_wls(self, A: np.ndarray, b: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve Weighted Least Squares: p_hat = (A^T W A)^(-1) A^T W b
        
        Args:
            A: Design matrix
            b: Observation vector
            W: Weight matrix
            
        Returns:
            Position estimate, covariance matrix, residual error
        """
        try:
            # Compute weighted normal equations
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b
            
            # Check for numerical stability
            cond_num = np.linalg.cond(AtWA)
            if cond_num > 1e12:
                warnings.warn(f"Ill-conditioned matrix (condition number: {cond_num:.2e})")
            
            # Solve for position
            p_hat = np.linalg.solve(AtWA, AtWb)
            
            # Compute covariance matrix
            cov_matrix = np.linalg.inv(AtWA)
            
            # Compute residual error
            residual = A @ p_hat - b
            residual_error = np.sqrt(np.mean(residual**2))
            
            return p_hat, cov_matrix, residual_error
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to solve WLS equations: {e}")
    
    def compute_confidence_score(self, residual_error: float, cov_matrix: np.ndarray) -> float:
        """
        Compute confidence score based on residual error and geometry
        
        Args:
            residual_error: RMS residual error
            cov_matrix: Covariance matrix
            
        Returns:
            Confidence score between 0 and 1
        """
        # Geometry factor (trace of covariance matrix)
        geometry_factor = np.trace(cov_matrix)
        
        # Confidence function
        confidence = np.exp(-self.alpha * residual_error) * np.exp(-self.beta * geometry_factor)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def estimate_emitter_location(self, sensors: List[Sensor], signal_features: SignalFeatures) -> EmitterEstimate:
        """
        Main function to estimate emitter location
        
        Args:
            sensors: List of sensor measurements
            signal_features: Common signal characteristics
            
        Returns:
            EmitterEstimate with location and confidence
        """
        if len(sensors) < 2:
            raise ValueError("At least 2 sensors required for triangulation")
        
        # Setup coordinate system
        self.setup_coordinate_system(sensors)
        
        # Estimate DOA uncertainties and compute weights
        doa_sigma = self.uncertainty_model.estimate_doa_sigma(signal_features)
        weights = np.ones(len(sensors)) / (np.radians(doa_sigma)**2)  # weight = 1/sigma^2
        
        # Build and solve WLS system
        A, b, W = self.build_wls_matrices(sensors, weights)
        p_hat, cov_matrix, residual_error = self.solve_wls(A, b, W)
        
        # Convert back to lat/lon
        emitter_lat, emitter_lon = self.coordinate_converter.cartesian_to_latlon(p_hat[0], p_hat[1])
        
        # Compute confidence score
        confidence_score = self.compute_confidence_score(residual_error, cov_matrix)
        
        return EmitterEstimate(
            latitude=emitter_lat,
            longitude=emitter_lon,
            confidence_score=confidence_score,
            residual_error=residual_error,
            covariance_trace=np.trace(cov_matrix)
        )