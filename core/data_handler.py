"""
Data handling and storage module for IBVS training data
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import csv
import logging
from dataclasses import dataclass
from typing import List, Union

from machinevisiontoolbox import *
from spatialmath import *

logger = logging.getLogger(__name__)

@dataclass
class IBVSDataPoint:
    """Single data point for IBVS training"""
    sequence: int = 0
    step: int = 0
    current_features: np.ndarray = None
    desired_features: np.ndarray = None
    velocity_cmd: np.ndarray = None
    error_norm: float = 0.0
    
    def __post_init__(self):
        if self.current_features is None:
            self.current_features = np.zeros(8)
        if self.desired_features is None:
            self.desired_features = np.zeros(8)
        if self.velocity_cmd is None:
            self.velocity_cmd = np.zeros(6)
    
    def set_features(self, current_points: np.ndarray, desired_points: np.ndarray):
        """Set feature points from 2xN arrays"""
        self.current_features = current_points.flatten(order='F')
        self.desired_features = desired_points.flatten(order='F')
    
    def set_velocity(self, velocity: np.ndarray):
        """Set velocity command"""
        self.velocity_cmd = np.array(velocity)
    
    def calculate_error(self):
        """Calculate and set error norm"""
        self.error_norm = np.linalg.norm(self.current_features - self.desired_features)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export"""
        data = {
            'sequence': self.sequence,
            'step': self.step,
            'error_norm': self.error_norm
        }
        
        for i, val in enumerate(self.current_features):
            data[f'current_f{i}'] = val
        for i, val in enumerate(self.desired_features):
            data[f'desired_f{i}'] = val
        for i, val in enumerate(self.velocity_cmd):
            data[f'vel_{i}'] = val
            
        return data
    
    @classmethod
    def from_simulation(cls, simulation_step, seq_id: int, step_id: int):
        """Create data point from simulation step"""
        point = cls()
        point.sequence = seq_id
        point.step = step_id
        
        try:
            current_points = simulation_step.camera.project_point(P=simulation_step.P, pose=simulation_step.camera.pose)
            point.current_features = current_points.flatten(order='F')
            
            point.desired_features = simulation_step.p_star.flatten(order='F')
            
            if len(simulation_step.history) > 0:
                point.velocity_cmd = simulation_step.history[-1].vel
            
            point.calculate_error()
            
        except Exception as e:
            logger.error(f"Error creating data point from simulation: {e}")
            raise
        
        return point

class DatasetWriter:
    """Handles writing dataset to file"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.data_points = []
        self.file_counter = 0
        self.current_file = None
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def add_data_point(self, data_point: IBVSDataPoint):
        """Add a single data point"""
        self.data_points.append(data_point)
    
    def add_data_points(self, data_points: List[IBVSDataPoint]):
        """Add multiple data points"""
        self.data_points.extend(data_points)
    
    def save_batch(self, batch_size: int = 1000):
        """Save a batch of data points to file"""
        if len(self.data_points) < batch_size:
            return
        
        batch = self.data_points[:batch_size]
        self.data_points = self.data_points[batch_size:]
        
        df_data = [point.to_dict() for point in batch]
        df = pd.DataFrame(df_data)
        
        filename = f"{self.output_path}_batch_{self.file_counter}.csv"
        
        df.to_csv(filename, index=False)
        logger.info(f"Saved batch of {len(batch)} data points to {filename}")
        
        self.file_counter += 1
    
    def save_all(self):
        """Save all remaining data points"""
        if len(self.data_points) > 0:
            df_data = [point.to_dict() for point in self.data_points]
            df = pd.DataFrame(df_data)
            
            filename = f"{self.output_path}_final.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved final batch of {len(self.data_points)} data points to {filename}")
            
            self.data_points.clear()
    
    def get_headers(self) -> List[str]:
        """Get CSV headers"""
        headers = ['sequence', 'step']
        headers.extend([f'current_f{i}' for i in range(8)])
        headers.extend([f'desired_f{i}' for i in range(8)])
        headers.extend([f'vel_{i}' for i in range(6)])
        headers.append('error_norm')
        return headers

class DatasetReader:
    """Handles reading dataset from file"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with {len(self.data)} samples from {self.data_path}")
        return self.data
    
    def get_features_and_targets(self, feature_type: str = None, target_type: str = None) -> tuple:
        """Extract features and targets from dataset"""
        if self.data is None:
            self.load_data()
        
        from config.settings import FeatureConfig, TargetConfig
        
        feature_cols = FeatureConfig.get_feature_columns(feature_type)
        target_cols = TargetConfig.get_target_columns(target_type)
        
        if (feature_type or FeatureConfig.FEATURE_TYPE) == 'error_only':
            current_cols = [f'current_f{i}' for i in range(8)]
            desired_cols = [f'desired_f{i}' for i in range(8)]
            
            current_features = self.data[current_cols].values
            desired_features = self.data[desired_cols].values
            features = current_features - desired_features
        else:
            features = self.data[feature_cols].values
        
        targets = self.data[target_cols].values
        
        logger.info(f"Features extracted: {features.shape} using type '{feature_type or FeatureConfig.FEATURE_TYPE}'")
        logger.info(f"Targets extracted: {targets.shape} using type '{target_type or TargetConfig.TARGET_TYPE}'")
        
        return features, targets
    
    def get_sequence_data(self, sequence_id: int) -> pd.DataFrame:
        """Get data for a specific sequence"""
        if self.data is None:
            self.load_data()
        
        return self.data[self.data['sequence'] == sequence_id]
    
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        if self.data is None:
            self.load_data()
        
        stats = {
            'total_samples': len(self.data),
            'num_sequences': self.data['sequence'].nunique(),
            'avg_sequence_length': self.data.groupby('sequence').size().mean(),
            'feature_stats': {},
            'target_stats': {}
        }
        
        feature_cols = [f'current_f{i}' for i in range(8)]
        for col in feature_cols:
            stats['feature_stats'][col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max()
            }
        
        target_cols = ['vel_0', 'vel_1', 'vel_2']
        for col in target_cols:
            stats['target_stats'][col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max()
            }
        
        return stats 