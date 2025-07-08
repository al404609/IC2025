"""
IBVS Simulation Engine
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from machinevisiontoolbox import *
from spatialmath import *

from core.data_handler import IBVSDataPoint
from config.settings import DataConfig

logger = logging.getLogger(__name__)

@dataclass
class CameraSetup:
    """Camera configuration for IBVS"""
    camera: CentralCamera = field(default=None)
    min_distance: float = field(default=DataConfig.MIN_DISTANCE)
    max_distance: float = field(default=DataConfig.MAX_DISTANCE)
    
    def __post_init__(self):
        if self.camera is None:
            self.reset_camera()
    
    def reset_camera(self):
        """Reset camera to random position"""
        x = np.random.uniform(-self.max_distance, self.max_distance)
        y = np.random.uniform(-self.max_distance, self.max_distance)
        z = np.random.uniform(-self.max_distance, -self.min_distance)
        
        self.camera = CentralCamera.Default(pose=SE3.Trans(x, y, z))

@dataclass
class TargetSetup:
    """Target configuration for IBVS"""
    world_points: np.ndarray = field(default=None)
    desired_points: np.ndarray = field(default=None)
    grid_size: int = field(default=DataConfig.GRID_SIZE)
    grid_side: float = field(default=DataConfig.GRID_SIDE)
    min_distance: float = field(default=DataConfig.MIN_DISTANCE)
    max_distance: float = field(default=DataConfig.MAX_DISTANCE)
    
    def generate_target_points(self):
        """Generate random target points"""
        x = np.random.uniform(-self.max_distance, self.max_distance)
        y = np.random.uniform(-self.max_distance, self.max_distance)
        z = np.random.uniform(self.min_distance, self.max_distance)
        
        self.world_points = mkgrid(self.grid_size, side=self.grid_side, pose=SE3.Trans(x, y, z))
    
    def set_desired_pattern(self, camera: CentralCamera):
        """Set desired image pattern"""
        self.desired_points = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]

class SimulationEngine:
    """Main simulation engine for IBVS"""
    
    def __init__(self, max_iterations: int = DataConfig.MAX_ITERATIONS,
                 lambda_gain: float = DataConfig.LAMBDA_VALUE,
                 convergence_threshold: float = DataConfig.CONVERGENCE_THRESHOLD):
        self.max_iterations = max_iterations
        self.lambda_gain = lambda_gain
        self.convergence_threshold = convergence_threshold
        
        self.camera_setup = CameraSetup()
        self.target_setup = TargetSetup()
        
        self.current_controller = None
        self.data_points = []
        
        logger.info(f"Simulation engine initialized with {max_iterations} max iterations")
    
    def prepare_simulation(self):
        """Prepare a new simulation with random configuration"""
        self.camera_setup.reset_camera()
        self.target_setup.generate_target_points()
        self.target_setup.set_desired_pattern(self.camera_setup.camera)
        
        self.current_controller = IBVS(
            camera=self.camera_setup.camera,
            P=self.target_setup.world_points,
            p_d=self.target_setup.desired_points,
            graphics=False
        )
    
    def run_single_simulation(self, sequence_id: int) -> List[IBVSDataPoint]:
        """Run a single IBVS simulation sequence"""
        if self.current_controller is None:
            raise ValueError("Simulation not prepared. Call prepare_simulation() first.")
        
        data_points = []
        
        for step in range(self.max_iterations):
            self.current_controller.step(step)
            
            try:
                data_point = IBVSDataPoint.from_simulation(
                    self.current_controller, sequence_id, step
                )
                data_points.append(data_point)
                
            except Exception as e:
                logger.error(f"Error creating data point at step {step}: {e}")
                continue
            
            if len(self.current_controller.history) > 0:
                error_norm = np.linalg.norm(self.current_controller.history[-1].e)
                if error_norm < self.convergence_threshold:
                    break
        
        logger.info(f"Simulation {sequence_id} completed with {len(data_points)} data points")
        return data_points
    
    def run_batch_simulations(self, num_sequences: int, start_sequence: int = 0) -> List[IBVSDataPoint]:
        """Run a batch of simulations"""
        all_data_points = []
        
        for i in range(num_sequences):
            sequence_id = start_sequence + i
            
            try:
                self.prepare_simulation()
                sequence_data = self.run_single_simulation(sequence_id)
                all_data_points.extend(sequence_data)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{num_sequences} simulations")
                    
            except Exception as e:
                logger.error(f"Error in simulation {sequence_id}: {e}")
                continue
        
        logger.info(f"Batch simulation completed. Generated {len(all_data_points)} total data points")
        return all_data_points
    
    def get_simulation_statistics(self) -> dict:
        """Get statistics about the current simulation setup"""
        if self.current_controller is None:
            return {}
        
        stats = {
            'max_iterations': self.max_iterations,
            'lambda_gain': self.lambda_gain,
            'convergence_threshold': self.convergence_threshold,
            'camera_pose': str(self.camera_setup.camera.pose),
            'target_points_shape': self.target_setup.world_points.shape if self.target_setup.world_points is not None else None,
            'desired_points_shape': self.target_setup.desired_points.shape if self.target_setup.desired_points is not None else None
        }
        
        return stats 