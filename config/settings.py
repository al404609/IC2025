"""
Central configuration file for IBVS project
"""
import os

# Global settings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42
DEVICE = 'cuda'

# Data generation settings
class DataConfig:
    MIN_DISTANCE = 0.5
    MAX_DISTANCE = 2.0
    MAX_ITERATIONS = 300
    LAMBDA_VALUE = 0.1
    GRID_SIZE = 2
    GRID_SIDE = 0.5
    CONVERGENCE_THRESHOLD = 1e-3
    SAVE_BATCH_SIZE = 100

# Training settings
class TrainingConfig:
    BATCH_SIZE = 512
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.0005
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-5

# Feature extraction options
class FeatureConfig:
    FEATURE_TYPE = 'current_desired'  # Best performing configuration
    
    @staticmethod
    def get_feature_columns(feature_type: str = None):
        """Get feature columns based on type"""
        if feature_type is None:
            feature_type = FeatureConfig.FEATURE_TYPE
            
        if feature_type == 'current_only':
            return [f'current_f{i}' for i in range(8)]
        elif feature_type == 'current_desired':
            features = [f'current_f{i}' for i in range(8)]
            features.extend([f'desired_f{i}' for i in range(8)])
            return features
        elif feature_type == 'error_only':
            return ['error_norm'] + [f'current_f{i}' for i in range(8)]
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    @staticmethod
    def get_input_size(feature_type: str = None):
        """Get input size for model based on feature type"""
        if feature_type is None:
            feature_type = FeatureConfig.FEATURE_TYPE
            
        if feature_type == 'current_only':
            return 8
        elif feature_type == 'current_desired': 
            return 16
        elif feature_type == 'error_only':
            return 8
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

# Target extraction options  
class TargetConfig:
    TARGET_TYPE = 'linear_only'  # Best performing configuration
    
    @staticmethod
    def get_target_columns(target_type: str = None):
        """Get target columns based on type"""
        if target_type is None:
            target_type = TargetConfig.TARGET_TYPE
            
        if target_type == 'linear_only':
            return ['vel_0', 'vel_1', 'vel_2']
        elif target_type == 'all_velocities':
            return [f'vel_{i}' for i in range(6)]
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    @staticmethod
    def get_output_size(target_type: str = None):
        """Get output size for model based on target type"""
        if target_type is None:
            target_type = TargetConfig.TARGET_TYPE
            
        if target_type == 'linear_only':
            return 3
        elif target_type == 'all_velocities':
            return 6
        else:
            raise ValueError(f"Unknown target type: {target_type}")

# Model architectures
class ModelArchitectures:
    FNN = {
        'input_size': 16,  # Optimized for current_desired features
        'hidden_sizes': [64, 32],
        'output_size': 3,
        'dropout_rate': 0.3,
        'use_batch_norm': True
    }
    
    LSTM = {
        'input_size': 16,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 3,
        'dropout_rate': 0.2,
        'sequence_length': 10
    }
    
    RESNET = {
        'input_size': 16,
        'hidden_size': 64,
        'num_blocks': 3,
        'output_size': 3,
        'dropout_rate': 0.2
    }

# File paths
class Paths:
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained')
    EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
    
    @staticmethod
    def ensure_dirs():
        """Create necessary directories if they don't exist"""
        for path in [Paths.DATA_DIR, Paths.MODELS_DIR, Paths.EXPERIMENTS_DIR]:
            os.makedirs(path, exist_ok=True) 