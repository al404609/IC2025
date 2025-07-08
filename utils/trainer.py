"""
Neural Network Training Framework for IBVS
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
import logging
from typing import Dict, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
from models.architectures import ModelFactory
from core.data_handler import DatasetReader

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scalers and transform data"""
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_targets = self.target_scaler.fit_transform(targets)
        self.is_fitted = True
        return scaled_features, scaled_targets
    
    def transform(self, features: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_transform first.")
        
        scaled_features = self.feature_scaler.transform(features)
        scaled_targets = self.target_scaler.transform(targets) if targets is not None else None
        
        return scaled_features, scaled_targets
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets"""
        return self.target_scaler.inverse_transform(targets)
    
    def save_scalers(self, filepath: str):
        """Save scalers to file"""
        with open(f"{filepath}_feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(f"{filepath}_target_scaler.pkl", 'wb') as f:
            pickle.dump(self.target_scaler, f)
    
    def load_scalers(self, filepath: str):
        """Load scalers from file"""
        with open(f"{filepath}_feature_scaler.pkl", 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f"{filepath}_target_scaler.pkl", 'rb') as f:
            self.target_scaler = pickle.load(f)
        self.is_fitted = True

class ModelTrainer:
    """Main training class for neural networks"""
    
    def __init__(self, model_type: str, model_config: dict, 
                 training_config: dict = None, device: str = 'cuda'):
        
        self.model_type = model_type
        self.model_config = model_config
        self.training_config = training_config or {}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = ModelFactory.create_model(model_type, model_config)
        self.model.to(self.device)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Model created: {self.model_type}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def train(self, dataset_path: str, output_dir: str = "models") -> Dict[str, Any]:
        """Train the model"""
        
        # Load and prepare data
        reader = DatasetReader(dataset_path)
        features, targets = reader.get_features_and_targets()
        
        # Preprocess data
        features_norm, targets_norm = self.preprocessor.fit_transform(features, targets)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features_norm)
        targets_tensor = torch.FloatTensor(targets_norm)
        
        # Create dataset
        dataset = TensorDataset(features_tensor, targets_tensor)
        
        # Split data
        val_split = self.training_config.get('val_split', TrainingConfig.VAL_SPLIT)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        batch_size = self.training_config.get('batch_size', TrainingConfig.BATCH_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.get('learning_rate', TrainingConfig.LEARNING_RATE)
        )
        
        num_epochs = self.training_config.get('num_epochs', TrainingConfig.NUM_EPOCHS)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch
                
                # Save model
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), 
                          os.path.join(output_dir, f"{self.model_type}_best.pth"))
            
            epoch_time = time.time() - epoch_start_time
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, "
                           f"Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {avg_val_loss:.6f}, "
                           f"Time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Save final artifacts
        self.save_training_artifacts(output_dir)
        
        # Return training metrics
        return {
            'num_epochs': num_epochs,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'total_time': total_time,
            'avg_epoch_time': total_time / num_epochs
        }
    
    def save_training_artifacts(self, output_dir: str):
        """Save training artifacts"""
        base_path = os.path.join(output_dir, self.model_type)
        
        # Save scalers
        self.preprocessor.save_scalers(base_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1],
                'total_epochs': len(self.train_losses)
            }
        }
        
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training artifacts saved to {output_dir}")

def train_best_model(dataset_path: str, output_dir: str = "models"):
    """Train the best performing model configuration"""
    
    # Best configuration: current_desired -> linear_only with FNN
    model_config = {
        'input_size': 16,  # current_desired features
        'hidden_sizes': [64, 32],
        'output_size': 3,  # linear_only targets
        'dropout_rate': 0.3,
        'use_batch_norm': True
    }
    
    training_config = {
        'batch_size': 512,
        'num_epochs': 150,
        'learning_rate': 0.0005,
        'val_split': 0.2
    }
    
    trainer = ModelTrainer('fnn', model_config, training_config)
    
    logger.info("Starting training of best model configuration...")
    metrics = trainer.train(dataset_path, output_dir)
    logger.info(f"Training completed. Best validation loss: {metrics['best_val_loss']:.6f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train IBVS neural network')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV file')
    parser.add_argument('--output', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    train_best_model(args.dataset, args.output) 