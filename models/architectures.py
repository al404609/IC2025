"""
Neural Network Architectures for IBVS Control
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNet(nn.Module):
    """Feedforward Neural Network for IBVS - Best performing architecture"""
    
    def __init__(self, input_size=16, hidden_sizes=[64, 32], output_size=3, 
                 dropout_rate=0.3, use_batch_norm=True):
        super(FeedForwardNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class LSTMNet(nn.Module):
    """LSTM Neural Network for IBVS"""
    
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, output_size=3, 
                 dropout_rate=0.2):
        super(LSTMNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        for module in self.fc_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc_head(lstm_out[:, -1, :])
        
        return out

class ResidualBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    """ResNet for IBVS"""
    
    def __init__(self, input_size=16, hidden_size=64, num_blocks=3, output_size=3, 
                 dropout_rate=0.2):
        super(ResNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize ResNet weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        out = self.output_head(x)
        
        return out

class ModelFactory:
    """Factory class for creating neural network models"""
    
    @staticmethod
    def create_model(model_type: str, config: dict):
        """Create a model based on type and configuration"""
        if model_type.lower() == 'fnn':
            return FeedForwardNet(**config)
        elif model_type.lower() == 'lstm':
            return LSTMNet(**config)
        elif model_type.lower() == 'resnet':
            return ResNet(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        } 