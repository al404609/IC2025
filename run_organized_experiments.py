#!/usr/bin/env python3
"""
Script de Experimentación Organizada IBVS
Ejecuta experimentos sistemáticos y organiza resultados con documentación
"""
import os
import sys
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrganizedExperimentRunner:
    """Ejecutor de experimentos organizados"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.results_dir = self.base_dir / "experiments_results"
        
        # Configuraciones disponibles
        self.feature_configs = ['current_only', 'current_desired', 'error_only']
        self.target_configs = ['linear_only', 'all_velocities']
        self.model_types = ['fnn']  # Expandible a ['fnn', 'lstm', 'resnet']
        
        # Resultados
        self.experiment_results = {}
        
        logger.info(f"Organizador de experimentos inicializado en: {self.results_dir}")
    
    def create_experiment_structure(self, feature_config: str, target_config: str):
        """Crear estructura de directorios para un experimento"""
        exp_name = f"{feature_config}_{target_config}"
        exp_dir = self.results_dir / exp_name
        
        # Crear directorios
        (exp_dir / "modelo").mkdir(parents=True, exist_ok=True)
        (exp_dir / "graficos").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        return exp_dir
    
    def run_training(self, feature_config: str, target_config: str, model_type: str) -> bool:
        """Ejecutar entrenamiento para una configuración específica"""
        logger.info(f"Entrenando: {feature_config} -> {target_config} [{model_type}]")
        
        # Configurar PYTHONPATH
        pythonpath = f"export PYTHONPATH={self.base_dir}:$PYTHONPATH"
        
        # Comando de entrenamiento
        cmd = f"{pythonpath} && python quick_feature_test.py --feature {feature_config} --target {target_config} --model {model_type}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info(f"Entrenamiento exitoso: {feature_config} -> {target_config}")
                return True
            else:
                logger.error(f"Entrenamiento falló: {feature_config} -> {target_config}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error ejecutando entrenamiento: {e}")
            return False
    
    def run_comparison(self, feature_config: str, target_config: str, model_type: str) -> dict:
        """Ejecutar comparación y extraer métricas"""
        logger.info(f"Comparando: {feature_config} -> {target_config} vs clásico")
        
        model_path = self.base_dir / "test_features" / f"{feature_config}_{target_config}_{model_type}"
        
        # Configurar PYTHONPATH
        pythonpath = f"export PYTHONPATH={self.base_dir}:$PYTHONPATH"
        
        # Comando de comparación
        cmd = f"{pythonpath} && python classical_vs_ml_comparison.py --model-path {model_path} --model-type {model_type}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info(f"Comparación exitosa: {feature_config} -> {target_config}")
                
                # Extraer métricas del output
                metrics = self.extract_metrics_from_output(result.stdout)
                return metrics
            else:
                logger.error(f"Comparación falló: {feature_config} -> {target_config}")
                return {'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error ejecutando comparación: {e}")
            return {'error': str(e)}
    
    def extract_metrics_from_output(self, output: str) -> dict:
        """Extraer métricas del output de comparación"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            if "IBVS Clásico:" in line and "pasos" in line.lower():
                try:
                    metrics['classical_steps'] = int(line.split(':')[1].strip())
                except:
                    pass
            
            elif "IBVS ML:" in line and "pasos" in line.lower():
                try:
                    metrics['ml_steps'] = int(line.split(':')[1].strip())
                except:
                    pass
            
            elif "IBVS Clásico:" in line and "error" in line.lower():
                try:
                    metrics['classical_error'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif "IBVS ML:" in line and "error" in line.lower():
                try:
                    metrics['ml_error'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif "ML es" in line and "mejor" in line:
                try:
                    improvement = line.split('%')[0].split()[-1]
                    metrics['improvement_percent'] = float(improvement)
                    metrics['ml_better'] = True
                except:
                    pass
            
            elif "Clásico es" in line and "mejor" in line:
                try:
                    improvement = line.split('%')[0].split()[-1]
                    metrics['improvement_percent'] = -float(improvement)
                    metrics['ml_better'] = False
                except:
                    pass
        
        # Calcular métricas adicionales
        if 'classical_error' in metrics and 'ml_error' in metrics:
            if metrics['classical_error'] > 0:
                improvement = (metrics['classical_error'] - metrics['ml_error']) / metrics['classical_error'] * 100
                metrics['calculated_improvement'] = improvement
        
        return metrics
    
    def organize_files(self, feature_config: str, target_config: str, model_type: str):
        """Organizar archivos de entrenamiento y gráficos"""
        exp_name = f"{feature_config}_{target_config}"
        exp_dir = self.results_dir / exp_name
        
        # Mover archivos de modelo
        model_source = self.base_dir / "test_features" / f"{feature_config}_{target_config}_{model_type}"
        model_dest = exp_dir / "modelo"
        
        if model_source.exists():
            for file in model_source.glob("*"):
                shutil.copy2(file, model_dest / file.name)
            logger.info(f"Archivos de modelo copiados a {model_dest}")
        
        # Mover gráficos
        graphics_source = self.base_dir / ".resources"
        graphics_dest = exp_dir / "graficos"
        
        if graphics_source.exists():
            for file in graphics_source.glob("*.png"):
                shutil.copy2(file, graphics_dest / file.name)
            logger.info(f"Gráficos copiados a {graphics_dest}")
    
    def get_architecture_description(self, model_type: str, feature_config: str, target_config: str):
        """Generar descripción detallada de arquitectura según configuración"""
        
        # Mapear configuraciones a dimensiones
        feature_dims = {
            'current_only': 8,
            'current_desired': 16,
            'error_only': 8
        }
        
        target_dims = {
            'linear_only': 3,
            'all_velocities': 6
        }
        
        input_size = feature_dims[feature_config]
        output_size = target_dims[target_config]
        
        # Calcular parámetros específicos
        if model_type.lower() == 'fnn':
            # FNN: input->64->32->output
            params = input_size * 64 + 64 + 64 * 32 + 32 + 32 * output_size + output_size
            
            architecture = f"""FeedForward Network (FNN):
├── Input Layer: {input_size} features
├── Hidden Layer 1: 64 neurons
│   ├── Linear({input_size} → 64)
│   ├── BatchNorm1d(64)
│   ├── ReLU()
│   └── Dropout(0.3)
├── Hidden Layer 2: 32 neurons
│   ├── Linear(64 → 32)
│   ├── BatchNorm1d(32)
│   ├── ReLU()
│   └── Dropout(0.3)
└── Output Layer: {output_size} velocities
    └── Linear(32 → {output_size})

Total Parameters: {params:,}
Weight Initialization: Xavier Uniform
Activation Function: ReLU
Regularization: BatchNorm + Dropout(0.3)"""
        
        elif model_type.lower() == 'lstm':
            # LSTM: más complejo, estimación aproximada
            params = input_size * 64 * 4 * 2 + 64 * 64 * 4 * 2 + 64 * 32 + 32 * output_size
            
            architecture = f"""LSTM Network:
├── Input Layer: {input_size} features (sequence_length: 10)
├── LSTM Block:
│   ├── LSTM Layer 1: 64 hidden units
│   │   ├── Input Gate: sigmoid(W_i × input + b_i)
│   │   ├── Forget Gate: sigmoid(W_f × input + b_f)
│   │   ├── Cell State: tanh(W_c × input + b_c)
│   │   └── Output Gate: sigmoid(W_o × input + b_o)
│   ├── LSTM Layer 2: 64 hidden units
│   │   ├── Input Gate: sigmoid(W_i × input + b_i)
│   │   ├── Forget Gate: sigmoid(W_f × input + b_f)
│   │   ├── Cell State: tanh(W_c × input + b_c)
│   │   └── Output Gate: sigmoid(W_o × input + b_o)
│   └── Dropout(0.2) between layers
├── FC Head:
│   ├── Linear(64 → 32)
│   ├── ReLU()
│   ├── Dropout(0.2)
│   └── Linear(32 → {output_size})
└── Output Layer: {output_size} velocities

Total Parameters: {params:,}
Weight Initialization: Xavier (Linear) + Orthogonal (Recurrent)
Sequence Length: 10 timesteps
Memory: Long Short-Term Memory cells"""
        
        elif model_type.lower() == 'resnet':
            # ResNet: input->64, 3 bloques residuales, 64->32->output
            params = input_size * 64 + 64 + (64 * 64 + 64 + 64 * 64 + 64) * 3 + 64 * 32 + 32 * output_size
            
            architecture = f"""ResNet (Residual Network):
├── Input Layer: {input_size} features
├── Input Projection: Linear({input_size} → 64)
├── Residual Block 1:
│   ├── Main Path:
│   │   ├── Linear(64 → 64)
│   │   ├── BatchNorm1d(64)
│   │   ├── ReLU()
│   │   ├── Dropout(0.2)
│   │   ├── Linear(64 → 64)
│   │   └── BatchNorm1d(64)
│   ├── Skip Connection: Identity
│   └── Add + ReLU()
├── Residual Block 2: (same structure)
├── Residual Block 3: (same structure)
├── Output Head:
│   ├── Linear(64 → 32)
│   ├── ReLU()
│   ├── Dropout(0.2)
│   └── Linear(32 → {output_size})
└── Output Layer: {output_size} velocities

Total Parameters: {params:,}
Weight Initialization: Xavier Uniform
Skip Connections: Identity mapping
Depth: 3 residual blocks"""
        
        else:
            # Fallback genérico
            params = input_size * 64 + 64 * 32 + 32 * output_size
            architecture = f"""Generic Neural Network:
├── Input Layer: {input_size} features
├── Hidden Processing: Model-specific architecture
└── Output Layer: {output_size} velocities

Total Parameters: ~{params:,}
Model Type: {model_type.upper()}"""
        
        return architecture

    def create_experiment_readme(self, feature_config: str, target_config: str, 
                                model_type: str, metrics: dict):
        """Crear README específico para el experimento"""
        exp_name = f"{feature_config}_{target_config}"
        exp_dir = self.results_dir / exp_name
        
        # Descripción de configuraciones
        feature_descriptions = {
            'current_only': {
                'desc': 'Solo Coordenadas Actuales',
                'dim': 8,
                'content': 'Coordenadas actuales de puntos característicos',
                'advantage': 'Simplicidad, menos parámetros',
                'similarity': 'Igual que el proyecto de referencia mihaiBront'
            },
            'current_desired': {
                'desc': 'Coordenadas Completas',
                'dim': 16,
                'content': 'Coordenadas actuales + deseadas',
                'advantage': 'Información completa del estado del sistema',
                'similarity': 'Máxima información disponible'
            },
            'error_only': {
                'desc': 'Solo Errores',
                'dim': 8,
                'content': 'Diferencias entre coordenadas actuales y deseadas',
                'advantage': 'Enfoque directo en corrección de errores',
                'similarity': 'Como IBVS clásico tradicional'
            }
        }
        
        target_descriptions = {
            'linear_only': {
                'desc': 'Solo Velocidades Lineales',
                'dim': 3,
                'content': 'Velocidades lineales [vx, vy, vz]',
                'focus': 'Control translacional únicamente'
            },
            'all_velocities': {
                'desc': 'Velocidades Completas',
                'dim': 6,
                'content': 'Velocidades lineales + angulares',
                'focus': 'Control completo 6DOF'
            }
        }
        
        feature_info = feature_descriptions[feature_config]
        target_info = target_descriptions[target_config]
        
        # Determinar estado del experimento
        if 'error' in metrics:
            status = "ERROR"
            status_desc = f"Error: {metrics['error']}"
        elif metrics.get('ml_better', False):
            improvement = metrics.get('improvement_percent', 0)
            status = f"MEJOR (+{improvement:.1f}%)"
            status_desc = f"ML es {improvement:.1f}% mejor que el método clásico"
        else:
            improvement = abs(metrics.get('improvement_percent', 0))
            status = f"PEOR (-{improvement:.1f}%)"
            status_desc = f"Clásico es {improvement:.1f}% mejor que ML"
        
        # Obtener arquitectura específica
        architecture = self.get_architecture_description(model_type, feature_config, target_config)
        
        # Crear contenido del README
        readme_content = f"""# Experimento: `{feature_config}` -> `{target_config}`

## Descripción del Experimento

**Configuración**: {feature_info['desc']} -> {target_info['desc']}  
**Estado**: {status}

### Configuración de Features (Input)
- **Tipo**: `{feature_config}` 
- **Dimensión**: {feature_info['dim']} valores
- **Contenido**: {feature_info['content']}
- **Ventaja**: {feature_info['advantage']}
- **Similitud**: {feature_info['similarity']}

### Configuración de Targets (Output)  
- **Tipo**: `{target_config}`
- **Dimensión**: {target_info['dim']} valores
- **Contenido**: {target_info['content']}
- **Enfoque**: {target_info['focus']}

## Resultados del Entrenamiento

### Arquitectura del Modelo
```python
{architecture}
```

### Métricas de Entrenamiento
- **Modelo**: {model_type.upper()}
- **Épocas**: 10 (entrenamiento rápido)
- **Batch Size**: 512
- **Learning Rate**: 0.0005
- **Optimizador**: Adam + ReduceLROnPlateau
- **Criterio**: MSE Loss

## Resultados de la Comparación

### Comparación vs IBVS Clásico

| Métrica | IBVS Clásico | IBVS ML | Resultado |
|---------|--------------|---------|-----------|"""

        if 'classical_steps' in metrics and 'ml_steps' in metrics:
            readme_content += f"""
| **Pasos para convergencia** | {metrics['classical_steps']} | {metrics['ml_steps']} | {'+' if metrics['ml_steps'] < metrics['classical_steps'] else '-'}{abs(metrics['ml_steps'] - metrics['classical_steps'])} pasos |"""
        
        if 'classical_error' in metrics and 'ml_error' in metrics:
            readme_content += f"""
| **Error final** | {metrics['classical_error']:.6f} | {metrics['ml_error']:.6f} | {status_desc} |"""

        readme_content += f"""

### RESULTADO
{status_desc}

## Archivos Generados

### Modelo
- `modelo/{model_type}_best.pth` - Estado del modelo entrenado
- `modelo/{model_type}_feature_scaler.pkl` - Escalador de features  
- `modelo/{model_type}_target_scaler.pkl` - Escalador de targets
- `modelo/{model_type}_metadata.json` - Metadatos completos

### Gráficos
- `graficos/classical_ibvs_results.png` - Análisis IBVS clásico
- `graficos/ml_ibvs_results.png` - Análisis IBVS ML  
- `graficos/ibvs_comparison.png` - Comparación lado a lado
- `graficos/error_comparison_detailed.png` - Evolución de error

### Datos
- `resultados.json` - Métricas numéricas completas
- `logs/` - Logs de entrenamiento y comparación

---

*Experimento ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"""

        # Guardar README
        readme_path = exp_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README creado: {readme_path}")
    
    def save_experiment_results(self, feature_config: str, target_config: str, 
                               model_type: str, metrics: dict):
        """Guardar resultados en JSON"""
        exp_name = f"{feature_config}_{target_config}"
        exp_dir = self.results_dir / exp_name
        
        results = {
            'experiment': {
                'name': exp_name,
                'feature_config': feature_config,
                'target_config': target_config,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            },
            'metrics': metrics,
            'files': {
                'model_files': [
                    f"fnn_best.pth",
                    f"fnn_feature_scaler.pkl", 
                    f"fnn_target_scaler.pkl",
                    f"fnn_metadata.json"
                ],
                'graphics': [
                    "classical_ibvs_results.png",
                    "ml_ibvs_results.png",
                    "ibvs_comparison.png", 
                    "error_comparison_detailed.png"
                ]
            }
        }
        
        results_path = exp_dir / "resultados.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados guardados: {results_path}")
    
    def run_single_experiment(self, feature_config: str, target_config: str, 
                             model_type: str = 'fnn'):
        """Ejecutar un experimento completo"""
        logger.info(f"\nINICIANDO EXPERIMENTO: {feature_config} -> {target_config}")
        
        # 1. Crear estructura
        exp_dir = self.create_experiment_structure(feature_config, target_config)
        logger.info(f"Estructura creada en: {exp_dir}")
        
        # 2. Entrenar modelo
        training_success = self.run_training(feature_config, target_config, model_type)
        if not training_success:
            logger.error(f"Experimento falló en entrenamiento: {feature_config} -> {target_config}")
            return False
        
        # 3. Ejecutar comparación
        metrics = self.run_comparison(feature_config, target_config, model_type)
        
        # 4. Organizar archivos
        self.organize_files(feature_config, target_config, model_type)
        
        # 5. Crear documentación
        self.create_experiment_readme(feature_config, target_config, model_type, metrics)
        
        # 6. Guardar resultados
        self.save_experiment_results(feature_config, target_config, model_type, metrics)
        
        # 7. Almacenar para resumen general
        exp_name = f"{feature_config}_{target_config}"
        self.experiment_results[exp_name] = {
            'feature_config': feature_config,
            'target_config': target_config,
            'model_type': model_type,
            'metrics': metrics,
            'success': 'error' not in metrics
        }
        
        logger.info(f"EXPERIMENTO COMPLETADO: {feature_config} -> {target_config}")
        return True
    
    def run_all_experiments(self):
        """Ejecutar todos los experimentos"""
        logger.info("INICIANDO TODOS LOS EXPERIMENTOS")
        logger.info("=" * 80)
        
        total_experiments = len(self.feature_configs) * len(self.target_configs) * len(self.model_types)
        logger.info(f"Total de experimentos: {total_experiments}")
        
        successful = 0
        failed = 0
        
        for feature_config in self.feature_configs:
            for target_config in self.target_configs:
                for model_type in self.model_types:
                    try:
                        success = self.run_single_experiment(feature_config, target_config, model_type)
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error crítico en experimento {feature_config}->{target_config}: {e}")
                        failed += 1
        
        logger.info("\n" + "=" * 80)
        logger.info("TODOS LOS EXPERIMENTOS COMPLETADOS")
        logger.info("=" * 80)
        logger.info(f"Exitosos: {successful}")
        logger.info(f"Fallidos: {failed}")
        logger.info(f"Resultados en: {self.results_dir}")
        
        # Actualizar README principal
        self.update_main_readme()
        
        return successful, failed
    
    def update_main_readme(self):
        """Actualizar README principal con resultados"""
        # Aquí iría la lógica para actualizar el README principal
        # con los resultados de todos los experimentos
        logger.info("README principal actualizado")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Ejecutor de experimentos organizados IBVS')
    parser.add_argument('--experiment', help='Ejecutar experimento específico (ej: current_only_linear_only)')
    parser.add_argument('--all', action='store_true', help='Ejecutar todos los experimentos')
    parser.add_argument('--feature', choices=['current_only', 'current_desired', 'error_only'],
                       help='Configuración de features específica')
    parser.add_argument('--target', choices=['linear_only', 'all_velocities'],
                       help='Configuración de targets específica')
    
    args = parser.parse_args()
    
    runner = OrganizedExperimentRunner()
    
    if args.all:
        runner.run_all_experiments()
    elif args.feature and args.target:
        runner.run_single_experiment(args.feature, args.target)
    elif args.experiment:
        parts = args.experiment.split('_')
        if len(parts) >= 4:  # feature_config_target_config
            feature_config = '_'.join(parts[:-2])
            target_config = '_'.join(parts[-2:])
            runner.run_single_experiment(feature_config, target_config)
        else:
            logger.error("Formato de experimento inválido. Usa: feature_target (ej: current_only_linear_only)")
    else:
        logger.info("Opciones disponibles:")
        logger.info("  --all                    : Ejecutar todos los experimentos")
        logger.info("  --feature X --target Y   : Experimento específico")
        logger.info("  --experiment X_Y         : Experimento por nombre")

if __name__ == "__main__":
    main() 