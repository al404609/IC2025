# Sistema IBVS Mejorado con Redes Neuronales

Sistema de control visual basado en imágenes (IBVS) mejorado con redes neuronales que logra **97.5% de mejora** sobre librería usada en clase.

## Instalación Rápida

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import torch; import numpy; print('Todo instalado')"
```

## Uso Básico

### 1. Generar Datos de Entrenamiento
```bash
# Generar dataset completo
python data/dataset_generator.py

# Verificar datos generados
ls data/generated/
```

### 2. Entrenar el Mejor Modelo
```bash
# Entrenar modelo óptimo (current_desired -> linear_only)
python utils/trainer.py --config best_model

# Entrenar modelo personalizado
python utils/trainer.py --features current_desired --target linear_only --epochs 100
```

### 3. Ejecutar Simulaciones
```bash
# Simulación con modelo neuronal
python core/simulation.py --model results/best_model.pth

# Simulación clásica (comparación)
python core/simulation.py --classical

# Simulación con visualización
python core/simulation.py --model results/best_model.pth --plot
```

### 4. Comparar Resultados
```bash
# Comparar todos los modelos
python utils/compare_models.py

# Comparar específicos
python utils/compare_models.py --models current_desired linear_only classical
```

## Configuraciones Disponibles

| Configuración | Características | Salida | Mejora |
|---------------|----------------|---------|--------|
| `current_desired` | 16 (actual + deseado) | 3 (linear) | **+97.5%** |
| `current_only` | 8 (solo actual) | 3 (linear) | 0.0% |
| `error_only` | 8 (error) | 3 (linear) | 0.0% |
| `*_all_velocities` | * | 6 (lin+ang) | -126.7% |

## Estructura del Proyecto

```
entrega/
├── config/settings.py      # Configuración principal
├── core/
│   ├── data_handler.py     # Manejo de datos
│   └── simulation.py       # Motor de simulación
├── models/architectures.py # Arquitecturas NN
├── utils/trainer.py        # Sistema de entrenamiento
├── data/dataset_generator.py # Generador de datos
├── results/                # Modelos entrenados
└── requirements.txt        # Dependencias
```

## Comandos Útiles

```bash
# Ver configuración actual
python -c "from config.settings import *; print(f'Features: {INPUT_FEATURES}, Target: {TARGET_TYPE}')"

# Entrenar rápido (pocas épocas)
python utils/trainer.py --epochs 10 --quick

# Evaluar modelo existente
python utils/trainer.py --eval-only --model results/best_model.pth

# Generar gráficos de resultados
python utils/plot_results.py --model results/best_model.pth
```

## Resultados 

- **Mejor configuración**: `current_desired` → `linear_only`
- **Arquitectura**: FNN [16] → [64] → [32] → [3]
- **Mejora**: 97.5% vs IBVS clásico
- **Tiempo de convergencia**: ~50 épocas
- **Precisión**: 0.01mm en posición final

## Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Error: "RVC3 not found"
```bash
pip install robotics-toolbox-python
```

### Error: "No data found"
```bash
python data/dataset_generator.py  # Generar datos primero
```

