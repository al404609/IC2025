# Sistema de Experimentación y Comparación IBVS

Este directorio contiene el sistema completo de experimentación y comparación usado para obtener el resultado de **97.5% de mejora** del modelo neuronal sobre el IBVS clásico.

## Archivos Principales

### 1. `run_organized_experiments.py`
- **Función**: Ejecutor principal de experimentos sistemáticos
- **Capacidades**: Entrena modelos, ejecuta comparaciones y organiza resultados
- **Configuraciones**: 6 combinaciones de features y targets
- **Automatización**: Genera documentación y gráficos automáticamente

### 2. `classical_vs_ml_comparison.py`
- **Función**: Comparador directo entre IBVS clásico y ML
- **Análisis**: Genera 4 gráficos de análisis por cada método
- **Métricas**: Extrae métricas de rendimiento y precisión
- **Visualización**: Crea gráficos comparativos lado a lado

## Configuraciones Experimentales

### Configuraciones de Features (Input)
- **`current_only`**: Solo coordenadas actuales (8 valores)
- **`current_desired`**: Coordenadas actuales + deseadas (16 valores) [MEJOR]
- **`error_only`**: Solo errores entre actual y deseado (8 valores)

### Configuraciones de Targets (Output)
- **`linear_only`**: Solo velocidades lineales [vx, vy, vz] (3 valores) [MEJOR]
- **`all_velocities`**: Velocidades lineales + angulares (6 valores)

### Combinaciones Experimentales
El sistema prueba todas las combinaciones posibles:
1. `current_only` → `linear_only`
2. `current_desired` → `linear_only` **[MEJOR: +97.5%]**
3. `error_only` → `linear_only`
4. `current_only` → `all_velocities`
5. `current_desired` → `all_velocities`
6. `error_only` → `all_velocities`

## Uso del Sistema

### Ejecutar Todos los Experimentos
```bash
cd ivan/entrega/experimentos
python run_organized_experiments.py --all
```

### Ejecutar Experimento Específico
```bash
cd ivan/entrega/experimentos
python run_organized_experiments.py --feature current_desired --target linear_only
```

### Ejecutar Solo Comparación
```bash
cd ivan/entrega/experimentos
python classical_vs_ml_comparison.py --model-path ../results/modelo --model-type fnn
```

## Proceso de Experimentación

### 1. Entrenamiento
- Carga dataset de entrenamiento
- Entrena modelo con configuración específica
- Guarda modelo, scalers y metadatos

### 2. Comparación
- Ejecuta IBVS clásico en condiciones controladas
- Ejecuta IBVS ML con modelo entrenado
- Compara rendimiento y precisión

### 3. Análisis
- Genera 4 gráficos por método:
  - Evolución del error
  - Trayectoria en plano imagen
  - Componentes de velocidad
  - Trayectoria de cámara 3D

### 4. Documentación
- Crea README específico por experimento
- Genera archivo JSON con métricas
- Organiza archivos en estructura jerárquica

## Estructura de Resultados

```
experiments_results/
├── current_desired_linear_only/    # MEJOR MODELO
│   ├── README.md
│   ├── resultados.json
│   ├── modelo/
│   │   ├── fnn_best.pth
│   │   ├── fnn_feature_scaler.pkl
│   │   ├── fnn_target_scaler.pkl
│   │   └── fnn_metadata.json
│   └── graficos/
│       ├── classical_ibvs_results.png
│       ├── ml_ibvs_results.png
│       ├── ibvs_comparison.png
│       └── error_comparison_detailed.png
├── current_only_linear_only/
├── error_only_linear_only/
├── current_desired_all_velocities/
├── current_only_all_velocities/
└── error_only_all_velocities/
```

## Métricas de Evaluación

### Métricas de Precisión
- **Error final**: Norma del error al final de la simulación
- **Pasos para convergencia**: Número de iteraciones necesarias
- **Mejora porcentual**: Comparación relativa entre métodos

### Métricas de Arquitectura
- **Parámetros totales**: Número de parámetros entrenables
- **Tiempo de entrenamiento**: Duración del proceso de entrenamiento
- **Pérdida de validación**: Métrica de calidad del modelo

## Mejor Modelo Encontrado

### Configuración Ganadora
- **Features**: `current_desired` (16 valores)
- **Targets**: `linear_only` (3 valores)
- **Arquitectura**: FNN [16] → [64] → [32] → [3]
- **Mejora**: 97.5% sobre IBVS clásico

### Especificaciones Técnicas
- **Regularización**: Dropout 0.3 + BatchNorm
- **Activación**: ReLU
- **Inicialización**: Xavier Uniform
- **Parámetros**: 3,267 total
- **Entrenamiento**: 6 épocas, 148.56 segundos

## Dependencias

### Librerías Principales
- **RVC3**: Robotics Vision and Control Library
- **PyTorch**: Framework de deep learning
- **NumPy**: Computación numérica
- **Matplotlib**: Visualización
- **SpatialMath**: Matemáticas espaciales

### Módulos Internos
- **core.simulation**: Motor de simulación IBVS
- **core.ml_controller**: Controlador IBVS basado en ML
- **models.architectures**: Arquitecturas de redes neuronales
- **config.settings**: Configuraciones del sistema

## Reproducibilidad

### Semilla Aleatoria
- **Valor fijo**: 42 para reproducibilidad
- **Aplicada a**: NumPy, PyTorch, generación de datos

### Condiciones Controladas
- **Posición de cámara**: Fija para comparaciones
- **Puntos target**: Generados consistentemente
- **Parámetros IBVS**: Valores estándar del RVC3

## Limitaciones y Consideraciones

### Limitaciones Actuales
- Solo arquitectura FNN implementada completamente
- Dataset limitado a configuraciones específicas
- Comparación en condiciones simuladas únicamente

### Posibles Mejoras
- Implementar LSTM y ResNet completos
- Ampliar dataset con más variabilidad
- Probar en condiciones de hardware real
- Optimizar hiperparámetros automáticamente

## Resultados Históricos

### Experimentos Ejecutados
- **Fecha**: 2024-2025
- **Total configuraciones**: 6
- **Mejor resultado**: 97.5% mejora
- **Tiempo total**: ~2 horas de experimentación

### Hallazgos Principales
1. **Información completa es clave**: `current_desired` supera a `current_only`
2. **Simplicidad funciona**: `linear_only` supera a `all_velocities`
3. **FNN es suficiente**: Arquitectura simple pero efectiva
4. **Regularización importante**: Dropout y BatchNorm mejoran generalización

## Contacto y Mantenimiento

Este sistema fue desarrollado como parte del proyecto de investigación en IBVS neuronal. Para preguntas o mejoras, consultar la documentación principal del proyecto. 