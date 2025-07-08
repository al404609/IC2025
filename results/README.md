# Experimento: `current_desired` → `linear_only`

##  Descripción del Experimento

**Configuración**: Coordenadas Completas → Solo Velocidades Lineales  
  **MEJOR** (+97.5%)

### Configuración de Features (Input)
- **Tipo**: `current_desired` 
- **Dimensión**: 16 valores
- **Contenido**: Coordenadas actuales + deseadas
- **Ventaja**: Información completa del estado del sistema
- **Similitud**: Máxima información disponible

### Configuración de Targets (Output)  
- **Tipo**: `linear_only`
- **Dimensión**: 3 valores
- **Contenido**: Velocidades lineales [vx, vy, vz]
- **Enfoque**: Control translacional únicamente

## esultados del Entrenamiento

### Arquitectura del Modelo
```python
FeedForward Network (FNN):
├── Input Layer: 16 features
├── Hidden Layer 1: 64 neurons
│   ├── Linear(16 → 64)
│   ├── BatchNorm1d(64)
│   ├── ReLU()
│   └── Dropout(0.3)
├── Hidden Layer 2: 32 neurons
│   ├── Linear(64 → 32)
│   ├── BatchNorm1d(32)
│   ├── ReLU()
│   └── Dropout(0.3)
└── Output Layer: 3 velocities
    └── Linear(32 → 3)

Total Parameters: 3,267
Weight Initialization: Xavier Uniform
Activation Function: ReLU
Regularization: BatchNorm + Dropout(0.3)
```

### Métricas de Entrenamiento
- **Modelo**: FNN
- **Épocas**: 10 (entrenamiento rápido)
- **Batch Size**: 512
- **Learning Rate**: 0.0005
- **Optimizador**: Adam + ReduceLROnPlateau
- **Criterio**: MSE Loss


### **RESULTADO**
ML es 97.5% mejor que el método clásico

## Archivos Generados

### Modelo
- `modelo/fnn_best.pth` - Estado del modelo entrenado
- `modelo/fnn_feature_scaler.pkl` - Escalador de features  
- `modelo/fnn_target_scaler.pkl` - Escalador de targets
- `modelo/fnn_metadata.json` - Metadatos completos

### Gráficos
- `graficos/classical_ibvs_results.png` - Análisis IBVS clásico
- `graficos/ml_ibvs_results.png` - Análisis IBVS ML  
- `graficos/ibvs_comparison.png` - Comparación lado a lado
- `graficos/error_comparison_detailed.png` - Evolución de error

### Datos
- `resultados.json` - Métricas numéricas completas
- `logs/` - Logs de entrenamiento y comparación

---

*Experimento ejecutado: 2025-07-06 08:54*
