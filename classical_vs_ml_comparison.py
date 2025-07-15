#!/usr/bin/env python3
"""
Comparación Directa: Clásico vs ML IBVS
Basado en machine_vs_classical_comparation.py del proyecto de referencia
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RVC3 libraries
from machinevisiontoolbox import *
from spatialmath import *

# Import our modules
from core.ml_controller import MLIBVSController
from comparison_study import IBVSTestSetup

def ensure_results_folder():
    """Crear carpeta .resources si no existe"""
    if not os.path.exists('.resources'):
        os.makedirs('.resources')
        print("Carpeta .resources creada para resultados")

def test_classical_ibvs():
    """Probar IBVS clásico (para comparación)"""
    print("=" * 60)
    print("Probando IBVS Clásico (para comparación)")
    print("=" * 60)
    
    # Replicar condiciones exactas del entrenamiento
    # Generar posición de cámara como en entrenamiento (posición aleatoria detrás del target)
    camera = CentralCamera.Default(pose=SE3.Trans(1.5, 0.8, -1.5))
    
    # Generar puntos del mundo exactamente como en entrenamiento
    P = mkgrid(2, side=0.5, pose=SE3.Trans(0.2, -0.3, 0.8))
    
    # Usar mismo patrón de puntos deseados que en entrenamiento
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
    
    print(f"Pose de cámara: {camera.pose}")
    print(f"Puntos del mundo P: {P.shape}")
    print(f"Puntos deseados pd: {pd.shape}")
    
    # Crear controlador IBVS clásico
    ibvs = IBVS(camera, P=P, p_d=pd, graphics=False)
    
    # Ejecutar simulación
    print("Ejecutando IBVS clásico...")
    ibvs.run(50)
    
    print(f"IBVS clásico completado en {len(ibvs.history)} pasos")
    print(f"Error final: {ibvs.history[-1].enorm:.6f}")
    
    # Limpiar gráficos existentes
    plt.close('all')
    
    # Crear gráficos limpios y personalizados
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Evolución del error
    ax1 = axes[0, 0]
    errors = [h.enorm for h in ibvs.history]
    ax1.plot(errors, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Paso de tiempo')
    ax1.set_ylabel('Norma del error')
    ax1.set_title('IBVS Clásico - Evolución del Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Gráfico 2: Trayectoria en el plano imagen
    ax2 = axes[0, 1]
    # Plotear puntos deseados
    ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=12, label='Deseados', markeredgecolor='darkred')
    
    # Plotear trayectoria para cada punto característico
    colors = ['blue', 'green', 'orange', 'purple']
    for i in range(pd.shape[1]):
        p_traj = np.array([h.p[0, i] for h in ibvs.history])
        q_traj = np.array([h.p[1, i] for h in ibvs.history])
        ax2.plot(p_traj, q_traj, color=colors[i], linewidth=2, alpha=0.7, label=f'Punto {i+1}')
        # Marcadores de inicio y fin
        ax2.plot(p_traj[0], q_traj[0], 'o', color=colors[i], markersize=8, markeredgecolor='black')
        ax2.plot(p_traj[-1], q_traj[-1], 's', color=colors[i], markersize=8, markeredgecolor='black')
    
    ax2.set_xlabel('u (píxeles)')
    ax2.set_ylabel('v (píxeles)')
    ax2.set_title('IBVS Clásico - Trayectoria en Plano Imagen')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Gráfico 3: Componentes de velocidad
    ax3 = axes[1, 0]
    velocities = np.array([h.vel for h in ibvs.history])
    time_steps = range(len(ibvs.history))
    
    # Plotear velocidades lineales
    ax3.plot(time_steps, velocities[:, 0], 'r-', linewidth=2, label='vx')
    ax3.plot(time_steps, velocities[:, 1], 'g-', linewidth=2, label='vy')
    ax3.plot(time_steps, velocities[:, 2], 'b-', linewidth=2, label='vz')
    ax3.set_xlabel('Paso de tiempo')
    ax3.set_ylabel('Velocidad lineal (m/s)')
    ax3.set_title('IBVS Clásico - Velocidades Lineales')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Trayectoria de cámara 3D (vista superior)
    ax4 = axes[1, 1]
    poses = [h.pose for h in ibvs.history]
    x_traj = [pose.t[0] for pose in poses]
    y_traj = [pose.t[1] for pose in poses]
    
    ax4.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.7, label='Trayectoria cámara')
    ax4.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Inicio')
    ax4.plot(x_traj[-1], y_traj[-1], 'ro', markersize=10, label='Fin')
    
    # Plotear proyección de puntos del mundo
    ax4.plot(P[0, :], P[1, :], 'k*', markersize=12, label='Puntos target')
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('IBVS Clásico - Trayectoria Cámara (Vista Superior)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig('.resources/classical_ibvs_results.png', dpi=150, bbox_inches='tight')
    print("Gráficos IBVS clásico guardados en .resources/classical_ibvs_results.png")
    
    return ibvs

def test_ml_ibvs(model_path="models/trained", model_type="fnn"):
    """Probar IBVS basado en ML"""
    print("=" * 60)
    print("Probando IBVS basado en ML")
    print("=" * 60)
    
    # Verificar si existen archivos del modelo
    required_files = [
        f"{model_path}/{model_type}_best.pth",
        f"{model_path}/{model_type}_feature_scaler.pkl",
        f"{model_path}/{model_type}_target_scaler.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Faltan archivos del modelo:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPor favor asegúrate de haber entrenado el modelo y guardado los archivos necesarios.")
        return None
    
    # Usar MISMAS condiciones que IBVS clásico
    camera = CentralCamera.Default(pose=SE3.Trans(1.5, 0.8, -1.5))
    P = mkgrid(2, side=0.5, pose=SE3.Trans(0.2, -0.3, 0.8))
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
    
    print(f"Pose de cámara: {camera.pose}")
    print(f"Puntos del mundo P: {P.shape}")
    print(f"Puntos deseados pd: {pd.shape}")
    
    try:
        # Crear controlador IBVS basado en ML
        print(f"Cargando modelo ML desde: {model_path}")
        ml_ibvs = MLIBVSController(
            camera=camera,
            P=P,
            p_d=pd,
            model_path=model_path,
            model_type=model_type,
            graphics=False
        )
        print("Modelo ML cargado exitosamente!")
        
        # Ejecutar simulación
        print("Ejecutando IBVS basado en ML...")
        
        # Ejecutar paso a paso para mejor control
        for step in range(100):  
            status = ml_ibvs.step(step)
            
            if status == 1:  # Convergido ok
                print(f"Convergió en paso {step}")
                break
            elif status == -1:  # Error :(
                print(f"Error en paso {step}")
                break
        
        print(f"IBVS ML completado en {len(ml_ibvs.history)} pasos")
        print(f"Error final: {ml_ibvs.history[-1].enorm:.6f}")
        
        # Limpiar gráficos existentes
        plt.close('all')
        
        # Crear gráficos limpios y personalizados
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: Evolución del error
        ax1 = axes[0, 0]
        errors = [h.enorm for h in ml_ibvs.history]
        ax1.plot(errors, 'r-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Paso de tiempo')
        ax1.set_ylabel('Norma del error')
        ax1.set_title('IBVS ML - Evolución del Error')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Gráfico 2: Trayectoria en el plano imagen
        ax2 = axes[0, 1]
        # Plotear puntos deseados
        ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=12, label='Deseados', markeredgecolor='darkred')
        
        # Plotear trayectoria para cada punto característico
        colors = ['blue', 'green', 'orange', 'purple']
        for i in range(pd.shape[1]):
            p_traj = np.array([h.p[0, i] for h in ml_ibvs.history])
            q_traj = np.array([h.p[1, i] for h in ml_ibvs.history])
            ax2.plot(p_traj, q_traj, color=colors[i], linewidth=2, alpha=0.7, label=f'Punto {i+1}')
            # Marcadores de inicio y fin
            ax2.plot(p_traj[0], q_traj[0], 'o', color=colors[i], markersize=8, markeredgecolor='black')
            ax2.plot(p_traj[-1], q_traj[-1], 's', color=colors[i], markersize=8, markeredgecolor='black')
        
        ax2.set_xlabel('u (píxeles)')
        ax2.set_ylabel('v (píxeles)')
        ax2.set_title('IBVS ML - Trayectoria en Plano Imagen')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Gráfico 3: Componentes de velocidad
        ax3 = axes[1, 0]
        velocities = np.array([h.vel for h in ml_ibvs.history])
        time_steps = range(len(ml_ibvs.history))
        
        # Plotear velocidades lineales
        ax3.plot(time_steps, velocities[:, 0], 'r-', linewidth=2, label='vx')
        ax3.plot(time_steps, velocities[:, 1], 'g-', linewidth=2, label='vy')
        ax3.plot(time_steps, velocities[:, 2], 'b-', linewidth=2, label='vz')
        ax3.set_xlabel('Paso de tiempo')
        ax3.set_ylabel('Velocidad lineal (m/s)')
        ax3.set_title('IBVS ML - Velocidades Lineales')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Trayectoria de cámara 3D (vista superior)
        ax4 = axes[1, 1]
        poses = [h.pose for h in ml_ibvs.history]
        x_traj = [pose.t[0] for pose in poses]
        y_traj = [pose.t[1] for pose in poses]
        
        ax4.plot(x_traj, y_traj, 'r-', linewidth=2, alpha=0.7, label='Trayectoria cámara')
        ax4.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Inicio')
        ax4.plot(x_traj[-1], y_traj[-1], 'ro', markersize=10, label='Fin')
        
        # Plotear proyección de puntos del mundo
        ax4.plot(P[0, :], P[1, :], 'k*', markersize=12, label='Puntos target')
        
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('IBVS ML - Trayectoria Cámara (Vista Superior)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.savefig('.resources/ml_ibvs_results.png', dpi=150, bbox_inches='tight')
        print("Gráficos IBVS ML guardados en .resources/ml_ibvs_results.png")
        
        return ml_ibvs
        
    except Exception as e:
        print(f"ERROR: Fallo al ejecutar IBVS basado en ML: {e}")
        return None

def compare_results(classical_ibvs, ml_ibvs):
    """Comparar resultados de IBVS clásico vs basado en ML"""
    if ml_ibvs is None:
        print("No se pueden comparar resultados - IBVS ML falló")
        return
    
    print("=" * 60)
    print("Comparación IBVS Clásico vs ML")
    print("=" * 60)
    
    # Extraer datos de ambos controladores
    classical_steps = len(classical_ibvs.history)
    ml_steps = len(ml_ibvs.history)
    
    classical_final_error = classical_ibvs.history[-1].enorm
    ml_final_error = ml_ibvs.history[-1].enorm
    
    print(f"Pasos para completar:")
    print(f"  IBVS Clásico: {classical_steps}")
    print(f"  IBVS ML:      {ml_steps}")
    
    print(f"\nError final:")
    print(f"  IBVS Clásico: {classical_final_error:.6f}")
    print(f"  IBVS ML:      {ml_final_error:.6f}")
    
    # Determinar mejora de rendimiento
    if ml_final_error < classical_final_error:
        improvement = (classical_final_error - ml_final_error) / classical_final_error * 100
        print(f"  IBVS ML es {improvement:.1f}% mejor!")
    else:
        degradation = (ml_final_error - classical_final_error) / classical_final_error * 100
        print(f"  IBVS Clásico es {degradation:.1f}% mejor")
    
    # Limpiar gráficos existentes
    plt.close('all')
    
    # Crear gráfico limpio y enfocado de comparación
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Comparación de evolución de error
    ax1 = axes[0]
    classical_errors = [h.enorm for h in classical_ibvs.history]
    ml_errors = [h.enorm for h in ml_ibvs.history]
    
    ax1.plot(classical_errors, 'b-', label='IBVS Clásico', linewidth=3, alpha=0.8)
    ax1.plot(ml_errors, 'r--', label='IBVS ML', linewidth=3, alpha=0.8)
    ax1.set_xlabel('Paso de tiempo', fontsize=12)
    ax1.set_ylabel('Norma del error', fontsize=12)
    ax1.set_title('Comparación de Evolución del Error', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Agregar anotaciones de error final
    ax1.annotate(f'Final: {classical_final_error:.3f}', 
                xy=(classical_steps-1, classical_final_error), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                fontsize=10)
    ax1.annotate(f'Final: {ml_final_error:.3f}', 
                xy=(ml_steps-1, ml_final_error), 
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                fontsize=10)
    
    # Gráfico 2: Comparación de trayectorias en plano imagen
    ax2 = axes[1]
    
    # Plotear puntos deseados (iguales para ambos)
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[classical_ibvs.camera.pp]
    ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=15, label='Deseados', 
             markeredgecolor='darkred', markeredgewidth=2)
    
    # Plotear posiciones finales para ambos métodos
    classical_final_p = classical_ibvs.history[-1].p
    ml_final_p = ml_ibvs.history[-1].p
    
    ax2.plot(classical_final_p[0, :], classical_final_p[1, :], 'bo', 
             markersize=10, label='Final clásico', markeredgecolor='darkblue')
    ax2.plot(ml_final_p[0, :], ml_final_p[1, :], 'rs', 
             markersize=10, label='Final ML', markeredgecolor='darkred')
    
    # Conectar deseados a posiciones finales con líneas
    for i in range(pd.shape[1]):
        ax2.plot([pd[0, i], classical_final_p[0, i]], 
                [pd[1, i], classical_final_p[1, i]], 
                'b--', alpha=0.5, linewidth=1)
        ax2.plot([pd[0, i], ml_final_p[0, i]], 
                [pd[1, i], ml_final_p[1, i]], 
                'r--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('u (píxeles)', fontsize=12)
    ax2.set_ylabel('v (píxeles)', fontsize=12)
    ax2.set_title('Comparación de Puntos Finales', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('.resources/ibvs_comparison.png', dpi=150, bbox_inches='tight')
    print("Gráficos de comparación guardados en .resources/ibvs_comparison.png")
    
    # Crear gráfico detallado de error para publicación
    plt.figure(figsize=(10, 6))
    plt.plot(classical_errors, 'b-', label='IBVS Clásico', linewidth=3, alpha=0.8)
    plt.plot(ml_errors, 'r--', label='IBVS ML', linewidth=3, alpha=0.8)
    plt.xlabel('Paso de tiempo', fontsize=14)
    plt.ylabel('Norma del error', fontsize=14)
    plt.title('Comparación de Rendimiento IBVS', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Agregar caja de texto de resumen de rendimiento
    if ml_final_error < classical_final_error:
        improvement = (classical_final_error - ml_final_error) / classical_final_error * 100
        summary_text = f'IBVS ML logra {improvement:.1f}% mejor precisión\n' + \
                      f'Clásico: {classical_final_error:.3f}\n' + \
                      f'ML: {ml_final_error:.3f}'
        box_color = 'lightgreen'
    else:
        degradation = (ml_final_error - classical_final_error) / classical_final_error * 100
        summary_text = f'IBVS Clásico es {degradation:.1f}% mejor\n' + \
                      f'Clásico: {classical_final_error:.3f}\n' + \
                      f'ML: {ml_final_error:.3f}'
        box_color = 'lightyellow'
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('.resources/error_comparison_detailed.png', dpi=150, bbox_inches='tight')
    print("Comparación detallada de error guardada en .resources/error_comparison_detailed.png")

def main():
    """Función principal de comparación"""
    print("Comparación Machine Learning vs IBVS Clásico")
    print("Basado en ejemplos del Capítulo 15 de RVC3")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Comparación IBVS Clásico vs ML')
    parser.add_argument('--model-path', default='models/trained', help='Ruta al modelo entrenado')
    parser.add_argument('--model-type', choices=['fnn', 'lstm', 'resnet'], default='fnn', 
                       help='Tipo de modelo a usar')
    
    args = parser.parse_args()
    
    # Asegurar que existe carpeta de salida
    ensure_results_folder()
    
    # Establecer semilla aleatoria para reproducibilidad
    np.random.seed(42)
    
    # Probar IBVS clásico primero
    classical_ibvs = test_classical_ibvs()
    
    print("\n" + "=" * 60)
    print("Continuando con prueba IBVS basado en ML...")
    
    # Probar IBVS basado en ML
    ml_ibvs = test_ml_ibvs(args.model_path, args.model_type)
    
    # Comparar resultados si ambos funcionaron
    if classical_ibvs and ml_ibvs:
        print("\n" + "=" * 60)
        print("Generando gráficos de comparación...")
        compare_results(classical_ibvs, ml_ibvs)
    
    print("\n" + "=" * 60)
    print("Comparación ok!")
    print("Todos los resultados guardados en carpeta .resources/:")
    print("  - classical_ibvs_results.png (análisis detallado de 4 paneles)")
    if ml_ibvs:
        print("  - ml_ibvs_results.png (análisis detallado de 4 paneles)")
        print("  - ibvs_comparison.png (comparación lado a lado)")
        print("  - error_comparison_detailed.png (gráfico de error para publicación)")
    print("\nLos gráficos están ahora mucho más limpios y fáciles de leer!")
    print("Cada gráfico individual se enfoca en aspectos específicos:")
    print("  • Evolución del error con escala logarítmica")
    print("  • Trayectorias en plano imagen con marcadores claros")
    print("  • Componentes de velocidad lineal en el tiempo")
    print("  • Trayectoria de cámara en vista superior")
    print("  • Gráficos de comparación con métricas de rendimiento")

if __name__ == "__main__":
    main() 
