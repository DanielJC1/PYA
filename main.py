"""
Script principal para cuantificacion de riesgo de odio en texto.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, DATA_DIR, RESULTS_DIR
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from evaluator import ModelEvaluator
from experiments import ExperimentRunner

def main():
    """Función principal"""
    print("="*70)
    print("Cuantificacion del Odio en Redes Sociales")
    print("="*70)
    
    np.random.seed(RANDOM_SEED)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    loader = DataLoader(DATA_DIR)
    
    # Crear dataset de ejemplo
    df = loader.create_sample_dataset(n_samples=1000)
    print(f"Dataset creado con {len(df)} muestras")
    print(f"Distribución de clases:\n{loader.get_class_distribution(df)}")
    
    # 2. Preprocesar
    print("\n[2] Preprocesando texto...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, 'text')
    print("Preprocesamiento completado")
    
    # Estadísticas de texto
    stats = preprocessor.get_text_statistics(df['text'].tolist())
    print(f"Estadísticas de texto:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 3. Dividir datos
    print("\n[3] Dividiendo datos...")
    train, val, test = loader.split_data(df)
    
    X_train = train['text'].values
    y_train = train['label'].values
    X_test = test['text'].values
    y_test = test['label'].values
    
    # 4. Ejecutar experimentos
    print("\n[4] Ejecutando experimentos...")
    runner = ExperimentRunner(output_dir=RESULTS_DIR)
    
    # Experimento 1: Logistic Regression
    config1 = {
        'model_type': 'logistic_regression',
        'vectorizer_params': {'max_features': 5000, 'ngram_range': (1, 2)},
        'model_params': {'max_iter': 1000, 'random_state': RANDOM_SEED}
    }
    result1 = runner.run_experiment("Logistic Regression", X_train, y_train, X_test, y_test, config1)
    
    # Experimento 2: Random Forest
    config2 = {
        'model_type': 'random_forest',
        'vectorizer_params': {'max_features': 3000, 'ngram_range': (1, 2)},
        'model_params': {'n_estimators': 100, 'random_state': RANDOM_SEED, 'n_jobs': -1}
    }
    result2 = runner.run_experiment("Random Forest", X_train, y_train, X_test, y_test, config2)
    
    # Experimento 3: Naive Bayes
    config3 = {
        'model_type': 'naive_bayes',
        'vectorizer_params': {'max_features': 3000, 'ngram_range': (1, 2)},
        'model_params': {}
    }
    result3 = runner.run_experiment("Naive Bayes", X_train, y_train, X_test, y_test, config3)
    
    # 5. Evaluar modelos
    print("\n[5] Evaluando modelos...")
    evaluator = ModelEvaluator(output_dir=RESULTS_DIR)
    
    # Usar predicciones del mejor modelo (Logistic Regression)
    y_pred = np.array(result1['predictions'])
    y_proba = np.array(result1['probabilities'])
    
    # Graficar
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_proba)
    
    # Reporte detallado
    print("\nReporte de clasificacion orientativo:")
    print(evaluator.get_detailed_report(y_test, y_pred))
    
    # Análisis de errores
    print("\n[6] Analizando errores...")
    error_report = evaluator.generate_error_report(y_test, y_pred, X_test)
    
    error_summary = error_report['error_type'].value_counts()
    print("\nResumen de errores de estimacion:")
    print(error_summary)
    
    # Guardar reporte de errores
    error_report.to_csv(RESULTS_DIR / "error_report.csv", index=False, encoding='utf-8-sig')
    print(f"\nReporte de errores guardado: {RESULTS_DIR / 'error_report.csv'}")
    
    # 6. Comparar experimentos
    print("\n[7] Comparando experimentos...")
    comparison = runner.compare_experiments()
    print("\n" + str(comparison))
    
    # Guardar registro
    runner.save_experiment_log()
    
    print("\n" + "="*70)
    print("Proceso completado. Revisa los resultados en:", RESULTS_DIR)
    print("="*70)

if __name__ == "__main__":
    main()
