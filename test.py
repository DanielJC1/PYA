"""
Script de pruebas para el proyecto de cuantificacion de odio.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, DATA_DIR, RESULTS_DIR
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from evaluator import ModelEvaluator
from experiments import ExperimentRunner

def test_data_loading():
    """Prueba la carga de datos"""
    print("🧪 Probando carga de datos...")
    loader = DataLoader(DATA_DIR)

    df = loader.create_sample_dataset()
    assert len(df) > 0, "Dataset debería tener muestras"

    dist = loader.get_class_distribution(df)
    assert 0 in dist and 1 in dist, "Debería haber muestras de ambas clases"

    print(f"✅ Carga de datos funciona correctamente ({len(df)} muestras)")
    return df

def test_preprocessing():
    """Prueba el preprocesamiento de texto"""
    print("🧪 Probando preprocesamiento...")

    preprocessor = TextPreprocessor()

    # Texto de prueba
    test_text = "Hola @usuario! Esto es un TEXTO con URLs https://example.com y #hashtags"
    clean_text = preprocessor.clean_text(test_text)

    # Verificar que se removieron URLs y menciones
    assert "https://" not in clean_text, "URLs deberían ser removidas"
    assert "@usuario" not in clean_text, "Menciones deberían ser removidas"
    assert "#hashtags" not in clean_text, "Hashtags deberían ser removidos"
    assert clean_text.islower(), "Texto debería estar en minúsculas"

    print("✅ Preprocesamiento funciona correctamente")
    return preprocessor

def test_model_evaluation():
    """Prueba la evaluación de modelos"""
    print("🧪 Probando evaluación de modelos...")

    evaluator = ModelEvaluator(RESULTS_DIR)

    # Datos de prueba
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])

    # Calcular métricas
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)

    # Verificar métricas básicas
    assert 'accuracy' in metrics, "Debería calcular accuracy"
    assert 'precision' in metrics, "Debería calcular precision"
    assert 'recall' in metrics, "Debería calcular recall"
    assert 'f1' in metrics, "Debería calcular f1"
    assert 'roc_auc' in metrics, "Debería calcular roc_auc"

    # Verificar valores razonables
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy debería estar entre 0 y 1"
    assert 0 <= metrics['f1'] <= 1, "F1 debería estar entre 0 y 1"

    print("✅ Evaluación de modelos funciona correctamente")
    return evaluator

def test_experiment_runner():
    """Prueba el corredor de experimentos"""
    print("🧪 Probando corredor de experimentos...")

    runner = ExperimentRunner(RESULTS_DIR)

    # Datos de prueba pequeños
    X_train = ["texto bueno", "otro texto bueno", "texto malo", "muy malo"]
    y_train = [0, 0, 1, 1]
    X_test = ["texto positivo", "contenido negativo"]
    y_test = [0, 1]

    # Configuración simple
    config = {
        'model_type': 'logistic_regression',
        'vectorizer_params': {'max_features': 100},
        'model_params': {'max_iter': 100, 'random_state': RANDOM_SEED}
    }

    # Ejecutar experimento
    result = runner.run_experiment("Test Experiment", X_train, y_train, X_test, y_test, config)

    # Verificar resultado
    assert 'name' in result, "Resultado debería tener nombre"
    assert 'metrics' in result, "Resultado debería tener métricas"
    assert 'predictions' in result, "Resultado debería tener predicciones"
    assert len(result['predictions']) == len(y_test), "Debería haber una predicción por muestra de test"

    print("✅ Corredor de experimentos funciona correctamente")
    return runner

def test_full_pipeline():
    """Prueba el pipeline completo"""
    print("🧪 Probando pipeline completo...")

    loader = DataLoader(DATA_DIR)
    df = loader.create_sample_dataset()

    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)

    train, val, test = loader.split_data(df, test_size=0.3, val_size=0.2)

    runner = ExperimentRunner(RESULTS_DIR)
    config = {
        'model_type': 'naive_bayes',
        'vectorizer_params': {'max_features': 500},
        'model_params': {}
    }

    result = runner.run_experiment(
        "Pipeline Test",
        train['text'].values, train['label'].values,
        test['text'].values, test['label'].values,
        config
    )

    assert result['metrics']['accuracy'] > 0.5, "Accuracy debería ser mayor a 0.5"

    print("✅ Pipeline completo funciona correctamente")

def test_error_analysis():
    """Prueba el análisis de errores"""
    print("🧪 Probando análisis de errores...")

    evaluator = ModelEvaluator(RESULTS_DIR)

    # Datos con errores conocidos
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])  # Un FP y un FN
    texts = np.array(["texto bueno", "texto bueno", "texto malo", "texto malo", "texto bueno", "texto malo"])

    # Analizar errores
    errors = evaluator.analyze_errors(y_true, y_pred, texts)

    # Verificar tipos de error
    assert len(errors['false_positives']) == 1, "Debería haber 1 falso positivo"
    assert len(errors['false_negatives']) == 1, "Debería haber 1 falso negativo"
    assert len(errors['true_positives']) == 2, "Debería haber 2 verdaderos positivos"
    assert len(errors['true_negatives']) == 2, "Debería haber 2 verdaderos negativos"

    print("✅ Análisis de errores funciona correctamente")

def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("="*60)
    print("🧪 EJECUTANDO PRUEBAS DEL PROYECTO")
    print("="*60)

    try:
        # Ejecutar pruebas
        test_data_loading()
        test_preprocessing()
        test_model_evaluation()
        test_experiment_runner()
        test_full_pipeline()
        test_error_analysis()

        print("\n" + "="*60)
        print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
