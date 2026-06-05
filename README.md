# Cuantificación del Odio en Redes Sociales

Proyecto para estimar y analizar la probabilidad de contenido de odio en redes sociales. El sistema usa una arquitectura híbrida: combina un clasificador TF-IDF con reglas de contexto para cuantificar riesgo sin tratar cualquier insulto como una confirmación automática.

## Alcance

- Estima riesgo a partir de texto y reglas de contexto.
- No determina de forma definitiva si un mensaje constituye odio.
- Debe usarse como apoyo para análisis, priorización y revisión humana.

## Dataset

El sistema entrena sobre un corpus combinado de dos fuentes:

- **HateCheck en español** (Paul/hatecheck-spanish): 3,745 casos de prueba anotados para detección de odio en español, con cobertura de 7 grupos objetivo: personas gay, trans, indígenas, discapacitadas, mujeres, negras y judías.
- **Dataset sintético balanceado**: 1,000 ejemplos en español (500 odio / 500 no odio) con frases cotidianas de no odio para balancear la distribución del corpus.

El corpus combinado tiene 4,745 muestras con distribución 66% odio / 34% no odio.

## Resultados

Métricas sobre el conjunto de prueba (949 muestras):

| Modelo              | Accuracy | F1     | ROC-AUC |
|---------------------|----------|--------|---------|
| Logistic Regression | 94.2%    | 94.1%  | 97.7%   |
| Random Forest       | **96.6%**| **96.6%**| **99.2%**|
| Naive Bayes         | 94.1%    | 93.9%  | 96.4%   |

Análisis de errores (Logistic Regression):

| Tipo                 | Cantidad |
|----------------------|----------|
| Verdaderos positivos | 626      |
| Verdaderos negativos | 268      |
| Falsos positivos     | 55       |
| Falsos negativos     | 0        |

Los 55 falsos positivos se agrupan en cinco categorías: afirmaciones positivas sobre grupos objetivo, violencia dirigida a objetos o conceptos, lenguaje hiperbólico fuera de contexto, autocrítica en tiempo pasado, y casos de prueba con inversión semántica del odio.

## Estructura del Proyecto

```
PYA/
├── data/
│   ├── test_paul_hatecheck_spanish.csv  # Corpus HateCheck español
│   └── sample_data.csv                  # Dataset sintético balanceado
├── docs/                                # Versión estática para GitHub Pages
├── extras/                              # Scripts opcionales
│   ├── fetch_twitter_data.py            # Descarga tweets desde X/Twitter
│   └── twitter_client.py               # Cliente API de X
├── src/                                 # Código fuente
│   ├── config.py                        # Configuración general
│   ├── data_loader.py                   # Carga y preprocesamiento de datos
│   ├── evaluator.py                     # Evaluación y análisis de errores
│   ├── experiments.py                   # Gestión de experimentos
│   ├── predictor.py                     # Predictor híbrido (modelo + reglas)
│   └── preprocessor.py                 # Limpieza y normalización de texto
├── app.py                               # Frontend con Streamlit
├── main.py                              # Script principal
├── test.py                              # Pruebas unitarias
├── requirements.txt                     # Dependencias
└── README.md                            # Este archivo

```

## Instalación

```bash
git clone https://github.com/DanielJC1/PYA.git
cd PYA
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Uso

### Ejecutar pipeline completo

```bash
python main.py
```

Carga el corpus combinado, entrena los tres modelos, genera métricas y guarda reportes en `results/`.

### Ejecutar pruebas unitarias

```bash
python test.py
```

### Ejecutar interfaz local

```bash
streamlit run app.py
```

La interfaz permite escribir texto, ver la estimación de riesgo y explorar cómo contribuyen el modelo y las reglas por separado mediante tres vistas: Decisión final, Modelo de regresión y Reglas de contexto.

## Arquitectura híbrida

El predictor combina dos componentes:

- **Clasificador TF-IDF** (peso 70%): Regresión Logística entrenada sobre vectores TF-IDF con n-gramas (1,3) y hasta 8,000 características.
- **Reglas de contexto** (peso 30%): Sistema léxico que pondera insultos, grupos objetivo, exclusión, violencia y contexto atenuante.

Cuando las reglas detectan señal fuerte (score ≥ 0.5), la ponderación se ajusta dinámicamente a 50/50 para dar más peso a la evidencia léxica directa. El sistema también aplica descuentos por contexto positivo hacia grupos y por crítica no identitaria, reduciendo falsos positivos por uso intracomunitario o lenguaje figurado.

## Modelos Soportados

- Logistic Regression
- Random Forest
- Naive Bayes

Cada modelo usa vectorización TF-IDF con n-gramas (1,2) en el pipeline de experimentos, y (1,3) en el predictor híbrido interactivo.

## Próximos Pasos

- [ ] Integrar modelos de transformers (BERT multilingüe)
- [ ] Análisis de sesgo por grupo objetivo
- [ ] Validación cruzada
- [ ] Análisis de importancia de características
- [ ] Validación con anotadores humanos

## Autores



## Licencia

MIT
