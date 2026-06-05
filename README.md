# Cuantificación del Odio en Redes Sociales

Proyecto para estimar y analizar la probabilidad de contenido de odio en redes sociales. El sistema usa una arquitectura híbrida: combina un clasificador TF-IDF con reglas de contexto para cuantificar riesgo sin tratar cualquier insulto como una confirmación automática.

## Alcance

- Estima riesgo a partir de texto y reglas de contexto.
- No determina de forma definitiva si un mensaje constituye odio.
- Debe usarse como apoyo para análisis, priorización y revisión humana.

## Dataset

Se usa el corpus **HateCheck en español** (Paul/hatecheck-spanish), un conjunto de 3,745 casos de prueba anotados para detección de odio en español, con cobertura de 7 grupos objetivo: personas gay, trans, indígenas, discapacitadas, mujeres, negras y judías.

## Resultados

Métricas sobre el conjunto de prueba (749 muestras, 70% odio / 30% no odio):

| Modelo              | Accuracy | F1     | ROC-AUC |
|---------------------|----------|--------|---------|
| Logistic Regression | 92.7%    | 92.3%  | 97.5%   |
| Random Forest       | **95.3%**| **95.2%**| **98.1%**|
| Naive Bayes         | 91.9%    | 91.4%  | 95.5%   |

Análisis de errores (Logistic Regression):

| Tipo               | Cantidad |
|--------------------|----------|
| Verdaderos positivos | 526    |
| Verdaderos negativos | 168    |
| Falsos positivos     | 55     |
| Falsos negativos     | 0      |

## Estructura del Proyecto

```
PYA/
├── data/                         # Datos del proyecto
│   └── test_paul_hatecheck_spanish.csv
├── docs/                         # Versión estática para GitHub Pages
├── results/                      # Resultados y reportes
├── src/                          # Código fuente
│   ├── config.py                 # Configuración general
│   ├── data_loader.py            # Carga y preprocesamiento de datos
│   ├── evaluator.py              # Evaluación y análisis de errores
│   ├── experiments.py            # Gestión de experimentos
│   ├── predictor.py              # Predictor híbrido (modelo + reglas)
│   └── preprocessor.py          # Limpieza y normalización de texto
├── app.py                        # Frontend con Streamlit
├── main.py                       # Script principal
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo

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

Esto carga el dataset, entrena los tres modelos, genera métricas y guarda los reportes en `results/`.

### Ejecutar interfaz local

```bash
streamlit run app.py
```

La interfaz permite escribir texto, ver la estimación de riesgo y explorar cómo contribuyen el modelo y las reglas por separado.

## Arquitectura híbrida

El predictor combina dos componentes:

- **Clasificador TF-IDF** (peso 70%): Regresión Logística entrenada sobre vectores TF-IDF con n-gramas (1,3).
- **Reglas de contexto** (peso 30%): Sistema léxico que pondera insultos, grupos objetivo, exclusión, violencia y contexto atenuante.

La combinación permite distinguir agresión verbal genérica de odio identitario dirigido, reduciendo falsos positivos por uso intracomunitario o lenguaje figurado.

## Modelos Soportados

- Logistic Regression
- Random Forest
- Naive Bayes

Cada modelo usa vectorización TF-IDF con n-gramas (1,2).

## Próximos Pasos

- [ ] Integrar modelos de transformers (BERT multilingüe)
- [ ] Análisis de sesgo por grupo objetivo
- [ ] Validación cruzada
- [ ] Análisis de importancia de características
- [ ] Validación con anotadores humanos

## Autores


## Licencia

MIT