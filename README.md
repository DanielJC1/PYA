# Cuantificación del Odio en Redes Sociales

Proyecto para estimar y analizar la probabilidad de contenido de odio en redes sociales. El sistema usa una arquitectura híbrida: combina un clasificador TF-IDF con reglas de contexto para cuantificar riesgo sin tratar cualquier insulto como una confirmación automática.

## Alcance

- Estima riesgo a partir de texto y reglas de contexto.
- No determina de forma definitiva si un mensaje constituye odio.
- Debe usarse como apoyo para análisis, priorización y revisión humana.

## Demo

La interfaz está desplegada públicamente con backend real:

🔗 [https://apit-mors.streamlit.app](https://apit-mors.streamlit.app)

## Dataset

El sistema entrena sobre un corpus combinado de dos fuentes:

- **HateCheck en español** (Paul/hatecheck-spanish): 3,745 casos de prueba anotados para detección de odio en español, con cobertura de 7 grupos objetivo: personas gay, trans, indígenas, discapacitadas, mujeres, negras y judías.
- **Dataset sintético balanceado**: 1,000 ejemplos en español (500 odio / 500 no odio) con frases cotidianas de no odio para balancear la distribución del corpus.

El corpus combinado tiene 4,745 muestras con distribución 66% odio / 34% no odio.

## Resultados

### Pipeline de entrenamiento (949 muestras de prueba)

| Modelo              | Accuracy | F1     | ROC-AUC |
|---------------------|----------|--------|---------|
| Logistic Regression | 93.9%    | 93.7%  | 97.7%   |
| Random Forest       | **97.3%**| **97.2%**| **99.4%**|
| Naive Bayes         | 93.6%    | 93.4%  | 96.4%   |

### Análisis de errores — Pipeline (Logistic Regression, 949 muestras)

| Tipo                 | Cantidad |
|----------------------|----------|
| Verdaderos positivos | 626      |
| Verdaderos negativos | 265      |
| Falsos positivos     | 58       |
| Falsos negativos     | 0        |

### Análisis de errores — Predictor híbrido (HateCheck completo, 3,745 muestras)

| Tipo                 | Cantidad |
|----------------------|----------|
| Verdaderos positivos | 2,310    |
| Verdaderos negativos | 1,102    |
| Falsos positivos     | 11       |
| Falsos negativos     | 322      |

Los 11 falsos positivos corresponden a afirmaciones positivas sobre grupos objetivo, violencia dirigida a objetos o conceptos, y lenguaje hiperbólico fuera de contexto. Los 322 falsos negativos reflejan una limitación del sistema ante expresiones de odio implícito que no usan slurs o términos explícitos, como "Odio a los gays" o "Estos negros me dan asco", donde el modelo estadístico no alcanza el umbral de decisión sin señal léxica directa.

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

### Requisitos previos

Se requiere **Python 3.10 o superior**. Puedes verificar tu versión con:

```bash
python --version
```

Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/) o instálalo por línea de comandos:

```bash
# Windows (usando winget)
winget install Python.Python.3.13

# Mac (usando Homebrew)
brew install python

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

### Pasos

```bash
git clone https://github.com/DanielJC1/PYA.git
cd PYA
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Uso

### Demo en línea

La forma más rápida de probar el sistema es acceder directamente a la demo desplegada:

🔗 [https://apit-mors.streamlit.app](https://apit-mors.streamlit.app)

### Ejecutar pipeline completo (local)

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
- Veraza Garcia Amy Valentina
- Magno García Omar
- Jiménez Chávez Daniel Orlando

## Licencia

MIT
