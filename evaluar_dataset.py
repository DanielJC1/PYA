import sys
sys.path.insert(0, 'src')
import pandas as pd
from predictor import HateSpeechPredictor
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from config import DATA_DIR

loader = DataLoader(DATA_DIR)
df = loader.create_sample_dataset()
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataframe(df, 'text')
predictor = HateSpeechPredictor()
predictor.fit(df['text'].tolist(), df['label'].tolist())

# Cargar sample_data original (sin preprocesar)
sample = pd.read_csv('data/sample_data.csv')

errores = []
for _, row in sample.iterrows():
    text = row['text']
    label_real = row['label']
    r = predictor.predict_one(str(text))
    label_pred = r['prediction']
    if label_real != label_pred:
        errores.append({
            'texto': text,
            'real': 'odio' if label_real == 1 else 'no_odio',
            'predicho': r['content_type'],
            'score': r['probabilities']['odio'],
            'rule_score': r['rule_score'],
            'hits': r['matched_terms'],
        })

print(f"Total evaluados: {len(sample)}")
print(f"Errores encontrados: {len(errores)}")
print(f"Tasa de error: {len(errores)/len(sample)*100:.1f}%")
print()
print("=== DETALLE DE ERRORES ===")
for e in errores:
    print(f"[{e['real']} → {e['predicho']}] score:{e['score']:.2f} rule:{e['rule_score']:.2f} | {e['texto']}")
    if e['hits']:
        print(f"  hits: {e['hits']}")

r = predictor.predict_one('Eres un parásito para la sociedad')
print(f"label: {r['label']}, content_type: {r['content_type']}")
print(f"score: {r['probabilities']['odio']:.3f}, rule_score: {r['rule_score']:.3f}")
print(f"hits: {r['matched_terms']}, reasons: {r['reasons']}")
print(f"has_targets: {r['target_groups']}")