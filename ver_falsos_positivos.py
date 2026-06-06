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

# Cargar dataset HateCheck original
hatecheck = pd.read_csv('data/test_paul_hatecheck_spanish.csv')

fp = []
fn = []
for _, row in hatecheck.iterrows():
    text = str(row['test_case'])
    label_real = 1 if row['label_gold'] == 'hateful' else 0
    r = predictor.predict_one(text)
    label_pred = r['prediction']
    if label_real == 0 and label_pred == 1:
        fp.append({'texto': text, 'score': r['probabilities']['odio'], 'hits': r['matched_terms']})
    elif label_real == 1 and label_pred == 0:
        fn.append({'texto': text, 'score': r['probabilities']['odio'], 'hits': r['matched_terms']})

print(f"Falsos positivos: {len(fp)}")
print(f"Falsos negativos: {len(fn)}")
print()
print("=== FALSOS POSITIVOS ===")
for e in fp:
    print(f"  score:{e['score']:.2f} hits:{e['hits']} | {e['texto']}")

print("=== MUESTRA DE FALSOS NEGATIVOS ===")
for e in fn[:20]:
    print(f"  score:{e['score']:.2f} | {e['texto']}")