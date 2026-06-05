import sys
sys.path.insert(0, 'src')
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

casos = [
    'No estoy de acuerdo con tu propuesta, pero podemos debatirlo.',
    'Eres un idiota.',
    'Esa gente no merece derechos y deberian expulsarlos.',
    'Los trans como tu son heroes.',
    'Pinches indios, nunca aprenden.',
    'La pelicula estuvo horrible y perdi mi dinero.',
]

print()
for texto in casos:
    r = predictor.predict_one(texto)
    print(f"{r['label']:10}  {r['probabilities']['odio']:.2f}  {texto[:55]}")

r = predictor.predict_one('Los trans como tu son heroes.')
print(f"rule_score: {r['rule_score']}")
print(f"hits: {r['matched_terms']}")
print(f"reasons: {r['reasons']}")
print(f"targets: {r['target_groups']}")