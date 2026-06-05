import sys
sys.path.insert(0, 'src')

from predictor import HateSpeechPredictor
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from config import DATA_DIR

loader = DataLoader(DATA_DIR)
df = loader.create_sample_dataset()
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataframe(df, "text")

predictor = HateSpeechPredictor()
predictor.fit(df["text"].tolist(), df["label"].tolist())

text = "Pinches indios, nunca aprenden."
clean = preprocessor.clean_text(text)

print(f"Texto original: {text}")
print(f"Texto limpio:   {clean}")
print()

result = predictor.predict_one(text)
print(f"Predicción con texto original:")
print(f"  label:      {result['label']}")
print(f"  score_odio: {result['probabilities']['odio']:.4f}")
print(f"  rule_score: {result['rule_score']:.4f}")
print(f"  hits:       {result['matched_terms']}")
print()

result2 = predictor.predict_one(clean)
print(f"Predicción con texto limpio:")
print(f"  label:      {result2['label']}")
print(f"  score_odio: {result2['probabilities']['odio']:.4f}")
print(f"  rule_score: {result2['rule_score']:.4f}")
print(f"  hits:       {result2['matched_terms']}")