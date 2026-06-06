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

text = 'Debería haber genocidio contra ustedes otra vez'
r = predictor.predict_one(text)
print(f"label:        {r['label']}")
print(f"content_type: {r['content_type']}")
print(f"hate_score:   {r['probabilities']['odio']:.3f}")
print(f"model_prob:   {r['model_probability']:.3f}")
print(f"rule_score:   {r['rule_score']:.3f}")
print(f"reasons:      {r['reasons']}")
print(f"hits:         {r['matched_terms']}")