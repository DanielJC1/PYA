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
    ('neutral',          'No estoy de acuerdo con tu propuesta, pero podemos debatirlo.'),
    ('agresion_verbal',  'Eres un idiota.'),
    ('odio',             'Esa gente no merece derechos y deberian expulsarlos.'),
    ('neutral',          'Los trans como tu son heroes.'),
    ('odio',             'Pinches indios, nunca aprenden.'),
    ('neutral',          'La pelicula estuvo horrible y perdi mi dinero.'),
    ('odio',             'Deberia haber genocidio contra ustedes otra vez'),
    ('odio',             'Ojala maten a todos esos maricones'),
    ('odio',             'Eres un parasito para la sociedad'),
    ('odio',             'Eres un idiota y no mereces vivir'),
    ('odio', 'No mereces derechos'),
    ('odio', 'Los tuyos son todos criminales'),
    ('odio', 'Tu raza es inferior'),
    ('odio', 'Deberían deportarte del país'),
    ('odio', 'Maldita sea tu religión'),
]

print(f"{'ESPERADO':16}  {'RESULTADO':16}  {'SCORE':5}  TEXTO")
print("-" * 80)
ok_count = 0
for esperado, texto in casos:
    r = predictor.predict_one(texto)
    resultado = r['content_type']
    score = r['probabilities']['odio']
    ok = '✓' if esperado == resultado else '✗'
    if ok == '✓':
        ok_count += 1
    print(f"{ok} {esperado:16}  {resultado:16}  {score:.2f}  {texto[:50]}")

print(f"\nResultado: {ok_count}/{len(casos)} correctos")

print()
print("=== DEBUG ===")
for texto in ['No mereces derechos', 'Los tuyos son todos criminales']:
    r = predictor.predict_one(texto)
    print(f"texto: {texto}")
    print(f"  hits: {r['matched_terms']}")
    print(f"  rule_score: {r['rule_score']:.3f}")
    print(f"  reasons: {r['reasons']}")
    print(f"  normalized: {r['normalized_text']}")
    print()

for texto in ['Tu raza es inferior', 'Maldita sea tu religión']:
    r = predictor.predict_one(texto)
    print(f"texto: {texto}")
    print(f"  hits: {r['matched_terms']}")
    print(f"  rule_score: {r['rule_score']:.3f}")
    print(f"  normalized: {r['normalized_text']}")
    print()