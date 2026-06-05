import sys
sys.path.insert(0, 'src')
import pandas as pd

df = pd.read_csv('results/error_report.csv')
fp = df[df['error_type'] == 'false_positives']
print(f'Total falsos positivos: {len(fp)}')
print()
for _, row in fp.iterrows():
    print(f"  [{row['index']}] {row['text_preview']}")