"""
Descarga posts recientes desde X/Twitter y los guarda como CSV.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_DIR
from preprocessor import TextPreprocessor
from twitter_client import TwitterClient


def main():
    query = input("Consulta de búsqueda en X: ").strip()
    if not query:
        print("Debes escribir una consulta.")
        return

    max_results_raw = input("Número de posts a descargar [10]: ").strip()
    max_results = int(max_results_raw) if max_results_raw else 10

    client = TwitterClient()
    df = client.fetch_recent_tweets(query=query, max_results=max_results)

    if df.empty:
        print("No se encontraron posts para esa consulta.")
        return

    preprocessor = TextPreprocessor()
    df["clean_text"] = df["text"].apply(preprocessor.clean_text)

    output_path = DATA_DIR / "twitter_posts.csv"
    df.to_csv(output_path, index=False)

    print(f"Posts guardados en: {output_path}")
    print(df[["id", "text", "clean_text"]].head())


if __name__ == "__main__":
    main()
