"""
Cliente simple para obtener posts recientes desde X API v2.
"""
import os
from typing import Dict, List, Optional

import pandas as pd
import requests


class TwitterClient:
    """Cliente para consultar posts recientes en X/Twitter."""

    BASE_URL = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or os.getenv("X_BEARER_TOKEN")
        if not self.bearer_token:
            raise ValueError(
                "No se encontró el token de X. Define la variable de entorno X_BEARER_TOKEN."
            )

    def fetch_recent_tweets(
        self,
        query: str,
        max_results: int = 10,
        lang: str = "es",
    ) -> pd.DataFrame:
        """Busca posts recientes y los devuelve en un DataFrame."""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
        }
        query_parts = [query.strip()]
        if lang:
            query_parts.append(f"lang:{lang}")
        query_parts.append("-is:retweet")

        params = {
            "query": " ".join(query_parts),
            "max_results": max_results,
            "tweet.fields": "created_at,lang,author_id,public_metrics",
        }

        response = requests.get(
            self.BASE_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        tweets = payload.get("data", [])
        return pd.DataFrame([self._normalize_tweet(tweet) for tweet in tweets])

    @staticmethod
    def _normalize_tweet(tweet: Dict) -> Dict:
        """Normaliza la respuesta del API a columnas planas."""
        metrics = tweet.get("public_metrics", {})
        return {
            "id": tweet.get("id"),
            "text": tweet.get("text", ""),
            "author_id": tweet.get("author_id"),
            "created_at": tweet.get("created_at"),
            "lang": tweet.get("lang"),
            "retweet_count": metrics.get("retweet_count", 0),
            "reply_count": metrics.get("reply_count", 0),
            "like_count": metrics.get("like_count", 0),
            "quote_count": metrics.get("quote_count", 0),
        }
