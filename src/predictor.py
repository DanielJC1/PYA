"""
Utilidades para entrenar un clasificador y enriquecer la prediccion
con reglas para bajar errores obvios en una demo local.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import CLASS_LABELS, RANDOM_SEED, MODEL_WEIGHT, RULES_WEIGHT, HATE_THRESHOLD


class HateSpeechPredictor:
    """Combina un clasificador TF-IDF con reglas de contexto simples."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        self.is_trained = False
        self.lexicon = {
            "target_groups": {
                "inmigrantes", "migrantes", "musulmanes", "negros", "indios",
                "indigenas", "indigena"
                "gays", "lesbianas", "trans", "mujeres", "judios", "pobres",
                "feministas", "discapacitados", "refugiados", "homosexuales",
            },
            "slurs": {
                "sudaca", "maricon", "joto", "puta", "zorra", "parasito",
                "plaga", "basura humana", "negro de mierda", "pinches indios",
                "pinches", "criminales", "criminal", "inferior", "raza inferior",
                "maldita sea tu religion", "maldita sea su religion",
            },
            "abuse": {
                "puta", "puto", "zorra", "naca",
                "tonta", "tonto", "idiota", "imbecil", "pendeja", "pendejo",
                "estupida", "estupido", "tarada", "tarado", "basura",
                "cabron", "cabrona", "asquerosa", "asqueroso",
                "gorda", "gordo",
            },
            "violence": {
                "matar", "maten", "genocidio", "linchar", "exterminar",
                "desaparecer", "morir", "encerrar", "matate", "suicidate",
                "no mereces vivir", "no merece vivir", "no merecen vivir",
            },
            "exclusion": {
                "deportar", "expulsar", "sin derechos", "quitarles derechos",
                "no merecen derechos", "fuera del pais", "no deberian entrar",
                "vete a tu pais", "regresate a tu pais", "largate de mi pais",
                "no mereces derechos", "no merece derechos", "no mereces", "no merecen",
                "deportarte", "deportarlos", "deportarlas",
            },
            "non_hate_context": {
                "pelicula", "trafico", "oficina", "examen", "app", "proyecto",
                "libro", "servicio", "debate", "argumento", "clima", "cancion",
            },
        }

    def _normalize_for_rules(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _contains_obfuscated_phrase(self, text: str, phrase: str) -> bool:
        """Detecta frases aunque tengan simbolos intermedios como m*tate."""
        compact = self._normalize_for_rules(text).replace(" ", "")
        if phrase in compact:
            return True
        special_patterns = {
            "matate": r"m[\W_]*[a4@*]?[\W_]*t[\W_]*a[\W_]*t[\W_]*e",
            "suicidate": r"s[\W_]*u[\W_]*i[\W_]*c[\W_]*i[\W_]*d[\W_]*[a4@*]?[\W_]*t[\W_]*e",
        }
        if phrase in special_patterns and re.search(special_patterns[phrase], text.lower()):
            return True
        letters = [re.escape(ch) for ch in phrase if ch.strip()]
        if not letters:
            return False
        pattern = r"[\W_]*".join(letters)
        return bool(re.search(pattern, text.lower()))

    def _rule_signal(self, text: str) -> Dict[str, object]:
        normalized = self._normalize_for_rules(text)
        hits: List[str] = []
        reasons: List[str] = []
        score = 0.0

        for term in self.lexicon["slurs"]:
            if term in normalized:
                score += 0.32
                hits.append(term)
        abuse_hits = []
        for term in self.lexicon["abuse"]:
            if term in normalized:
                score += 0.10
                hits.append(term)
                abuse_hits.append(term)
        for term in self.lexicon["violence"]:
            if term in normalized:
                score += 0.22
                hits.append(term)

        if self._contains_obfuscated_phrase(text, "matate"):
            score = max(score, 0.72)
            hits.append("matate")
            reasons.append("autolesion_explicita")

        if self._contains_obfuscated_phrase(text, "suicidate"):
            score = max(score, 0.76)
            hits.append("suicidate")
            reasons.append("autolesion_explicita")
        for term in self.lexicon["exclusion"]:
            if term in normalized:
                score += 0.22
                hits.append(term)

        if re.search(r"\b(vete|largate|regresate|vayanse)\b.*\b(pais|paisito)\b", normalized):
            score += 0.24
            hits.append("expulsion_nativista")
            reasons.append("expulsion_nativista")

        if re.search(r"\b(esa gente|ellos|ellas|ustedes|tu gente)\b.*\b(no merece[n]? derechos|sin derechos|quitarles derechos)\b", normalized):
            score = max(score, 0.68)
            hits.append("exclusion_dirigida")
            reasons.append("exclusion_dirigida")

        targets = [term for term in self.lexicon["target_groups"] if term in normalized]
        if targets:
            score += 0.16
            hits.extend(targets)
            reasons.append("grupo_objetivo")

        has_direct_attack = bool(re.search(r"\b(eres|son|ustedes|esa gente|ellos|ellas|tu gente)\b", normalized))
        if has_direct_attack and hits:
            score += 0.10
            reasons.append("ataque_directo")
        elif has_direct_attack and abuse_hits:
            score += 0.08
            reasons.append("insulto_directo")

        uppercase_ratio = (
            len(re.findall(r"[A-ZÁÉÍÓÚÑ]", text)) / max(len(re.sub(r"\s+", "", text)), 1)
            if text else 0
        )
        if uppercase_ratio > 0.35:
            score += 0.05
            reasons.append("enfasis_mayusculas")

        exclamations = len(re.findall(r"!", text))
        if exclamations >= 2 and hits:
            score += 0.04
            reasons.append("enfasis_exclamaciones")

        if not targets and any(term in normalized for term in self.lexicon["non_hate_context"]):
            score -= 0.12
            reasons.append("contexto_no_dirigido")

        if re.search(r"\b(no estoy de acuerdo|debatir|criticar|argumento|idea|propuesta)\b", normalized):
            score -= 0.08
            reasons.append("critica_no_identitaria")

        if re.search(r"\b(son heroes|son hermosos|vale muchisimo|no vale menos|son fantasticos|deberian admirar|como tu son)\b", normalized):
            score -= 0.20
            reasons.append("contexto_positivo_grupo")

        score = float(np.clip(score, 0.0, 0.95))
        return {
            "score": score,
            "hits": sorted(set(hits)),
            "reasons": reasons,
            "normalized": normalized,
            "abuse_hits": sorted(set(abuse_hits)),
            "targets": sorted(set(targets)),
        }

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Entrena el modelo con textos y etiquetas."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict_one(self, text: str) -> Dict:
        """Predice una sola pieza de texto."""
        if not self.is_trained:
            raise RuntimeError("El predictor aun no ha sido entrenado.")

        X = self.vectorizer.transform([text])
        model_proba = self.model.predict_proba(X)[0]
        rules = self._rule_signal(text)

        # Si las reglas dan señal fuerte (>= 0.5), aumentar su peso
        if rules["score"] >= 0.5:
            hate_score = float(np.clip((model_proba[1] * 0.5) + (rules["score"] * 0.5), 0.0, 1.0))
        else:
            hate_score = float(np.clip((model_proba[1] * MODEL_WEIGHT) + (rules["score"] * RULES_WEIGHT), 0.0, 1.0))
        # Si hay contexto positivo hacia un grupo, limitar el score
        if "contexto_positivo_grupo" in rules["reasons"]:
            hate_score = min(hate_score, 0.40)
            pred = 0
        pred = int(hate_score >= HATE_THRESHOLD)
        confidence = float(max(hate_score, 1 - hate_score))
        has_abuse = bool(rules["abuse_hits"])
        has_targets = bool(rules["targets"])
        has_self_harm = any(term in rules["hits"] for term in {"matate", "suicidate"})
        has_violence = any(term in rules["hits"] for term in {"genocidio", "matar", "maten", "linchar", "exterminar", "desaparecer", "encerrar"})
        has_hate_markers = bool(has_targets or rules["score"] >= 0.30 or has_self_harm or has_violence)

        if has_self_harm:
            content_type = "odio"
            pred = 1
            hate_score = max(hate_score, 0.72)
            confidence = float(max(hate_score, 1 - hate_score))
        elif has_abuse and not has_targets and rules["score"] < 0.35:
            content_type = "agresion_verbal"
            pred = 0
            hate_score = min(hate_score, 0.42)
            confidence = float(max(hate_score, 1 - hate_score))
        elif pred == 1 and has_hate_markers:
            content_type = "odio"
        elif has_abuse:
            content_type = "agresion_verbal"
        else:
            content_type = "neutral"

        return {
            "prediction": pred,
            "label": CLASS_LABELS[pred],
            "content_type": content_type,
            "confidence": confidence,
            "probabilities": {
                CLASS_LABELS[0]: float(1 - hate_score),
                CLASS_LABELS[1]: hate_score,
            },
            "model_probability": float(model_proba[1]),
            "rule_score": float(rules["score"]),
            "matched_terms": rules["hits"],
            "abuse_hits": rules["abuse_hits"],
            "target_groups": rules["targets"],
            "reasons": rules["reasons"],
            "normalized_text": rules["normalized"],
        }
