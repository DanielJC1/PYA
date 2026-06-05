"""
Modulo para cargar y gestionar datos de riesgo de odio.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Carga y prepara datos para cuantificacion de riesgo de odio."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Carga datos desde archivo CSV."""
        path = self.data_dir / filepath
        return pd.read_csv(path)
    
    def load_combined(self) -> pd.DataFrame:
        """Combina HateCheck español con dataset sintético balanceado."""
        # Cargar HateCheck
        hatecheck = self.load_hatecheck()
        hatecheck = hatecheck[["text", "label"]]

        # Cargar dataset sintético
        synthetic_path = self.data_dir / "sample_data.csv"
        synthetic = pd.read_csv(synthetic_path)[["text", "label"]]

        # Combinar
        combined = pd.concat([hatecheck, synthetic], ignore_index=True)
        combined = combined.dropna(subset=["text"])
        combined = combined[combined["text"].str.strip() != ""]
        combined = combined.reset_index(drop=True)

        print(f"Dataset combinado: {len(combined)} muestras")
        print(f"Distribución final:\n{combined['label'].value_counts()}")
        return combined


    def load_hatecheck(self, filename: str = "test_paul_hatecheck_spanish.csv") -> pd.DataFrame:
        """Carga y prepara el dataset HateCheck español."""
        path = self.data_dir / filename
        df = pd.read_csv(path)

        # Seleccionar solo columnas relevantes
        result = pd.DataFrame()
        result["text"] = df["test_case"]
        result["label"] = (df["label_gold"] == "hateful").astype(int)
        result["target"] = df["target_ident"].fillna("none")
        result["functionality"] = df["functionality"]

        # Eliminar filas con texto vacío
        result = result.dropna(subset=["text"])
        result = result[result["text"].str.strip() != ""]
        result = result.reset_index(drop=True)

        print(f"Dataset cargado: {len(result)} muestras")
        print(f"Distribución de clases:\n{result['label'].value_counts()}")
        print(f"Grupos objetivo:\n{result['target'].value_counts()}")
        return result

    def create_sample_dataset(self, filename: str = "sample_data.csv", n_samples: int = 1000):
        """Intenta cargar HateCheck; si no existe, genera dataset de ejemplo."""
        hatecheck_path = self.data_dir / "test_paul_hatecheck_spanish.csv"
        if hatecheck_path.exists():
            synthetic_path = self.data_dir / "sample_data.csv"
            if synthetic_path.exists():
                return self.load_combined()
            return self.load_hatecheck()

        # Fallback: dataset sintético
        np.random.seed(42)
        hate_templates = [
            "Eres un idiota y no mereces vivir",
            "Deberian deportarte del pais",
            "Tu gente es la culpable de todo",
            "Eres una basura humana",
            "Los tuyos son todos criminales",
            "No merecen derechos",
            "Deberia haber genocidio contra ustedes",
            "Eres un parasito para la sociedad",
            "Tu raza es inferior",
            "Maldita sea tu religion",
            "Asco de inmigrantes, sobran aqui",
            "Expulsen a esa gente del barrio",
            "Todos ustedes son una plaga",
            "No deberian dejar entrar a los musulmanes",
            "Las feministas arruinan todo, hay que callarlas",
            "Ojala maten a todos esos maricones",
            "Esa comunidad da asco y deberia desaparecer",
            "Los pobres solo sirven para delinquir",
            "Pinches indios, nunca aprenden",
            "No quiero ver negros en mi colonia",
        ]
        non_hate_templates = [
            "Hola, como estas?",
            "Tengo una pregunta sobre el proyecto",
            "Me gustaria discutir la politica publica",
            "Creo que deberiamos mejorar la educacion",
            "Que opinas sobre el clima?",
            "Me encanta este libro",
            "Vamos al cine el viernes",
            "Cual es tu pelicula favorita?",
            "No estoy de acuerdo con tu argumento, pero podemos debatirlo",
            "Ese jugador fue malisimo ayer",
            "La pelicula estuvo horrible y perdi mi dinero",
            "Odio madrugar para ir a la oficina",
            "Que coraje me da el trafico",
            "Tu propuesta tiene errores y hay que corregirla",
            "No deberian subir otra vez los impuestos",
            "Ese servicio fue una basura, pedi reembolso",
            "Estoy harto de los spoilers",
            "La comida estaba fria y pesima",
            "Criticar una ideologia no es atacar personas",
            "No me gusta esa cancion",
        ]
        suffixes = ["", " de verdad", " siempre", " otra vez", " aqui", " en este pais"]
        texts, labels = [], []
        for _ in range(n_samples // 2):
            texts.append(f"{np.random.choice(hate_templates)}{np.random.choice(suffixes)}".strip())
            labels.append(1)
            texts.append(f"{np.random.choice(non_hate_templates)}{np.random.choice(suffixes)}".strip())
            labels.append(0)

        df = pd.DataFrame({"text": texts, "label": labels})
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Dataset de ejemplo creado: {output_path}")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide datos en train, validacion y test."""
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val["label"],
        )
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def get_class_distribution(self, df: pd.DataFrame) -> dict:
        """Obtiene distribucion de clases."""
        return df["label"].value_counts().to_dict()