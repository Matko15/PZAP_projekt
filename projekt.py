import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class FitnessDataProcessor:
    """Prikupljanje, čišćenje i integracija podataka iz više izvora"""

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.kaggle_data = None
        self.api_data = None
        self.integrated_data = None

    def load_kaggle_json(self, filepath: str):
        """Učitaj exercises.json (Kaggle dataset, JSON format)"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            self.kaggle_data = df
            return df
        except FileNotFoundError:
            return None

    def fetch_api_data(self, api_url, headers=None, params=None):
        """
        Dohvati podatke s javnog fitness API-ja.
        Ako ne radi, koristi sample podatke.
        """
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            api_json = response.json()

            backup_path = self.data_dir / "api_backup.json"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(api_json, f, indent=2, ensure_ascii=False)

            if isinstance(api_json, list):
                df = pd.DataFrame(api_json)
            else:
                df = pd.json_normalize(api_json)

            self.api_data = df
            return df

        except requests.exceptions.RequestException:
            self.api_data = self._create_sample_api_data()
            return self.api_data

    def _create_sample_api_data(self):
        """Sample podaci za slučaj da API ne radi"""
        sample_data = {
            "id": ["api_1", "api_2", "api_3", "api_4", "api_5"],
            "name": [
                "barbell bench press",
                "deadlift",
                "barbell squat",
                "pull-ups",
                "dumbbell rows",
            ],
            "target": ["chest", "back", "legs", "back", "back"],
            "bodyPart": ["chest", "back", "legs", "back", "back"],
            "equipment": ["barbell", "barbell", "barbell", "bodyweight", "dumbbell"],
            "difficulty": [
                "intermediate",
                "advanced",
                "intermediate",
                "intermediate",
                "beginner",
            ],
        }
        return pd.DataFrame(sample_data)

    def clean_data(self, df: pd.DataFrame, source_name="Unknown"):
        if df is None or df.empty:
            return df

        scalar_cols = [
            c
            for c in df.columns
            if df[c].map(lambda x: not isinstance(x, (list, dict))).all()
        ]
        df = df.drop_duplicates(subset=scalar_cols)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

        return df

    def _standardize_kaggle(self):
        """
        Kaggle exercises.json:
        - name
        - targetMuscles (lista)
        - bodyParts (lista)
        - equipments (lista)
        """
        df = self.kaggle_data.copy()
        df["source"] = "kaggle"

        def first_or_none(x):
            if isinstance(x, list) and len(x) > 0:
                return x[0]
            return None

        df["exercise_name"] = df["name"]
        df["target_muscle"] = df["targetMuscles"].apply(first_or_none)
        df["body_part"] = df["bodyParts"].apply(first_or_none)
        df["equipment"] = df["equipments"].apply(first_or_none)

        if "secondaryMuscles" in df.columns:
            df["secondary_muscles"] = df["secondaryMuscles"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else ""
            )
        else:
            df["secondary_muscles"] = ""

        df["target_muscle"] = df["target_muscle"].fillna("other")
        df["body_part"] = df["body_part"].fillna("other")
        df["equipment"] = df["equipment"].fillna("body weight")

        df["difficulty"] = "intermediate"

        cols = [
            "exercise_name",
            "target_muscle",
            "body_part",
            "equipment",
            "secondary_muscles",
            "difficulty",
            "source",
        ]
        return df[cols]

    def _standardize_api(self):
        """
        API (ili sample):
        - name
        - target (string)
        - bodyPart (string)
        - equipment (string)
        - difficulty (može postojati / ne)
        """
        df = self.api_data.copy()
        df["source"] = "api"

        col_map = {
            "name": "exercise_name",
            "target": "target_muscle",
            "bodyPart": "body_part",
            "equipment": "equipment",
        }
        df = df.rename(columns=col_map)

        if "secondary_muscles" not in df.columns:
            df["secondary_muscles"] = ""

        if "difficulty" not in df.columns:
            df["difficulty"] = "unknown"

        cols = [
            "exercise_name",
            "target_muscle",
            "body_part",
            "equipment",
            "secondary_muscles",
            "difficulty",
            "source",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols]

    def integrate_datasets(self):
        """Integracija Kaggle + API u jedan DataFrame"""
        if self.kaggle_data is None or self.api_data is None:
            return None

        kaggle_std = self._standardize_kaggle()
        api_std = self._standardize_api()

        integrated = pd.concat([kaggle_std, api_std], ignore_index=True)

        integrated = integrated.drop_duplicates(
            subset=["exercise_name", "target_muscle"], keep="first"
        )

        self.integrated_data = integrated
        return integrated

    def save_integrated_data(self, filename="integrated_fitness_data.csv"):
        if self.integrated_data is None:
            return None

        path = self.data_dir / filename
        self.integrated_data.to_csv(path, index=False, encoding="utf-8")
        return path

class FitnessAnalyzer:
    """Analiza integriranih podataka"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}

    def analyze_by_muscle_group(self):
        dist = self.data["target_muscle"].value_counts()
        self.results["muscles"] = dist
        return dist

    def analyze_by_difficulty(self):
        dist = self.data["difficulty"].value_counts()
        self.results["difficulty"] = dist
        return dist

    def analyze_by_equipment(self):
        dist = self.data["equipment"].value_counts()
        self.results["equipment"] = dist
        return dist

    def analyze_by_source(self):
        dist = self.data["source"].value_counts()
        self.results["sources"] = dist
        return dist

    def get_training_recommendations(self, goal="strength"):
        """
        Vrati preporuke vježbi ovisno o cilju:
        - strength
        - hypertrophy
        - full_body
        - push_pull_legs
        """
        df = self.data

        if goal == "strength":
            recs = df[
                (df["equipment"].isin(["barbell", "dumbbell", "bodyweight"]))
                & (df["difficulty"].isin(["intermediate", "advanced"]))
            ]
            focus = "Teži višezglobni pokreti (compound lifts) za snagu."

        elif goal == "hypertrophy":
            recs = df[
                (df["equipment"].isin(["dumbbell", "cable", "machine"]))
                & (df["difficulty"].isin(["beginner", "intermediate", "unknown"]))
            ]
            focus = "Kontrolirani pokreti, veći volumen za hipertrofiju."

        elif goal == "full_body":
            recs = df[df["difficulty"] != "advanced"]
            focus = "Balansiran izbor vježbi za cijelo tijelo."

        elif goal == "push_pull_legs":
            recs = df
            focus = "Vježbe raspoređene po push/pull/legs skupinama."

        else:
            recs = df
            focus = "Opći popis vježbi."

        return {"focus": focus, "exercises": recs, "count": len(recs)}

    def run_full_analysis(self):
        self.analyze_by_muscle_group()
        self.analyze_by_difficulty()
        self.analyze_by_equipment()
        self.analyze_by_source()


class FitnessVisualizer:
    """Vizualizacije rezultata analize"""

    def __init__(self, data: pd.DataFrame, analyzer: FitnessAnalyzer):
        self.data = data
        self.analyzer = analyzer

    def visualize_all(self, output_dir="visualizations"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.plot_muscle_distribution(out)
        self.plot_difficulty_distribution(out)
        self.plot_equipment_distribution(out)
        self.plot_source_comparison(out)
        self.plot_muscle_difficulty_heatmap(out)

    def plot_muscle_distribution(self, output_dir: Path):
        fig, ax = plt.subplots()
        muscles = self.analyzer.results["muscles"].head(15)
        muscles.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Broj vježbi")
        ax.set_ylabel("Mišična skupina")
        ax.set_title("Raspodjela vježbi po mišičnim skupinama")
        plt.tight_layout()
        path = output_dir / "01_muscle_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_difficulty_distribution(self, output_dir: Path):
        fig, ax = plt.subplots()
        diff = self.analyzer.results["difficulty"]
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]
        ax.pie(
            diff,
            labels=diff.index,
            autopct="%1.1f%%",
            colors=colors[: len(diff)],
            startangle=90,
        )
        ax.set_title("Raspodjela vježbi po težini (difficulty)")
        plt.tight_layout()
        path = output_dir / "02_difficulty_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_equipment_distribution(self, output_dir: Path):
        fig, ax = plt.subplots()
        eq = self.analyzer.results["equipment"].head(15)
        eq.plot(kind="bar", ax=ax, color="coral")
        ax.set_xlabel("Oprema")
        ax.set_ylabel("Broj vježbi")
        ax.set_title("Raspodjela vježbi po opremi")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = output_dir / "03_equipment_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_source_comparison(self, output_dir: Path):
        fig, ax = plt.subplots()
        src = self.analyzer.results["sources"]
        src.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"])
        ax.set_xlabel("Izvor")
        ax.set_ylabel("Broj vježbi")
        ax.set_title("Usporedba izvora podataka (Kaggle vs API)")
        plt.tight_layout()
        path = output_dir / "04_source_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_muscle_difficulty_heatmap(self, output_dir: Path):
        table = pd.crosstab(self.data["target_muscle"], self.data["difficulty"]).fillna(
            0
        )
        table = table.iloc[:15]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            table,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Broj vježbi"},
        )
        ax.set_title("Matrica vježbi: mišične skupine vs težina")
        ax.set_xlabel("Težina (difficulty)")
        ax.set_ylabel("Mišična skupina")
        plt.tight_layout()
        path = output_dir / "05_muscle_difficulty_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    processor = FitnessDataProcessor()

    kaggle_path = "data/exercises.json"
    processor.load_kaggle_json(kaggle_path)

    api_url = "https://exercisedb.p.rapidapi.com/exercises"
    headers = {}
    processor.fetch_api_data(api_url, headers=headers)

    if processor.kaggle_data is not None:
        processor.kaggle_data = processor.clean_data(processor.kaggle_data, "Kaggle")
    if processor.api_data is not None:
        processor.api_data = processor.clean_data(processor.api_data, "API")

    integrated = processor.integrate_datasets()
    if integrated is None:
        return

    processor.save_integrated_data()

    analyzer = FitnessAnalyzer(integrated)
    analyzer.run_full_analysis()

    for goal in ["strength", "hypertrophy", "full_body", "push_pull_legs"]:
        analyzer.get_training_recommendations(goal)

    visualizer = FitnessVisualizer(integrated, analyzer)
    visualizer.visualize_all()


if __name__ == "__main__":
    main()