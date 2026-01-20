import pandas as pd
from pathlib import Path
import json
from flask import Flask, jsonify, request, Response
DATA_PATH = Path("data") / "integrated_fitness_data.csv"

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame(
        columns=[
            "exercise_name",
            "target_muscle",
            "body_part",
            "equipment",
            "secondary_muscles",
            "difficulty",
            "source",
        ]
    )

@app.route("/", methods=["GET"])
def index():
    return """
    <h1>Fitness Exercises API</h1>
    <ul>
      <li><a href="/exercises?limit=10">Prvih 10 vježbi</a></li>
      <li><a href="/exercises?muscle=back">Vježbe za leđa</a></li>
      <li><a href="/exercises?muscle=chest">Vježbe za prsa</a></li>
      <li><a href="/exercises?muscle=legs">Vježbe za noge</a></li>
      <li><a href="/recommendations?goal=strength">Preporuke za snagu (strength)</a></li>
      <li><a href="/recommendations?goal=hypertrophy">Preporuke za hipertrofiju (hypertrophy)</a></li>
    </ul>
    """
@app.route("/exercises", methods=["GET"])
def get_exercises():
    """
    Vrati listu vježbi.

    Query parametri (svi opcionalni):
        muscle      -> filter po target_muscle
        equipment   -> filter po equipment
        difficulty  -> filter po difficulty
        source      -> filter po source (kaggle/api)
        limit       -> max broj zapisa (default 50)
    """
    if df.empty:
        return jsonify({"error": "Nema učitanih podataka."}), 500

    filtered = df.copy()

    muscle = request.args.get("muscle")
    equipment = request.args.get("equipment")
    difficulty = request.args.get("difficulty")
    source = request.args.get("source")
    limit = request.args.get("limit", default=50, type=int)

    if muscle:
        filtered = filtered[filtered["target_muscle"].str.contains(muscle.lower(), na=False)]
    if equipment:
        filtered = filtered[filtered["equipment"].str.contains(equipment.lower(), na=False)]
    if difficulty:
        filtered = filtered[filtered["difficulty"].str.contains(difficulty.lower(), na=False)]
    if source:
        filtered = filtered[filtered["source"].str.contains(source.lower(), na=False)]

    filtered = filtered.head(limit)
    filtered = filtered.fillna("")
    return jsonify(filtered.to_dict(orient="records"))


@app.route("/exercises/<name>", methods=["GET"])
def get_exercise_by_name(name: str):
    """
    Vrati jednu vježbu prema imenu (approx match).
    Primjer:
        /exercises/squat
    """
    if df.empty:
        return jsonify({"error": "Nema učitanih podataka."}), 500

    subset = df[df["exercise_name"].str.contains(name.lower(), na=False)]
    if subset.empty:
        return jsonify({"error": f"Nije pronađena vježba koja sadrži: {name}"}), 404

    row = subset.iloc[0].to_dict()
    return jsonify(row)


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    if df.empty:
        return jsonify({"error": "Nema učitanih podataka."}), 500

    goal = request.args.get("goal", default="strength")
    data = df.copy()

    if goal == "strength":
        recs = data[
            (data["equipment"].isin(["barbell", "dumbbell", "bodyweight"]))
            & (data["difficulty"].isin(["intermediate", "advanced"]))
        ]
        focus = "Teži višezglobni pokreti (compound lifts) za snagu."

    elif goal == "hypertrophy":
        recs = data[
            (data["equipment"].isin(["dumbbell", "cable", "machine", "smith machine"]))
            & (data["difficulty"].isin(["beginner", "intermediate"]))
        ]
        focus = "Kontrolirani pokreti s većim volumenom za hipertrofiju."

    elif goal == "full_body":
        recs = data[data["difficulty"].isin(["beginner", "intermediate"])]
        focus = "Balansiran izbor vježbi za cijelo tijelo."

    elif goal == "push_pull_legs":
        recs = data
        focus = "Vježbe raspoređene po push/pull/legs skupinama."

    else:
        recs = data
        focus = "Opći popis vježbi (neprepoznat goal, vraćam sve)."
    recs = recs.fillna("")
    result = {
        "goal": goal,
        "focus": focus,
        "count": int(len(recs)),
        "examples": recs[["exercise_name", "target_muscle", "equipment", "difficulty"]]
        .head(10)
        .to_dict(orient="records"),
    }
    return Response(
        json.dumps(result, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8"
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)