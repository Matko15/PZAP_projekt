# PZAP_projekt – Fitness data analiza

Projekt za kolegij PZAP koji analizira podatke o vježbama iz više izvora (Kaggle dataset + javni API). Integrira ih u jedinstveni skup podataka i generira vizualizacije kao i preporuke za trening.

## Struktura projekta

- `projekt.py` – služi za učitavanje podataka, čišćenje, integracija, analiza i vizualizacije
- `api.py` – služi za rad s API‑jem (po potrebi profesora)
- `data/`
  - `exercises.json` – ulazni skup podataka (Kaggle)
  - `integrated_fitness_data.csv` – integrirani i očišćeni podaci (generira skripta)
- `visualizations/` – generirani grafovi (`.png`)
- `requirements.txt` – popis Python paketa potrebnih za pokretanje

## Kako pokrenuti projekt

1. Klonirati repozitorij i ući u folder:
git clone https://github.com/Matko15/PZAP_projekt.git
cd PZAP_projekt