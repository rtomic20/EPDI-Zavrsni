# Eduza AI Asistent — Tema 7

AI asistent za preporuke edukacija s platforme eduza.hr.

## Pokretanje (redoslijed!)

```bash
# 1. Instaliraj pakete (jednom)
pip install -r requirements.txt

# 2. Skini tečajeve s eduza.hr
python scraper.py

# 3. Indeksiraj u ChromaDB
python indexer.py

# 4. Pokreni aplikaciju
streamlit run app.py
```

## Struktura

```
EPDIProjekt/
├── .env              ← API ključ (nije u gitu)
├── .gitignore
├── requirements.txt
├── scraper.py        ← Skida tečajeve s eduza.hr
├── indexer.py        ← Gradi vektorsku bazu
├── recommender.py    ← AI engine (Gemini + ChromaDB)
├── app.py            ← Streamlit UI
├── tecajevi.json     ← Skinuti tečajevi (auto)
└── chroma_db/        ← Lokalna vektorska baza (auto)
```
