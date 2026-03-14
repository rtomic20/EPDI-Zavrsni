"""
indexer.py — Učitava tecajevi.json i indeksira u ChromaDB vektorsku bazu.
Pokreni jednom (ili kad se podaci promijene).
"""

import sys
import json
import os
import chromadb

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "eduza_tecajevi"
JSON_FILE = "tecajevi.json"

# Embedding model koji razumije i hrvatski i engleski
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def build_index():
    # Provjeri postoji li JSON
    if not os.path.exists(JSON_FILE):
        print(f"Nema {JSON_FILE}! Prvo pokreni: python scraper.py")
        return

    with open(JSON_FILE, encoding="utf-8") as f:
        courses = json.load(f)

    print(f"Ucitano {len(courses)} tecajeva iz {JSON_FILE}")

    # Postavi ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Embedding funkcija (lokalni model, bez API-ja)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Obriši staru kolekciju ako postoji (za osvježavanje)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Stara baza obrisana, kreiram novu...")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Pripremi podatke za unos
    documents = []
    metadatas = []
    ids = []

    for i, course in enumerate(courses):
        # Tekst koji se indeksira = spoj naslova + opisa (za bolje pretraživanje)
        doc_text = f"{course['title']}\n{course.get('category', '')}\n{course.get('description', '')}"
        documents.append(doc_text[:4000])  # ChromaDB limit

        metadatas.append({
            "title": course["title"][:500],
            "url": course["url"][:500],
            "category": course.get("category", "")[:200],
            "price": course.get("price", "")[:100],
            "duration": course.get("duration", "")[:100],
        })
        ids.append(f"course_{i}")

    # Unesi u ChromaDB (u batchevima od 50)
    batch_size = 50
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"  Indeksiram {start+1}-{end} od {total}...")
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"\nIndeksiranje zavrseno! {total} tecajeva u ChromaDB ({CHROMA_PATH}/)")


if __name__ == "__main__":
    build_index()
