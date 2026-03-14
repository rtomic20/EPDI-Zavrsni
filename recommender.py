"""
recommender.py — AI engine: prima upit, pretražuje ChromaDB, vraća preporuke.
Koristi Google Gemini 2.0 Flash kao primarni model.
"""

import os
import warnings
import logging
import chromadb
from groq import Groq
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import random

# Makni sve nepotrebne warningove
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "eduza_tecajevi"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    return _collection


FREE_KEYWORDS = ["besplatno", "besplatan", "besplatna", "free", "gratis", "bez naknade", "0 eur", "0€"]


def _wants_free(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in FREE_KEYWORDS)


def search_courses(query: str, n_results: int = 8) -> list[dict]:
    """
    Pretraži ChromaDB i vrati top N tečajeva.
    Ako upit sadrži 'besplatno/free', filtrira samo besplatne tečajeve.
    Bias mitigation: diversifikacija po kategorijama.
    """
    collection = get_collection()

    where_filter = {"price": "Besplatno"} if _wants_free(query) else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where_filter if where_filter else None,
        )
    except Exception:
        # Fallback bez filtra ako filtar ne uspije (npr. nema besplatnih)
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

    courses = []
    if not results["metadatas"] or not results["metadatas"][0]:
        return courses

    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        courses.append({
            **meta,
            "relevance_score": round(1 - dist, 3),  # cosine sličnost
            "snippet": doc[:300],
        })

    # Bias mitigation: ako ima previše iz iste kategorije, diversificiraj
    courses = _diversify(courses)

    return courses


def _diversify(courses: list[dict], max_per_category: int = 2) -> list[dict]:
    """
    Ne vrati više od max_per_category tečajeva iz iste kategorije.
    Ostatak zamijeni s manjim relevantnim ali iz drugih kategorija.
    """
    seen_categories: dict[str, int] = {}
    result = []
    skipped = []

    for c in courses:
        cat = c.get("category", "ostalo") or "ostalo"
        count = seen_categories.get(cat, 0)
        if count < max_per_category:
            result.append(c)
            seen_categories[cat] = count + 1
        else:
            skipped.append(c)

    # Dodaj neke od preskočenih ako ima mjesta
    random.shuffle(skipped)
    result.extend(skipped[:max(0, 5 - len(result))])

    return result[:5]


SYSTEM_PROMPT = """Ti si prijateljski AI asistent za preporuke edukacija na platformi Eduza.hr.

PRAVILA PONAŠANJA:
- NIKAD ne počinjaj s "Dragi korisniku", "Hvala na povjerenju", "Lijep pozdrav" ili sličnim formalnostima
- Piši PRIRODNO i DIREKTNO, kao da razgovaraš s prijateljem
- Svaki odgovor mora biti DRUGAČIJI — ne kopiraj strukturu prethodnih odgovora
- Budi konkretan: objasni ZAŠTO baš taj tečaj odgovara korisniku
- Ako tečaj nije idealan, reci to iskreno umjesto da hvališ sve jednako
- Kratki uvod (1 rečenica max), odmah na preporuke
- NE završavaj s potpisom, emailom ili standardnim pozdravom
- Odgovaraj na HRVATSKOM jeziku"""


def get_ai_recommendation(user_query: str, courses: list[dict], history: list[dict] | None = None, include_weak: bool = False) -> str:
    if not courses:
        return "Za ovaj upit nisam pronašao odgovarajuće tečajeve u bazi. Pokušaj opisati konkretnije što te zanima ili koji problem želiš riješiti."

    course_list = "\n".join([
        f"- {c['title']} (relevantnost: {c['relevance_score']:.0%}, cijena: {c.get('price','?')})"
        for c in courses
    ])

    weak_note = "\nNAPOMENA: Ovi tečajevi imaju manju semantičku sličnost s upitom (ispod 50%). Naglasi to korisniku i preporuči ih kao alternativne opcije, ne primarne." if include_weak else ""

    user_msg = f"""Korisnikov upit: "{user_query}"

Dostupni tečajevi (pretraživanje vektorske baze):{weak_note}
{course_list}

Preporuči 2-4 najprikladnija tečaja s kratkim objašnjenjem zašto svaki odgovara ovom upitu."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_msg})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.85,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Greška pri komunikaciji s AI modelom: {e}"


def is_off_topic(query: str) -> bool:
    """Provjeri je li upit potpuno nevezan za edukaciju."""
    off_topic_check = f"""Korisnik je postavio sljedeći upit sustavu za preporuke edukacija i tečajeva:
"{query}"

Je li ovaj upit VEZAN uz edukaciju, učenje, razvoj vještina, karijeru ili profesionalni razvoj?
Odgovori SAMO s jednom riječju: DA ili NE."""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": off_topic_check}],
            temperature=0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("NE")
    except Exception:
        return False  # U slučaju greške, propusti upit


RELEVANCE_THRESHOLD = 0.50


def recommend(user_query: str, history: list[dict] | None = None, include_weak: bool = False) -> tuple[str, list[dict], int]:
    """
    Glavna funkcija: prima upit i povijest razgovora.
    Vraća (AI odgovor, lista tečajeva, broj skrivenih slabih tečajeva).
    - include_weak=False → samo tečajevi >= 50% sličnosti
    - include_weak=True  → svi tečajevi, AI naglašava manju relevantnost
    """
    if is_off_topic(user_query):
        return (
            "Mogu pomoći samo s preporukama tečajeva i edukacija s **eduza.hr**. "
            "Opiši što želiš naučiti ili koji problem rješavaš! 😊",
            [],
            0,
        )

    all_courses = search_courses(user_query)
    good  = [c for c in all_courses if c["relevance_score"] >= RELEVANCE_THRESHOLD]
    weak  = [c for c in all_courses if c["relevance_score"] <  RELEVANCE_THRESHOLD]

    if include_weak:
        # Korisnik je tražio više — prikaži slabe, AI ih označava kao alternative
        ai_response = get_ai_recommendation(user_query, weak or all_courses, history, include_weak=True)
        return ai_response, weak or all_courses, 0

    if not good:
        # Nema ničega iznad praga — prikaži sve s napomenom
        ai_response = get_ai_recommendation(user_query, all_courses, history, include_weak=True)
        return ai_response, all_courses, 0

    ai_response = get_ai_recommendation(user_query, good, history, include_weak=False)
    return ai_response, good, len(weak)
