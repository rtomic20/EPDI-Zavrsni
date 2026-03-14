"""
scraper.py — Skida tecajeve s eduza.hr i sprema ih u tecajevi.json
"""

import sys
import requests
from bs4 import BeautifulSoup
import json
import time
import re

# Forsiraj UTF-8 ispis na Windows terminalu
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://www.eduza.hr"
LIST_URL = "https://www.eduza.hr/edukacije/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_course_links(page: int) -> list[str]:
    """Vraca listu URL-ova tecajeva s jedne stranice liste."""
    if page == 1:
        url = LIST_URL
    else:
        # Probaj WordPress /page/N/ format; ako ne uspije, probaj ?page=N
        url = f"{LIST_URL}page/{page}/"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        # Ako /page/N/ vrati redirect na stranicu 1 ili 404, probaj ?page=N
        if resp.status_code == 404 or (page > 1 and resp.url.rstrip("/") == LIST_URL.rstrip("/")):
            url = f"{LIST_URL}?page={page}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Greska pri dohvacanju stranice {page}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Tecajevi imaju format: /naziv-tecaja/123/
        if re.search(r"/\d+/$", href) and href not in ["/", "/edukacije/"]:
            full = href if href.startswith("http") else BASE_URL + href
            if full not in links:
                links.append(full)

    return links


def scrape_course(url: str) -> dict | None:
    """Skida detalje jednog tecaja."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Greska: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Naziv
    title_el = soup.find("h1") or soup.find("h2")
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        return None

    # Opis — trazi najdulje tekstualno podrucje
    desc_candidates = []
    for tag in ["article", "section", "div"]:
        for el in soup.find_all(tag):
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 150:
                desc_candidates.append(text)
    description = max(desc_candidates, key=len)[:3000] if desc_candidates else ""

    # Kategorija — samo iz breadcrumba, ne iz URL-a
    category = ""
    breadcrumb = soup.select(".breadcrumb a, nav[aria-label='breadcrumb'] a, .breadcrumbs a")
    if len(breadcrumb) > 1:
        cat_text = breadcrumb[-2].get_text(strip=True)
        # Ne koristiti ako je isti kao naslov ili prazno
        if cat_text and cat_text.lower() != title.lower():
            category = cat_text

    # Cijena — trazi EUR iznos ili "besplatno"
    price = ""
    for el in soup.find_all(string=True):
        text = el.strip()
        if not text:
            continue
        tl = text.lower()
        if "besplatno" in tl and len(text) < 30:
            price = "Besplatno"
            break
        if re.search(r'\d+[.,]\d*\s*(eur|€|kn)', tl) and len(text) < 50:
            price = text
            break

    # Trajanje — iskljuci UI tekst poput "odaberite termin"
    skip_words = ["odaberite", "termin", "lokaciju", "prijava", "kontakt", "rezerv"]
    duration = ""
    for el in soup.find_all(string=True):
        text = el.strip()
        if not text or len(text) > 60:
            continue
        tl = text.lower()
        if any(s in tl for s in skip_words):
            continue
        if re.search(r'\d+\s*(sat|dan|tjedn|min|hour|h\b)', tl):
            duration = text
            break

    # Edukator/predavac
    educator = ""
    for el in soup.find_all(["span", "p", "div"]):
        text = el.get_text(strip=True)
        if any(w in text.lower() for w in ["predavac", "edukator", "instruktor", "autor"]) and len(text) < 100:
            educator = text
            break

    return {
        "title": title,
        "url": url,
        "category": category,
        "description": description,
        "price": price,
        "duration": duration,
        "educator": educator,
    }


def run_scraper(max_pages: int = 50, max_courses: int = 9999):
    print("Pokrecem scraper za eduza.hr...\n")

    # Skupi linkove sa svih stranica
    all_links = []
    for page in range(1, max_pages + 1):
        print(f"  Stranica {page}...")
        links = get_course_links(page)
        if not links and page > 1:
            print(f"  Nema vise stranica, zaustavljam na stranici {page}.")
            break
        print(f"     -> {len(links)} linkova")
        all_links.extend(links)
        time.sleep(0.8)

    # Ukloni duplikate
    all_links = list(dict.fromkeys(all_links))
    print(f"\nUkupno {len(all_links)} jedinstvenih tecajeva. Skidanje detalja...\n")

    courses = []
    limit = min(len(all_links), max_courses)
    for i, url in enumerate(all_links[:limit]):
        print(f"  [{i+1}/{limit}] {url[:75]}...")
        course = scrape_course(url)
        if course and course["title"]:
            courses.append(course)
            print(f"     -> {course['title'][:60]}")
        time.sleep(0.5)

    # Spremi u JSON
    with open("tecajevi.json", "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

    print(f"\nUspjesno skinuto {len(courses)} tecajeva -> tecajevi.json")
    return courses


if __name__ == "__main__":
    run_scraper()
