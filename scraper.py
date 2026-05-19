"""
Mijndomein concurrent-prijzen tracker.

Scrapes domain pricing from 7 competitor registrars and appends today's
snapshot to history.json. Runs daily via GitHub Actions.

Best-effort scraper: uses HTTP fetch + BeautifulSoup + regex. Marks prices
as None when it can't extract them with confidence. Iterate on per-registrar
parsers in the EXTRACTORS dict over time.

Usage:
    python scraper.py

Outputs:
    history.json — updated with today's run appended
    scrape-log.txt — what each parser found / failed on
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

HISTORY_PATH = Path(__file__).parent / "history.json"
LOG_PATH = Path(__file__).parent / "scrape-log.txt"

USER_AGENT = "MijndomeinPriceMonitor/1.0 (competitive intelligence; one request per day per page)"

REQUEST_TIMEOUT = 15  # seconds
INTER_REQUEST_DELAY = 1.5  # be polite between requests
HISTORY_MAX_DAYS = 365  # trim to keep file small

EXTENSIONS = [".nl", ".com", ".eu", ".be", ".org", ".ai"]
REGISTRARS = ["TransIP", "Strato", "Vimexx", "Yourhosting", "Hostnet", "GoDaddy", "Shopify"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.5",
})

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def fetch(url: str) -> Optional[str]:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            log(f"  HTTP {r.status_code} for {url}")
            return None
        return r.text
    except requests.RequestException as e:
        log(f"  fetch error {url}: {e}")
        return None
    finally:
        time.sleep(INTER_REQUEST_DELAY)


# Match euro prices like "€ 0,49" / "€0.49" / "EUR 12,50" / "12,50 €"
PRICE_RX = re.compile(
    r"(?:€|EUR)\s*(\d{1,3}(?:[.,]\d{2})?)|(\d{1,3}[.,]\d{2})\s*(?:€|EUR)",
    re.IGNORECASE,
)


def parse_price(s: str) -> Optional[float]:
    """Pull a euro amount out of a short string."""
    if not s:
        return None
    m = PRICE_RX.search(s)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
    try:
        return round(float(raw), 2)
    except ValueError:
        return None


def find_prices_near(text: str, anchor: str, window: int = 200) -> list[float]:
    """Find euro prices in text within `window` chars after each occurrence of `anchor`."""
    found = []
    for m in re.finditer(re.escape(anchor), text, re.IGNORECASE):
        chunk = text[m.end():m.end() + window]
        for pm in PRICE_RX.finditer(chunk):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if 0.01 <= v <= 200:  # sanity range for domain prices
                    found.append(v)
            except ValueError:
                pass
    return found


@dataclass
class PriceRow:
    r: str  # registrar
    e: str  # extension
    fy: Optional[float] = None  # first year
    rn: Optional[float] = None  # renewal
    u: Optional[str] = None  # source url


# -----------------------------------------------------------------------------
# Per-registrar extractors
#
# Each function returns a dict { extension: PriceRow }. Improve these over time.
# Conservative principle: return None instead of guessing.
# -----------------------------------------------------------------------------


def scrape_transip() -> dict[str, PriceRow]:
    """TransIP has /domeinnaam/{ext}-domein/ pages for each extension."""
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl", ".com": "com", ".eu": "eu", ".be": "be", ".org": "org", ".ai": "ai"}
    for ext, slug in ext_map.items():
        url = f"https://www.transip.nl/domeinnaam/{slug}-domein/"
        html = fetch(url)
        if not html:
            out[ext] = PriceRow("TransIP", ext, None, None, url)
            continue
        # First-year price is usually in the hero / h1 area
        # Renewal often in a table or fine print near "reguliere prijs" / "verleng"
        prices_first = find_prices_near(html, f"{slug}-domein", 500)
        prices_renew = find_prices_near(html, "verleng", 300) + find_prices_near(html, "regulier", 300)
        fy = min(prices_first) if prices_first else None
        rn = max(prices_renew) if prices_renew else None
        log(f"  TransIP {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("TransIP", ext, fy, rn, url)
    return out


def scrape_strato() -> dict[str, PriceRow]:
    """Strato has /domeinnaam/{ext}-domein-kopen/ for some extensions."""
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl", ".com": "com", ".eu": "eu", ".be": "be", ".org": "org"}
    for ext, slug in ext_map.items():
        url = f"https://www.strato.nl/domeinnaam/{slug}-domein-kopen/"
        html = fetch(url)
        if not html:
            out[ext] = PriceRow("Strato", ext, None, None, url)
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        prices_first = find_prices_near(text, f".{slug}", 200) + find_prices_near(text, f"{slug} domein", 300)
        prices_renew = find_prices_near(text, "vanaf jaar 2", 200) + find_prices_near(text, "jaar 2", 100)
        fy = min(prices_first) if prices_first else None
        rn = max(prices_renew) if prices_renew else None
        log(f"  Strato {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("Strato", ext, fy, rn, url)
    out[".ai"] = PriceRow("Strato", ".ai", None, None, None)  # Strato verkoopt geen .ai
    return out


def scrape_vimexx() -> dict[str, PriceRow]:
    """Vimexx has one pricing table on /domeinnaam."""
    url = "https://www.vimexx.nl/domeinnaam/domeinnaam-extensies"
    out = {e: PriceRow("Vimexx", e, None, None, url) for e in EXTENSIONS}
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    # Vimexx toont prijzen in tabel-rijen per extensie. Pak voor elke ext de prijs uit dezelfde rij.
    for ext in EXTENSIONS:
        # Zoek het eerste element dat de extensie als losse tekst bevat
        node = soup.find(string=re.compile(r"^\s*\\" + re.escape(ext) + r"\s*$", re.IGNORECASE))
        if not node:
            node = soup.find(string=re.compile(re.escape(ext), re.IGNORECASE))
        if not node:
            log(f"  Vimexx {ext}: not found on page")
            continue
        # Klim naar de tabelrij/container en pak prijzen daarin
        parent = node
        for _ in range(5):
            parent = parent.parent if parent and parent.parent else parent
        chunk = parent.get_text(" ", strip=True) if parent else ""
        prices = []
        for pm in PRICE_RX.finditer(chunk):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if 0.01 <= v <= 200:
                    prices.append(v)
            except ValueError:
                pass
        # Bij Vimexx is eerste-jaar = verlenging (vlakke prijs). Pak min als beste schatting.
        fy = rn = min(prices) if prices else None
        log(f"  Vimexx {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("Vimexx", ext, fy, rn, url)
    return out


def scrape_yourhosting() -> dict[str, PriceRow]:
    url = "https://www.yourhosting.nl/domeinnaam-registreren/extensies/"
    out = {e: PriceRow("Yourhosting", e, None, None, url) for e in EXTENSIONS}
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    # Zoek tabelrijen die een extensie bevatten
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        first = cells[0].lower()
        for ext in EXTENSIONS:
            if first.strip().lstrip(".") == ext.lstrip("."):
                prices = []
                for c in cells[1:]:
                    p = parse_price(c)
                    if p is not None:
                        prices.append(p)
                if prices:
                    fy = min(prices)
                    rn = max(prices) if len(prices) > 1 else None
                    log(f"  Yourhosting {ext}: fy={fy} rn={rn}")
                    out[ext] = PriceRow("Yourhosting", ext, fy, rn, url)
                break
    return out


def scrape_hostnet() -> dict[str, PriceRow]:
    url = "https://www.hostnet.nl/prijzen-domeinnamen"
    out = {e: PriceRow("Hostnet", e, None, None, url) for e in EXTENSIONS}
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        first = cells[0].lower().strip()
        for ext in EXTENSIONS:
            if first == ext or first == ext.lstrip("."):
                prices = [p for p in (parse_price(c) for c in cells[1:]) if p is not None]
                if prices:
                    fy = min(prices)
                    rn = max(prices) if len(prices) > 1 else None
                    log(f"  Hostnet {ext}: fy={fy} rn={rn}")
                    out[ext] = PriceRow("Hostnet", ext, fy, rn, url)
                break
    return out


def scrape_godaddy() -> dict[str, PriceRow]:
    """GoDaddy is JS-heavy. We hit hun publieke TLD-pagina's en parsen wat we kunnen."""
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl-domein", ".com": "com-domein", ".eu": "eu-domein",
               ".be": "be-domein", ".org": "org-domein", ".ai": "ai-domein"}
    for ext, slug in ext_map.items():
        url = f"https://www.godaddy.com/nl/tlds/{slug}"
        html = fetch(url)
        if not html:
            out[ext] = PriceRow("GoDaddy", ext, None, None, url)
            continue
        # GoDaddy's prijzen staan in JSON-LD of in zichtbare tekst. Probeer beide.
        prices = []
        for pm in PRICE_RX.finditer(html):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if 0.01 <= v <= 200:
                    prices.append(v)
            except ValueError:
                pass
        fy = min(prices) if prices else None
        rn = max(prices) if len(prices) > 1 else None
        log(f"  GoDaddy {ext}: fy={fy} rn={rn} ({len(prices)} prices on page)")
        out[ext] = PriceRow("GoDaddy", ext, fy, rn, url)
    return out


def scrape_shopify() -> dict[str, PriceRow]:
    """Shopify heeft /domains/{ext} pagina's."""
    out: dict[str, PriceRow] = {}
    ext_map = {".com": "com", ".org": "org"}  # Shopify ondersteunt beperkt aantal TLDs
    for ext in EXTENSIONS:
        slug = ext_map.get(ext)
        if not slug:
            out[ext] = PriceRow("Shopify", ext, None, None, None)
            continue
        url = f"https://www.shopify.com/domains/{slug}"
        html = fetch(url)
        if not html:
            out[ext] = PriceRow("Shopify", ext, None, None, url)
            continue
        prices = []
        for pm in PRICE_RX.finditer(html):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if 5 <= v <= 50:  # Shopify TLD prijzen zitten in deze range
                    prices.append(v)
            except ValueError:
                pass
        # Shopify zijn jaarprijzen meestal vlak (geen aanbieding)
        fy = rn = min(prices) if prices else None
        log(f"  Shopify {ext}: fy={fy}")
        out[ext] = PriceRow("Shopify", ext, fy, rn, url)
    return out


EXTRACTORS: dict[str, Callable[[], dict[str, PriceRow]]] = {
    "TransIP": scrape_transip,
    "Strato": scrape_strato,
    "Vimexx": scrape_vimexx,
    "Yourhosting": scrape_yourhosting,
    "Hostnet": scrape_hostnet,
    "GoDaddy": scrape_godaddy,
    "Shopify": scrape_shopify,
}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def load_history() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return {
        "meta": {
            "your_brand": "Mijndomein",
            "competitors": REGISTRARS,
            "extensions": EXTENSIONS,
            "currency": "EUR",
            "default_incl_vat": True,
            "created": str(date.today()),
        },
        "runs": [],
    }


def save_history(h: dict) -> None:
    HISTORY_PATH.write_text(json.dumps(h, indent=2, ensure_ascii=False))


def main() -> int:
    today = str(date.today())
    log(f"=== Run {today} ===")

    history = load_history()

    # Idempotent: als vandaag al gedraaid, overschrijf die run.
    history["runs"] = [r for r in history["runs"] if r.get("date") != today]

    all_rows: list[PriceRow] = []
    for name, fn in EXTRACTORS.items():
        log(f"\n[{name}]")
        try:
            rows = fn()
            for ext in EXTENSIONS:
                row = rows.get(ext, PriceRow(name, ext, None, None, None))
                all_rows.append(row)
        except Exception as e:
            log(f"  ! {name} crashed: {e}")
            for ext in EXTENSIONS:
                all_rows.append(PriceRow(name, ext, None, None, None))

    run = {
        "date": today,
        "source": "GitHub Actions scraper",
        "prices": [
            {"r": r.r, "e": r.e, "fy": r.fy, "rn": r.rn, "u": r.u} for r in all_rows
        ],
    }
    history["runs"].append(run)

    # Trim
    if len(history["runs"]) > HISTORY_MAX_DAYS:
        history["runs"] = history["runs"][-HISTORY_MAX_DAYS:]

    save_history(history)

    filled = sum(1 for r in all_rows if r.fy is not None or r.rn is not None)
    total = len(all_rows)
    log(f"\nDone. {filled}/{total} cells filled. History now has {len(history['runs'])} run(s).")

    LOG_PATH.write_text("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
