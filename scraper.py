"""
Mijndomein concurrent-prijzen tracker.

Scraping pipeline:
  1. Plain HTTP fetch + BeautifulSoup → regex/JSON-LD parse
  2. Playwright (headless Chromium) for JS-heavy sites
  3. LLM fallback (Claude Haiku) when above strategies return < threshold of prices

Outputs:
  - history.json updated with today's run appended
  - scrape-log.txt with what each parser found / failed on

Required env (optional but recommended):
  - ANTHROPIC_API_KEY for LLM fallback
  - DISABLE_PLAYWRIGHT=1 to skip browser rendering (faster, less reliable)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
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

# Browser-like UA om CDN/WAF te omzeilen die strikt zijn op bot-UA's.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
)

REQUEST_TIMEOUT = 15
INTER_REQUEST_DELAY = 1.5
HISTORY_MAX_DAYS = 365
LLM_MODEL = "claude-haiku-4-5-20251001"

EXTENSIONS = [".nl", ".com", ".eu", ".be", ".org", ".ai"]
REGISTRARS = ["Mijndomein", "TransIP", "Strato", "Vimexx", "Yourhosting", "Hostnet", "GoDaddy", "Shopify"]

# Sites die zonder JS-render geen prijzen tonen.
USE_PLAYWRIGHT_FOR = {"Strato", "GoDaddy", "Shopify", "Mijndomein"}
DISABLE_PLAYWRIGHT = os.getenv("DISABLE_PLAYWRIGHT") == "1"

# Hoe veel prijzen we minstens verwachten per registrar voor we LLM-fallback inzetten.
LLM_FALLBACK_THRESHOLD = 2  # als < 2 van 6 extensies een prijs hebben → LLM erbij

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.5",
})

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


# -----------------------------------------------------------------------------
# Fetching
# -----------------------------------------------------------------------------


def fetch_http(url: str) -> Optional[str]:
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


_playwright_browser = None


def get_playwright_browser():
    global _playwright_browser
    if _playwright_browser is not None:
        return _playwright_browser
    if DISABLE_PLAYWRIGHT:
        return None
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        _playwright_browser = pw.chromium.launch(headless=True)
        return _playwright_browser
    except Exception as e:
        log(f"  Playwright init failed: {e}")
        return None


def fetch_rendered(url: str) -> Optional[str]:
    browser = get_playwright_browser()
    if not browser:
        return fetch_http(url)
    try:
        context = browser.new_context(user_agent=USER_AGENT, locale="nl-NL")
        page = context.new_page()
        page.goto(url, timeout=25000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        # Probeer cookie-banners weg te klikken (zo blijven prijzen niet verstopt).
        for selector in ['button:has-text("Akkoord")', 'button:has-text("Accepteer")',
                         'button:has-text("Accept all")', 'button:has-text("Toestaan")']:
            try:
                page.locator(selector).first.click(timeout=500)
                break
            except Exception:
                pass
        html = page.content()
        context.close()
        time.sleep(INTER_REQUEST_DELAY)
        return html
    except Exception as e:
        log(f"  Playwright error for {url}: {e}")
        return fetch_http(url)


def fetch(url: str, use_browser: bool = False) -> Optional[str]:
    if use_browser:
        return fetch_rendered(url)
    return fetch_http(url)


# -----------------------------------------------------------------------------
# Price extraction
# -----------------------------------------------------------------------------

PRICE_RX = re.compile(
    r"(?:€|EUR)\s*(\d{1,3}(?:[.,]\d{2})?)|(\d{1,3}[.,]\d{2})\s*(?:€|EUR)",
    re.IGNORECASE,
)


def parse_price(s: str) -> Optional[float]:
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


def find_prices_near(text: str, anchor: str, window: int = 200, min_p: float = 0.01, max_p: float = 200) -> list[float]:
    found = []
    for m in re.finditer(re.escape(anchor), text, re.IGNORECASE):
        chunk = text[m.end():m.end() + window]
        for pm in PRICE_RX.finditer(chunk):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if min_p <= v <= max_p:
                    found.append(v)
            except ValueError:
                pass
    return found


def extract_jsonld_offers(html: str) -> list[dict]:
    """Pull schema.org Offer blocks out of JSON-LD scripts. These usually contain
    clean {price, priceCurrency, ...} which is much more reliable than regex."""
    soup = BeautifulSoup(html, "html.parser")
    offers = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _walk_jsonld(data):
            if isinstance(node, dict) and node.get("@type") in {"Offer", "Product"}:
                offers.append(node)
    return offers


def _walk_jsonld(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_jsonld(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_jsonld(item)


@dataclass
class PriceRow:
    r: str
    e: str
    fy: Optional[float] = None
    rn: Optional[float] = None
    u: Optional[str] = None
    incl_vat: Optional[bool] = None  # None = onbekend, True = incl BTW, False = ex BTW


# -----------------------------------------------------------------------------
# LLM fallback (Claude Haiku)
# -----------------------------------------------------------------------------


def llm_extract(html: str, registrar: str, url: str) -> dict[str, PriceRow]:
    """Vraag Claude Haiku om prijzen uit de pagina te extraheren wanneer onze
    deterministische parsers tekortschieten. Idempotent — geen state."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {}
    try:
        import anthropic
    except ImportError:
        log("  LLM: anthropic package not installed")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)[:18000]  # max ~5k tokens

    prompt = f"""Je krijgt de tekst van een domein-registrar pagina. Extraheer de prijzen voor deze TLD-extensies: {", ".join(EXTENSIONS)}.

Registrar: {registrar}
Pagina: {url}

Voor elke extensie wil ik:
- first_year: aanbiedingsprijs eerste jaar in EUR (incl. BTW indien aangegeven; anders incl. BTW aannemen voor NL/BE)
- renewal: verlengingsprijs jaar 2+ in EUR
- incl_vat: true als prijs incl. BTW, false als ex BTW, null als niet duidelijk

REGELS:
- Alleen EUR prijzen (USD/anders converteren naar null).
- Als je twijfelt: null. Verzin NOOIT prijzen.
- Een "vanaf €X" prijs zonder duidelijke extensie-koppeling: null.
- Range-check: .nl is meestal €0,01–€20. .ai meestal €60–€120. Onrealistische waardes → null.

Geef ALLEEN deze JSON terug, geen uitleg, geen markdown:
{{".nl":{{"first_year":null,"renewal":null,"incl_vat":null}},".com":{{"first_year":null,"renewal":null,"incl_vat":null}},".eu":{{"first_year":null,"renewal":null,"incl_vat":null}},".be":{{"first_year":null,"renewal":null,"incl_vat":null}},".org":{{"first_year":null,"renewal":null,"incl_vat":null}},".ai":{{"first_year":null,"renewal":null,"incl_vat":null}}}}

PAGINA-TEKST:
{text}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        response = msg.content[0].text.strip()
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            log(f"  LLM: no JSON in response (got: {response[:200]})")
            return {}
        data = json.loads(match.group())
        out: dict[str, PriceRow] = {}
        for ext, prices in data.items():
            if ext not in EXTENSIONS:
                continue
            fy = prices.get("first_year")
            rn = prices.get("renewal")
            incl = prices.get("incl_vat")
            out[ext] = PriceRow(registrar, ext, fy, rn, url, incl)
        log(f"  LLM extracted {sum(1 for r in out.values() if r.fy is not None or r.rn is not None)}/6 prices")
        return out
    except Exception as e:
        log(f"  LLM error: {e}")
        return {}


def merge_rows(primary: dict[str, PriceRow], fallback: dict[str, PriceRow]) -> dict[str, PriceRow]:
    """Voor elke extensie: gebruik primary als hij prijzen heeft, anders fallback."""
    out = {}
    for ext in EXTENSIONS:
        p = primary.get(ext)
        f = fallback.get(ext)
        if p and (p.fy is not None or p.rn is not None):
            out[ext] = p
        elif f and (f.fy is not None or f.rn is not None):
            out[ext] = f
        else:
            out[ext] = p or f or PriceRow("?", ext, None, None, None)
    return out


def filled_count(rows: dict[str, PriceRow]) -> int:
    return sum(1 for r in rows.values() if r.fy is not None or r.rn is not None)


# -----------------------------------------------------------------------------
# Per-registrar extractors
# -----------------------------------------------------------------------------


def scrape_mijndomein() -> dict[str, PriceRow]:
    url = "https://www.mijndomein.nl/producten/domeinnaam"
    out = {e: PriceRow("Mijndomein", e, None, None, url, True) for e in EXTENSIONS}
    html = fetch(url, use_browser="Mijndomein" in USE_PLAYWRIGHT_FOR)
    if not html:
        log("  Mijndomein: page not loaded")
        return out
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 1. Tabelrijen
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        first = cells[0].lower().strip().lstrip(".")
        for ext in EXTENSIONS:
            if first == ext.lstrip("."):
                prices = [p for p in (parse_price(c) for c in cells[1:]) if p is not None]
                if prices:
                    fy = min(prices)
                    rn = max(prices) if len(prices) > 1 and max(prices) != fy else None
                    log(f"  Mijndomein {ext}: fy={fy} rn={rn} (table)")
                    out[ext] = PriceRow("Mijndomein", ext, fy, rn, url, True)
                break

    # 2. Proximity fallback
    for ext in EXTENSIONS:
        if out[ext].fy is not None:
            continue
        prices = find_prices_near(text, ext, 250)
        if prices:
            fy = min(prices)
            log(f"  Mijndomein {ext}: fy={fy} (proximity)")
            out[ext] = PriceRow("Mijndomein", ext, fy, None, url, True)

    return out


def scrape_transip() -> dict[str, PriceRow]:
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl", ".com": "com", ".eu": "eu", ".be": "be", ".org": "org", ".ai": "ai"}
    for ext, slug in ext_map.items():
        url = f"https://www.transip.nl/domeinnaam/{slug}-domein/"
        html = fetch(url)
        if not html:
            out[ext] = PriceRow("TransIP", ext, None, None, url, True)
            continue

        # JSON-LD eerst
        offers = extract_jsonld_offers(html)
        fy = rn = None
        for o in offers:
            p = o.get("price") or (o.get("offers", {}).get("price") if isinstance(o.get("offers"), dict) else None)
            if p:
                try:
                    val = float(str(p).replace(",", "."))
                    if fy is None or val < fy:
                        fy = val
                except ValueError:
                    pass

        # Fallback: regex
        if fy is None:
            prices_first = find_prices_near(html, f"{slug}-domein", 500)
            fy = min(prices_first) if prices_first else None
        prices_renew = find_prices_near(html, "verleng", 300) + find_prices_near(html, "regulier", 300)
        rn = max(prices_renew) if prices_renew else None

        log(f"  TransIP {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("TransIP", ext, fy, rn, url, True)
    return out


def scrape_strato() -> dict[str, PriceRow]:
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl", ".com": "com", ".eu": "eu", ".be": "be", ".org": "org"}
    for ext, slug in ext_map.items():
        url = f"https://www.strato.nl/domeinnaam/{slug}-domein-kopen/"
        html = fetch(url, use_browser=True)
        if not html:
            out[ext] = PriceRow("Strato", ext, None, None, url, True)
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        prices_first = find_prices_near(text, f".{slug}", 200) + find_prices_near(text, f"{slug} domein", 300)
        prices_renew = find_prices_near(text, "vanaf jaar 2", 200) + find_prices_near(text, "jaar 2", 100)
        fy = min(prices_first) if prices_first else None
        rn = max(prices_renew) if prices_renew else None
        log(f"  Strato {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("Strato", ext, fy, rn, url, True)
    out[".ai"] = PriceRow("Strato", ".ai", None, None, None, True)
    return out


def scrape_vimexx() -> dict[str, PriceRow]:
    url = "https://www.vimexx.nl/domeinnaam/domeinnaam-extensies"
    out = {e: PriceRow("Vimexx", e, None, None, url, False) for e in EXTENSIONS}  # Vimexx toont vaak ex BTW
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    # Fix: regex correct met escaped punt
    for ext in EXTENSIONS:
        pattern = re.compile(r"^\s*" + re.escape(ext) + r"\s*$", re.IGNORECASE)
        node = soup.find(string=pattern)
        if not node:
            node = soup.find(string=re.compile(re.escape(ext), re.IGNORECASE))
        if not node:
            log(f"  Vimexx {ext}: anchor not found")
            continue
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
        fy = rn = min(prices) if prices else None
        log(f"  Vimexx {ext}: fy={fy} rn={rn}")
        out[ext] = PriceRow("Vimexx", ext, fy, rn, url, False)
    return out


def scrape_yourhosting() -> dict[str, PriceRow]:
    url = "https://www.yourhosting.nl/domeinnaam-registreren/extensies/"
    out = {e: PriceRow("Yourhosting", e, None, None, url, True) for e in EXTENSIONS}
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        first = cells[0].lower()
        for ext in EXTENSIONS:
            if first.strip().lstrip(".") == ext.lstrip("."):
                prices = [p for p in (parse_price(c) for c in cells[1:]) if p is not None]
                if prices:
                    fy = min(prices)
                    rn = max(prices) if len(prices) > 1 and max(prices) != fy else None
                    log(f"  Yourhosting {ext}: fy={fy} rn={rn}")
                    out[ext] = PriceRow("Yourhosting", ext, fy, rn, url, True)
                break
    return out


def scrape_hostnet() -> dict[str, PriceRow]:
    url = "https://www.hostnet.nl/prijzen-domeinnamen"
    out = {e: PriceRow("Hostnet", e, None, None, url, True) for e in EXTENSIONS}
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
                    rn = max(prices) if len(prices) > 1 and max(prices) != fy else None
                    log(f"  Hostnet {ext}: fy={fy} rn={rn}")
                    out[ext] = PriceRow("Hostnet", ext, fy, rn, url, True)
                break
    return out


def scrape_godaddy() -> dict[str, PriceRow]:
    out: dict[str, PriceRow] = {}
    ext_map = {".nl": "nl-domein", ".com": "com-domein", ".eu": "eu-domein",
               ".be": "be-domein", ".org": "org-domein", ".ai": "ai-domein"}
    for ext, slug in ext_map.items():
        url = f"https://www.godaddy.com/nl/tlds/{slug}"
        html = fetch(url, use_browser=True)
        if not html:
            out[ext] = PriceRow("GoDaddy", ext, None, None, url, True)
            continue
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
        rn = max(prices) if len(prices) > 1 and max(prices) != fy else None
        log(f"  GoDaddy {ext}: fy={fy} rn={rn} ({len(prices)} candidates)")
        out[ext] = PriceRow("GoDaddy", ext, fy, rn, url, True)
    return out


def scrape_shopify() -> dict[str, PriceRow]:
    out: dict[str, PriceRow] = {}
    ext_map = {".com": "com", ".org": "org"}
    for ext in EXTENSIONS:
        slug = ext_map.get(ext)
        if not slug:
            out[ext] = PriceRow("Shopify", ext, None, None, None, True)
            continue
        url = f"https://www.shopify.com/domains/{slug}"
        html = fetch(url, use_browser=True)
        if not html:
            out[ext] = PriceRow("Shopify", ext, None, None, url, True)
            continue
        prices = []
        # Shopify shows USD by default. Search both EUR and $.
        usd_rx = re.compile(r"\$\s*(\d{1,3}(?:[.,]\d{2})?)")
        for pm in PRICE_RX.finditer(html):
            raw = pm.group(1) or pm.group(2)
            raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
            try:
                v = round(float(raw), 2)
                if 5 <= v <= 50:
                    prices.append(v)
            except ValueError:
                pass
        # USD prijzen converteren naar EUR (ruwweg)
        for pm in usd_rx.finditer(html):
            try:
                v = float(pm.group(1).replace(",", "."))
                if 5 <= v <= 50:
                    prices.append(round(v * 0.92, 2))
            except ValueError:
                pass
        fy = rn = min(prices) if prices else None
        log(f"  Shopify {ext}: fy={fy}")
        out[ext] = PriceRow("Shopify", ext, fy, rn, url, True)
    return out


EXTRACTORS: dict[str, Callable[[], dict[str, PriceRow]]] = {
    "Mijndomein": scrape_mijndomein,
    "TransIP": scrape_transip,
    "Strato": scrape_strato,
    "Vimexx": scrape_vimexx,
    "Yourhosting": scrape_yourhosting,
    "Hostnet": scrape_hostnet,
    "GoDaddy": scrape_godaddy,
    "Shopify": scrape_shopify,
}


# Voor LLM-fallback hebben we per registrar één URL nodig waar alle prijzen samenkomen.
LLM_FALLBACK_URLS = {
    "Mijndomein": "https://www.mijndomein.nl/producten/domeinnaam",
    "TransIP": "https://www.transip.nl/domeinnaam-kopen/",
    "Strato": "https://www.strato.nl/domeinnaam/kosten-domeinnaam/",
    "Vimexx": "https://www.vimexx.nl/domeinnaam/domeinnaam-extensies",
    "Yourhosting": "https://www.yourhosting.nl/domeinnaam-registreren/extensies/",
    "Hostnet": "https://www.hostnet.nl/prijzen-domeinnamen",
    "GoDaddy": "https://www.godaddy.com/nl/domeinen/goedkope-domeinnamen",
    "Shopify": "https://www.shopify.com/domains",
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
            "created": str(date.today()),
        },
        "runs": [],
    }


def save_history(h: dict) -> None:
    HISTORY_PATH.write_text(json.dumps(h, indent=2, ensure_ascii=False))


def main() -> int:
    today = str(date.today())
    log(f"=== Run {today} ===")
    log(f"Playwright: {'disabled' if DISABLE_PLAYWRIGHT else 'enabled'}")
    log(f"LLM fallback: {'enabled' if os.getenv('ANTHROPIC_API_KEY') else 'disabled (no ANTHROPIC_API_KEY)'}\n")

    history = load_history()
    history["meta"]["competitors"] = REGISTRARS
    history["meta"]["extensions"] = EXTENSIONS

    # Idempotent
    history["runs"] = [r for r in history["runs"] if r.get("date") != today]

    all_rows: list[PriceRow] = []
    for name, fn in EXTRACTORS.items():
        log(f"\n[{name}]")
        try:
            primary = fn()
        except Exception as e:
            log(f"  ! {name} primary scraper crashed: {e}")
            primary = {ext: PriceRow(name, ext, None, None, None) for ext in EXTENSIONS}

        # LLM-fallback als we te weinig hebben
        if filled_count(primary) < LLM_FALLBACK_THRESHOLD:
            log(f"  primary found {filled_count(primary)}/6 prices, trying LLM fallback...")
            url = LLM_FALLBACK_URLS.get(name)
            if url:
                html = fetch(url, use_browser=name in USE_PLAYWRIGHT_FOR)
                if html:
                    fallback = llm_extract(html, name, url)
                    primary = merge_rows(primary, fallback)

        if filled_count(primary) == 0:
            log(f"  ! {name}: 0 prices extracted — parser may need updating")

        for ext in EXTENSIONS:
            all_rows.append(primary.get(ext, PriceRow(name, ext, None, None, None)))

    # Cleanup Playwright
    if _playwright_browser:
        try:
            _playwright_browser.close()
        except Exception:
            pass

    run = {
        "date": today,
        "source": "GitHub Actions scraper (HTTP + Playwright + LLM-fallback)",
        "prices": [
            {"r": r.r, "e": r.e, "fy": r.fy, "rn": r.rn, "u": r.u,
             "incl_vat": r.incl_vat} for r in all_rows
        ],
    }
    history["runs"].append(run)

    if len(history["runs"]) > HISTORY_MAX_DAYS:
        history["runs"] = history["runs"][-HISTORY_MAX_DAYS:]

    save_history(history)

    filled = sum(1 for r in all_rows if r.fy is not None or r.rn is not None)
    total = len(all_rows)
    log(f"\n=== Done. {filled}/{total} cells filled. History: {len(history['runs'])} run(s). ===")

    LOG_PATH.write_text("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
