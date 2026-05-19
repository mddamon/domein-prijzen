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

# Realistische prijsbereiken per extensie (EUR/jaar). Buiten bereik → null.
EXT_PRICE_RANGE = {
    ".nl":  (0.01, 25),
    ".com": (0.50, 35),
    ".eu":  (0.50, 35),
    ".be":  (0.50, 35),
    ".org": (1.00, 40),
    ".ai":  (30,   200),
}

# Sites die zonder JS-render geen prijzen tonen.
USE_PLAYWRIGHT_FOR = {"Strato", "GoDaddy", "Shopify", "Mijndomein"}
DISABLE_PLAYWRIGHT = os.getenv("DISABLE_PLAYWRIGHT") == "1"

# Hoe veel prijzen we minstens verwachten per registrar voor we LLM-fallback inzetten.
# 4 betekent: bij <4 van 6 extensies gevuld → ook LLM raadplegen.
LLM_FALLBACK_THRESHOLD = 4


def validate_price(price: Optional[float], ext: str) -> Optional[float]:
    """Range-check: prijzen buiten realistische bereik per extensie → None."""
    if price is None:
        return None
    lo, hi = EXT_PRICE_RANGE.get(ext, (0.01, 200))
    if price < lo or price > hi:
        return None
    return price

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


def screenshot_via_url(url: str, wait_ms: int = 6000) -> Optional[bytes]:
    """Open URL met Playwright, wacht voor JS-render, neem full-page screenshot."""
    browser = get_playwright_browser()
    if not browser:
        return None
    try:
        context = browser.new_context(
            user_agent=USER_AGENT, locale="nl-NL",
            viewport={"width": 1400, "height": 1800}
        )
        page = context.new_page()
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=wait_ms)
        except Exception:
            pass
        for sel in ['button:has-text("Akkoord")', 'button:has-text("Accepteer")',
                    'button:has-text("Accept all")', 'button:has-text("Toestaan")']:
            try:
                page.locator(sel).first.click(timeout=400)
                break
            except Exception:
                pass
        page.wait_for_timeout(2000)
        img = page.screenshot(full_page=True, type="png")
        context.close()
        time.sleep(INTER_REQUEST_DELAY)
        return img
    except Exception as e:
        log(f"    screenshot_via_url error: {e}")
        return None


def screenshot_via_form_checker(
    checker_url: str,
    test_domain: str,
    input_selectors: Optional[list[str]] = None,
    wait_after_submit_ms: int = 6000,
) -> Optional[bytes]:
    """Open checker, typ domein, submit, wacht en screenshot het resultaat."""
    if input_selectors is None:
        input_selectors = [
            "input[placeholder*='omein' i]", "input[placeholder*='omain' i]",
            "input[type='search']", "input[name*='domain' i]",
            "input[name*='omein' i]", "input.search", "input.domain-search",
            "form input[type='text']",
        ]
    browser = get_playwright_browser()
    if not browser:
        return None
    try:
        context = browser.new_context(
            user_agent=USER_AGENT, locale="nl-NL",
            viewport={"width": 1400, "height": 1800}
        )
        page = context.new_page()
        page.goto(checker_url, timeout=25000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass
        for sel in ['button:has-text("Akkoord")', 'button:has-text("Accepteer")',
                    'button:has-text("Accept all")', 'button:has-text("Toestaan")']:
            try:
                page.locator(sel).first.click(timeout=400)
                break
            except Exception:
                pass
        filled = False
        for sel in input_selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(timeout=1500)
                loc.fill(test_domain)
                loc.press("Enter")
                filled = True
                log(f"    screenshot_form: filled '{sel}'")
                break
            except Exception:
                continue
        if not filled:
            log(f"    screenshot_form: no input at {checker_url}")
            context.close()
            return None
        try:
            page.wait_for_load_state("networkidle", timeout=wait_after_submit_ms)
        except Exception:
            pass
        page.wait_for_timeout(3000)
        img = page.screenshot(full_page=True, type="png")
        context.close()
        time.sleep(INTER_REQUEST_DELAY)
        return img
    except Exception as e:
        log(f"    screenshot_form error: {e}")
        return None


def vision_extract_prices(
    image_bytes: bytes,
    registrar: str,
    test_domain: str,
    url: str,
) -> dict[str, "PriceRow"]:
    """Stuur screenshot naar Claude Haiku (vision) om prijzen per extensie te lezen."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log("    vision: no ANTHROPIC_API_KEY")
        return {}
    try:
        import anthropic
        import base64
    except ImportError:
        return {}

    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = f"""Dit is een screenshot van een domeinchecker bij {registrar} na zoekopdracht "{test_domain}".

Lees uit de visueel zichtbare prijzen voor "{test_domain}.{{ext}}" voor deze extensies:
.nl, .com, .eu, .be, .org, .ai

Per extensie:
- first_year: aanbiedingsprijs eerste jaar in EUR (vaak prominent/vetgedrukt)
- renewal: reguliere/verlengingsprijs (vaak doorgestreept of in kleinere tekst)
- incl_vat: true/false/null

Strikte regels:
- Alleen prijzen die DUIDELIJK bij de extensie staan op de schermafbeelding.
- Andere TLDs (.pw, .shop, .store, .online, etc.) negeren.
- USD → null.
- Range-checks: .nl €0,01-25, .com €0,50-35, .eu €0,50-35, .be €0,50-35, .org €1-40, .ai €30-200.
- Bij twijfel: null. Geen prijzen verzinnen.

Antwoord ALLEEN met deze JSON:
{{".nl":{{"first_year":null,"renewal":null,"incl_vat":null}},".com":{{"first_year":null,"renewal":null,"incl_vat":null}},".eu":{{"first_year":null,"renewal":null,"incl_vat":null}},".be":{{"first_year":null,"renewal":null,"incl_vat":null}},".org":{{"first_year":null,"renewal":null,"incl_vat":null}},".ai":{{"first_year":null,"renewal":null,"incl_vat":null}}}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": img_b64
                    }},
                    {"type": "text", "text": prompt}
                ]
            }],
        )
        response = msg.content[0].text.strip()
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            log(f"    vision: no JSON in response: {response[:200]}")
            return {}
        data = json.loads(match.group())
        out: dict[str, PriceRow] = {}
        for ext, prices in data.items():
            if ext not in EXTENSIONS:
                continue
            if not isinstance(prices, dict):
                continue
            fy = validate_price(prices.get("first_year"), ext)
            rn = validate_price(prices.get("renewal"), ext)
            incl = prices.get("incl_vat")
            out[ext] = PriceRow(registrar, ext, fy, rn, url, incl)
        filled = sum(1 for r in out.values() if r.fy is not None or r.rn is not None)
        log(f"    vision: {filled}/6 prices for {registrar}")
        return out
    except Exception as e:
        log(f"    vision error: {e}")
        return {}


def fetch_via_form_checker(
    checker_url: str,
    test_domain: str,
    input_selectors: Optional[list[str]] = None,
    wait_after_submit_ms: int = 5000,
) -> Optional[str]:
    """Open checker page, typ test_domain in zoekveld, druk Enter, wacht op resultaat.
    Geeft de gerenderde HTML terug nadat de resultaten zijn geladen."""
    if input_selectors is None:
        input_selectors = [
            "input[placeholder*='omein' i]",       # 'domein'/'domain' (case-i)
            "input[placeholder*='omain' i]",
            "input[type='search']",
            "input[name*='domain' i]",
            "input[name*='omein' i]",
            "input.search",
            "input.domain-search",
            "form input[type='text']",
        ]
    browser = get_playwright_browser()
    if not browser:
        return None
    try:
        context = browser.new_context(user_agent=USER_AGENT, locale="nl-NL")
        page = context.new_page()
        page.goto(checker_url, timeout=25000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass
        # Cookie banners weg
        for sel in ['button:has-text("Akkoord")', 'button:has-text("Accepteer")',
                    'button:has-text("Accept all")', 'button:has-text("Toestaan")',
                    'button:has-text("Sta toe")']:
            try:
                page.locator(sel).first.click(timeout=400)
                break
            except Exception:
                pass
        # Vind inputveld
        filled = False
        for sel in input_selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(timeout=1500)
                loc.fill(test_domain)
                loc.press("Enter")
                filled = True
                log(f"    form_checker: filled selector '{sel}'")
                break
            except Exception:
                continue
        if not filled:
            log(f"    form_checker: no input found at {checker_url}")
            context.close()
            return None
        try:
            page.wait_for_load_state("networkidle", timeout=wait_after_submit_ms)
        except Exception:
            pass
        page.wait_for_timeout(2000)  # extra wachten op async content
        html = page.content()
        context.close()
        time.sleep(INTER_REQUEST_DELAY)
        return html
    except Exception as e:
        log(f"    form_checker error: {e}")
        return None


def llm_extract_checker_results(
    html: str,
    registrar: str,
    test_domain: str,
    url: str,
) -> dict[str, "PriceRow"]:
    """Stuur checker-resultaat HTML naar Claude Haiku met expliciete prompt
    om prijzen te vinden voor {test_domain}.{ext}."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {}
    try:
        import anthropic
    except ImportError:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)[:18000]

    prompt = f"""Je krijgt het HTML-resultaat van een domeinchecker bij {registrar}.
Iemand heeft de naam "{test_domain}" gezocht; de pagina toont nu prijzen per extensie.

Vind voor ELK van deze extensies de prijs voor "{test_domain}.{{ext}}":
- .nl, .com, .eu, .be, .org, .ai

Per extensie wil ik:
- first_year: aanbiedingsprijs eerste jaar in EUR (vaak vetgedrukt of prominent)
- renewal: reguliere prijs / verlengingsprijs (vaak doorgestreept of in fine print)
- incl_vat: true / false / null

KRITIEKE REGELS:
- Het anker is "{test_domain}.{{ext}}". Als die niet voorkomt in de tekst, gebruik null voor die extensie.
- Verschillende registrars tonen prijzen anders: soms regulier eerst dan actie, soms andersom. Pak de LAAGSTE als first_year, de HOOGSTE als renewal — tenzij duidelijk anders.
- Negeer prijzen voor andere TLDs (bv. .pw, .store, .online, .shop).
- Range-checks: .nl €0,01–€25, .com €0,50–€35, .eu €0,50–€35, .be €0,50–€35, .org €1–€40, .ai €30–€200.
- USD → null.
- Bij twijfel: null. Verzin geen prijzen.

Geef ALLEEN deze JSON, geen uitleg:
{{".nl":{{"first_year":null,"renewal":null,"incl_vat":null}},".com":{{"first_year":null,"renewal":null,"incl_vat":null}},".eu":{{"first_year":null,"renewal":null,"incl_vat":null}},".be":{{"first_year":null,"renewal":null,"incl_vat":null}},".org":{{"first_year":null,"renewal":null,"incl_vat":null}},".ai":{{"first_year":null,"renewal":null,"incl_vat":null}}}}

CHECKER-PAGINA TEKST:
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
            log(f"    LLM-checker no JSON: {response[:200]}")
            return {}
        data = json.loads(match.group())
        out: dict[str, PriceRow] = {}
        for ext, prices in data.items():
            if ext not in EXTENSIONS:
                continue
            if not isinstance(prices, dict):  # LLM kan None of string teruggeven
                continue
            fy = validate_price(prices.get("first_year"), ext)
            rn = validate_price(prices.get("renewal"), ext)
            incl = prices.get("incl_vat")
            out[ext] = PriceRow(registrar, ext, fy, rn, url, incl)
        filled = sum(1 for r in out.values() if r.fy is not None or r.rn is not None)
        log(f"    LLM-checker: {filled}/6 prices for {registrar}")
        return out
    except Exception as e:
        log(f"    LLM-checker error: {e}")
        return {}


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

    prompt = f"""Je krijgt de tekst van een domein-registrar pagina. Extraheer prijzen voor deze TLD-extensies: {", ".join(EXTENSIONS)}.

Registrar: {registrar}
Pagina: {url}

Voor elke extensie: first_year (aanbiedingsprijs jaar 1, EUR), renewal (jaar 2+, EUR), incl_vat (true/false/null).

KRITIEKE REGELS — LEES ZORGVULDIG:

1. Een prijs MAG alleen worden toegekend aan een extensie als die specifiek aan DIE extensie is gekoppeld in de tekst.
   - Voorbeeld GOED: tekst zegt ".nl voor €0,49, .com voor €4,99" → fy(.nl)=0.49, fy(.com)=4.99
   - Voorbeeld FOUT: tekst zegt "Domeinen vanaf €0,01" → fy van ALLE extensies invullen met 0.01. NIET DOEN.
   - Bij twijfel of een prijs echt bij die extensie hoort → null.

2. "Vanaf"-prijzen (zoals "vanaf €0,01", "v.a. €2,95", "starting at €X"):
   - Mag ALLEEN gebruikt worden voor de extensie waar dat "vanaf" letterlijk bij staat.
   - Mag NOOIT gerepliceerd worden over meerdere extensies.

3. Range-checks (waardes buiten range → null):
   - .nl: €0,01–€25
   - .com: €0,50–€35
   - .eu: €0,50–€35
   - .be: €0,50–€35
   - .org: €1,00–€40
   - .ai: €30–€200 (.ai is ALTIJD duur; iets goedkoper dan €30 is fout, gebruik null)

4. USD/andere valuta zonder EUR-equivalent → null.

5. Als een extensie niet voorkomt of geen specifieke prijs heeft → null voor first_year EN renewal.

6. Verzin NOOIT prijzen. Bij twijfel: null.

Geef ALLEEN deze JSON, geen uitleg of markdown:
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
        # Detect het "vanaf-prijs over alle extensies"-antipatroon: als de LLM
        # >= 4 extensies dezelfde first_year geeft, is dat vrijwel zeker fout.
        all_fy = [data.get(e, {}).get("first_year") for e in EXTENSIONS]
        non_null_fy = [v for v in all_fy if v is not None]
        suspect_uniform = (len(non_null_fy) >= 4 and len(set(non_null_fy)) == 1)
        if suspect_uniform:
            log(f"  LLM: detected suspect uniform price {non_null_fy[0]} on {len(non_null_fy)} ext — discarding")
        for ext, prices in data.items():
            if ext not in EXTENSIONS:
                continue
            fy = validate_price(prices.get("first_year"), ext) if not suspect_uniform else None
            rn = validate_price(prices.get("renewal"), ext) if not suspect_uniform else None
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
    """Screenshot van de Mijndomein checker, dan Claude Vision."""
    test_domain = "trackercheck" + str(date.today()).replace("-", "")
    url = f"https://www.mijndomein.nl/shop/check-domeinnaam?domeinnaam={test_domain}"
    out = {e: PriceRow("Mijndomein", e, None, None, url, True) for e in EXTENSIONS}
    img = screenshot_via_url(url)
    if not img:
        log("  Mijndomein: screenshot failed")
        return out
    vision_out = vision_extract_prices(img, "Mijndomein", test_domain, url)
    for ext in EXTENSIONS:
        if ext in vision_out:
            out[ext] = vision_out[ext]
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

        # Range-check (vangt o.a. .ai €4 wegens niet-gerelateerd prijsje op page)
        fy = validate_price(fy, ext)
        rn = validate_price(rn, ext)

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
        raw_min = min(prices) if prices else None
        fy = validate_price(raw_min, ext)
        rn = fy  # Vimexx hanteert vlakke prijzen
        log(f"  Vimexx {ext}: fy={fy} rn={rn} (raw {raw_min})")
        out[ext] = PriceRow("Vimexx", ext, fy, rn, url, False)

    # Anti-uniform check: als 4+ extensies dezelfde fy hebben, is dat waarschijnlijk
    # een "vanaf"-prijs die ten onrechte op iedereen plakte. Wis behalve eerste.
    fy_values = [out[e].fy for e in EXTENSIONS if out[e].fy is not None]
    if len(fy_values) >= 4 and len(set(fy_values)) == 1:
        log(f"  Vimexx: uniform fy={fy_values[0]} detected over {len(fy_values)} ext — clearing all (LLM zal opvullen)")
        for e in EXTENSIONS:
            out[e] = PriceRow("Vimexx", e, None, None, url, False)
    return out


def scrape_yourhosting() -> dict[str, PriceRow]:
    """Form-checker → screenshot → Claude Vision."""
    test_domain = "trackercheck" + str(date.today()).replace("-", "")
    checker_url = "https://www.yourhosting.nl/domeinnaam-registreren/domein-kopen/"
    out = {e: PriceRow("Yourhosting", e, None, None, checker_url, True) for e in EXTENSIONS}
    img = screenshot_via_form_checker(checker_url, test_domain)
    if not img:
        log("  Yourhosting: screenshot failed")
        return out
    vision_out = vision_extract_prices(img, "Yourhosting", test_domain, checker_url)
    for ext in EXTENSIONS:
        if ext in vision_out:
            out[ext] = vision_out[ext]
    return out


def scrape_hostnet() -> dict[str, PriceRow]:
    """Hostnet gebruikt div-based layout met class 'extension-row'. Elke row heeft
    .col-extension (TLD) en .col-price (prijzen). Volgorde in col-price:
    'REGULIER ACTIE' (regulier eerst, actie tweede). Bij slechts één prijs = regulier."""
    url = "https://www.hostnet.nl/prijzen-domeinnamen"
    out = {e: PriceRow("Hostnet", e, None, None, url, False) for e in EXTENSIONS}  # Hostnet toont ex BTW
    html = fetch(url)
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")

    # Strategie 1: native div-layout
    for row in soup.select(".extension-row"):
        ext_el = row.select_one(".col-extension")
        price_el = row.select_one(".col-price")
        if not ext_el or not price_el:
            continue
        ext_text = ext_el.get_text(" ", strip=True).lower().strip()
        for ext in EXTENSIONS:
            if ext_text == ext:
                # Pak alle prijzen in volgorde uit col-price
                price_text = price_el.get_text(" ", strip=True)
                price_matches = []
                for pm in PRICE_RX.finditer(price_text):
                    raw = pm.group(1) or pm.group(2)
                    raw = raw.replace(".", "").replace(",", ".") if raw.count(",") == 1 else raw.replace(",", "")
                    try:
                        v = round(float(raw), 2)
                        price_matches.append(v)
                    except ValueError:
                        pass
                # Ook losse getallen zonder € (komt voor bij Hostnet)
                if not price_matches:
                    for m in re.finditer(r"(\d{1,3}(?:[.,]\d{2}))", price_text):
                        try:
                            raw = m.group(1).replace(",", ".")
                            v = round(float(raw), 2)
                            if 0.01 <= v <= 500:
                                price_matches.append(v)
                        except ValueError:
                            pass
                # Hostnet-volgorde: [regulier, actie]
                rn = price_matches[0] if len(price_matches) >= 1 else None
                fy = price_matches[1] if len(price_matches) >= 2 else None
                # Als er maar 1 prijs is = regulier (geen actie nu lopend)
                fy = validate_price(fy, ext)
                rn = validate_price(rn, ext)
                log(f"  Hostnet {ext}: fy={fy} rn={rn} (col-price='{price_text[:30]}')")
                out[ext] = PriceRow("Hostnet", ext, fy, rn, url, False)
                break
    return out


def scrape_godaddy() -> dict[str, PriceRow]:
    """Form-checker → screenshot → Claude Vision."""
    test_domain = "trackercheck" + str(date.today()).replace("-", "")
    checker_url = "https://www.godaddy.com/nl"
    out = {e: PriceRow("GoDaddy", e, None, None, checker_url, True) for e in EXTENSIONS}
    img = screenshot_via_form_checker(checker_url, test_domain)
    if not img:
        log("  GoDaddy: screenshot failed")
        return out
    vision_out = vision_extract_prices(img, "GoDaddy", test_domain, checker_url)
    for ext in EXTENSIONS:
        if ext in vision_out:
            out[ext] = vision_out[ext]
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
