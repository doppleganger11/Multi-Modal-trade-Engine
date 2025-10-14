# groww_playwright_scrape.py
import csv, re, os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
import argparse
import pytz
from playwright.sync_api import sync_playwright
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

# Model choice: cheap+accurate for this use case
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
_openai_client = None

import os
os.environ.setdefault("OPENAI_API_KEY", "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")

print("[DEBUG] OPENAI_API_KEY present:", bool(os.environ.get("OPENAI_API_KEY")))

# === OpenAI LLM Sentiment Scorer (Chat Completions version) ===
import json, time, re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# Use a chat-completions-capable model that exists in older SDKs too
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_openai_client = None


import html, unicodedata, difflib, string

TIME_TOKENS_RE = re.compile(
    r'\b(?:\d+\s*(?:min(?:\(s\))?|mins?|minutes?|hr(?:\(s\))?|hrs?|hours?|day|days|week|weeks)\s*ago|today|yesterday)\b',
    re.IGNORECASE
)


def log(msg: str): print(f"[SENTI] {msg}")

import sys
print("[BOOT]", __file__, "| python:", sys.executable)


def _fix_mojibake(s: str) -> str:
    # Fix common ’ issues coming from UTF-8/Windows-1252 mix
    return (s or "").replace("â€™", "'").replace("â€œ", '"').replace("â€\x9d", '"').replace("â€“", "-").replace("â€”", "-")

def normalize_headline(text: str) -> str:
    """Lowercase, strip sources/time tokens/punct, collapse spaces."""
    t = html.unescape(_fix_mojibake(text or ""))
    t = unicodedata.normalize("NFKC", t)
    t = t.lower()

    # Remove source prefixes like "cnbc tv18 ·", "reuters -", etc.
    t = re.sub(r'^\s*[a-z0-9 .&+-]{2,}\s[·\-:]\s+', '', t)

    # Drop relative-time tokens (today / X hrs ago / yesterday)
    t = TIME_TOKENS_RE.sub(" ", t)

    # Keep only letters/digits/space
    allowed = set(string.ascii_lowercase + string.digits + " ")
    t = "".join(ch if ch in allowed else " " for ch in t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def deduplicate_items_by_headline(items, similar_threshold: float = 0.90):
    """
    Per firm: keep the most recent item; drop others whose normalized headline
    is ~the same (SequenceMatcher ratio >= threshold).
    """
    # Group by firm
    by_firm = {}
    for it in items:
        by_firm.setdefault(it["firm_name"], []).append(it)

    out = []
    for firm, group in by_firm.items():
        # Sort most recent first (fallback to scraped_at if published_at missing)
        def _dt(it):
            return it.get("published_at") or it.get("scraped_at") or ""
        group_sorted = sorted(group, key=_dt, reverse=True)

        kept_norms = []  # store normalized strings for kept
        kept = []

        for it in group_sorted:
            norm = normalize_headline(it.get("headline", ""))
            if not norm:
                kept.append(it)
                kept_norms.append(norm)
                continue

            # Compare with already-kept headlines for this firm
            is_dup = False
            for kn in kept_norms:
                # quick set overlap guard to short-circuit obvious non-dups
                # (avoids computing ratio for very different strings)
                if kn and norm:
                    # token overlap check
                    a = set(norm.split())
                    b = set(kn.split())
                    if len(a.intersection(b)) == 0:
                        continue
                # fuzzy ratio
                if difflib.SequenceMatcher(None, norm, kn).ratio() >= similar_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(it)
                kept_norms.append(norm)

        out.extend(kept)

    return out


def _get_client():
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it via setx / PyCharm env vars or inline fallback.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

# Remove posting-time noise like "· 2 hrs ago"
_TIME_JUNK_RE = re.compile(
    r'^\s*[^·]{2,}\s[·.]\s*(?:\d+\s*(?:min|mins?|minutes?|hr|hrs?|hours?|days?)\s*ago|today|yesterday)\s+',
    re.I
)
def _sanitize_headline_for_llm(s: str) -> str:
    if not s: return ""
    cleaned = _TIME_JUNK_RE.sub("", s).strip()
    return cleaned if len(cleaned) >= 5 else s.strip()

def _chunk(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

# robust JSON extraction (handles ```json code fences or plain JSON)
def _extract_json(text: str):
    if not text:
        return {}
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to pull from a ```json ... ``` fenced block
    m = re.search(r"```json\s*([\s\S]+?)```", text, re.I)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # try to pull the first {...} object
    m2 = re.search(r"(\{[\s\S]+)", text)
    if m2:
        candidate = m2.group(1).strip()
        # attempt to trim trailing text after last closing brace
        last = candidate.rfind("}")
        if last != -1:
            candidate = candidate[:last+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return {}

SYSTEM_PROMPT = (
    "You are a careful finance sentiment rater. "
    "For each headline, return exactly one real-valued score in [-5,5]: "
    "-5 is very negative for the firm, +5 very positive, 0 neutral/ambiguous. "
    "Ignore posting-time phrases like '2 hrs ago' or '1 day ago'. "
    "If the headline is not about the firm, or sentiment is unclear, return 0. "
    "Output STRICT JSON ONLY with the schema: "
    '{"results":[{"firm":string,"headline":string,"score":number,"reason":string}...]}'
)

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=0.8, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _score_chunk_with_openai(pairs):
    """
    pairs: list of dicts: {"firm": str, "headline": str}
    returns: list[{"firm","headline","score","reason"}]
    """
    client = _get_client()

    # compact numbered list to minimize tokens / increase reliability
    lines = [f"{i+1}. [{p['firm']}] {_sanitize_headline_for_llm(p['headline'])}" for i,p in enumerate(pairs)]
    user_msg = (
        "Score each line independently and return STRICT JSON only (no extra text). "
        "Schema: {\"results\":[{\"firm\":string,\"headline\":string,\"score\":number,\"reason\":string}]}\n\n"
        "Headlines:\n" + "\n".join(lines)
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    content = resp.choices[0].message.content or ""
    data = _extract_json(content)
    out = data.get("results", [])

    # Align with inputs and clip scores
    fixed = []
    for i, p in enumerate(pairs):
        try:
            r = out[i] if i < len(out) else {}
            firm = r.get("firm", p["firm"])
            headline = r.get("headline", p["headline"])
            score = float(r.get("score", 0.0))
            reason = r.get("reason", "")
        except Exception:
            firm = p["firm"]; headline = p["headline"]; score = 0.0; reason = ""
        score = max(-5.0, min(5.0, score))
        fixed.append({"firm": firm, "headline": headline, "score": score, "reason": reason})
    return fixed

# near your OpenAI scorer
import hashlib, json
from pathlib import Path

CACHE_PATH = Path("data/llm_score_cache.json")
try:
    _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
except Exception:
    _cache = {}

def _key(pair):
    s = f"{pair['firm']}||{pair['headline']}".strip().lower()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

import os, json, math, time, re
from typing import List
from openai import OpenAI

def _parse_scores(text: str, expected: int) -> List[float]:
    """
    Try to parse a JSON array of numbers. If that fails, extract numbers with regex.
    Always clamp to [-5, 5] and pad/truncate to expected length.
    """
    nums: List[float] = []
    # 1) try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            nums = [float(x) for x in obj]
    except Exception:
        pass
    # 2) fallback: regex for numbers like -4.5, 3, 0, 2.0
    if not nums:
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    # sanitize
    nums = [max(-5.0, min(5.0, float(x))) for x in nums]
    # align length
    if len(nums) < expected:
        nums += [0.0] * (expected - len(nums))
    elif len(nums) > expected:
        nums = nums[:expected]
    return nums

# ---------- OpenAI scorer (batched, logged, robust) ----------
import os, time, json
from typing import List
from openai import OpenAI

def score_headlines_openai(
    headlines: List[str],
    batch_size: int = 40,
    model: str | None = None,
    max_retries: int = 3,
    per_call_sleep: float = 0.4,  # gentle pacing to avoid rate limits
) -> List[float]:
    """
    Returns a list of sentiment scores in [-5, +5] for each headline using OpenAI.
    OpenAI-only path (no fallbacks).
    """
    # model selection + API key
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY missing in this shell. Set it, then re-run.")
    client = OpenAI(api_key=api_key)

    def _prompt_for(h: str) -> str:
        # keep the system tight to minimize cost and variance
        return (
            "You are a financial news sentiment rater.\n"
            "Rate the headline on trading sentiment in a single number in the range [-5,5], "
            "where -5 = very negative for the stock price, 0 = neutral, +5 = very positive. "
            "Use only the number as the entire response (no text, no explanation, no JSON).\n\n"
            f"Headline: {h.strip()}\n"
        )

    scores: List[float] = []
    i = 0
    # logging
    try:
        log  # will exist if you added the helper; if not, define a no-op
    except NameError:
        def log(msg: str):
            print(f"[SENTI] {msg}")

    log(f"OpenAI scorer active | model={model}")
    while i < len(headlines):
        batch = headlines[i : i + batch_size]
        log(f"OpenAI scoring batch i={i}..{i+len(batch)-1} size={len(batch)}")

        for h in batch:
            # retry loop per headline
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.responses.create(
                        model=model,
                        input=_prompt_for(h),
                        timeout=30,
                    )
                    txt = (resp.output_text or "").strip()
                    # sometimes model returns e.g. "4." or "4.0\n"
                    # keep only the first token that looks like a number
                    num_str = txt.split()[0]
                    try:
                        val = float(num_str)
                    except Exception:
                        # very defensive fallback—try to extract a number
                        import re as _re
                        m = _re.search(r"-?\d+(\.\d+)?", txt)
                        val = float(m.group(0)) if m else 0.0
                    # clip and append
                    if val < -5: val = -5.0
                    if val >  5: val =  5.0
                    scores.append(val)
                    break  # success -> exit retry loop
                except Exception as e:
                    last_err = e
                    time.sleep(1.2 * attempt)  # backoff
            else:
                # all retries failed
                log(f"OpenAI scoring FAILED for one headline, using 0. err={last_err}")
                scores.append(0.0)

            time.sleep(per_call_sleep)
        i += batch_size

    return scores




TZ = pytz.timezone("Asia/Kolkata")

NEWS_SELECTOR = (
    "div[class^='stockNews_newsRow__'], "
    "div[class*='stockNews_newsRow__'], "
    "section[class*='news'] article"
)

# hours/mins/today (primary)
REL_HOURS_TODAY_RE = re.compile(
    r'\b(?:\d+\s*(?:min(?:\(s\))?|mins?|minutes?|hr(?:\(s\))?|hrs?|hours?)\s*ago|today)\b',
    re.IGNORECASE
)
# exactly "yesterday" or "1 day ago" (fallback)
REL_YDAY_RE = re.compile(
    r'\b(?:yesterday|1\s*day(?:s)?\s*ago)\b',
    re.IGNORECASE
)


REL_TODAY_ONLY_RE = re.compile(
    r'\b(?:\d+\s*(?:min(?:\(s\))?|mins?|minutes?|hr(?:\(s\))?|hrs?|hours?)\s*ago|today)\b',
    re.IGNORECASE
)
REL_ANY_RE = re.compile(
    r'\b(?:\d+\s*(?:min(?:\(s\))?|mins?|minutes?|hr(?:\(s\))?|hrs?|hours?|days?|weeks?)\s*ago|yesterday|today)\b',
    re.IGNORECASE
)
TIME_NODE_SEL = "small, time, span, div[class*='bodyExtraSmall'], div[class*='caption'], div[class*='bodySmall']"

def extract_time_token_from_meta(row):
    """
    Look only in meta-ish nodes for time text; ignore title content.
    Returns the matched token (e.g., '19 hr(s) ago', '45 mins ago', 'today') or None.
    """
    # Check nodes that typically hold the 'source · 18 hrs ago' snippet
    nodes = row.query_selector_all(TIME_NODE_SEL)
    for n in nodes:
        txt = " ".join((n.inner_text() or "").split())
        if not txt:
            continue
        # require either 'ago' OR an isolated 'today'/'yesterday'
        if "ago" in txt.lower() or re.search(r"\b(today|yesterday)\b", txt, flags=re.I):
            m = REL_TODAY_ONLY_RE.search(txt)
            if m:
                return m.group(0)
            # not today-only, but keep for logging
            if REL_ANY_RE.search(txt):
                return None
    return None

def rel_to_dt_today(rel: str):
    now = datetime.now(TZ)
    s = (rel or "").strip().lower()
    if "today" in s:
        return now
    m = re.search(r'(\d+)\s*(?:min(?:\(s\))?|mins?|minutes?)\s*ago', s)
    if m: return now - timedelta(minutes=int(m.group(1)))
    m = re.search(r'(\d+)\s*(?:hr(?:\(s\))?|hrs?|hours?)\s*ago', s)
    if m: return now - timedelta(hours=int(m.group(1)))
    return None

def meta_text(row):
    """Concatenate text from meta-ish nodes only (no headline)."""
    return " ".join(
        " ".join(((n.inner_text() or "")).split())
        for n in row.query_selector_all(TIME_NODE_SEL)
    ).strip()

def dt_from_hours_token(rel: str):
    """Return dt for X min/hr ago or 'today'. No calendar-day check."""
    now = datetime.now(TZ)
    s = (rel or "").strip().lower()
    if "today" in s:
        return now
    m = re.search(r'(\d+)\s*(?:min(?:\(s\))?|mins?|minutes?)\s*ago', s)
    if m:
        return now - timedelta(minutes=int(m.group(1)))
    m = re.search(r'(\d+)\s*(?:hr(?:\(s\))?|hrs?|hours?)\s*ago', s)
    if m:
        return now - timedelta(hours=int(m.group(1)))
    return None

def dt_from_yday_token(rel: str):
    """Return dt for 'yesterday' / '1 day ago'."""
    now = datetime.now(TZ)
    return now - timedelta(days=1)

def stock_base(url: str) -> str:
    p = urlparse(url); segs = [s for s in p.path.split("/") if s]
    if segs and segs[-1] in {"market-news", "news"}: segs = segs[:-1]
    new_path = "/" + "/".join(segs) + "/"
    return urlunparse((p.scheme, p.netloc, new_path, "", "", ""))

def clean_headline(text: str) -> str:
    return re.sub(
        r'^\s*[^·]{2,}\s[·.]\s*(?:\d+\s*(?:min(?:\(s\))?|mins?|minutes?|hr(?:\(s\))?|hrs?|hours?)\s*ago|today|yesterday)\s+',
        '',
        text or '',
        flags=re.IGNORECASE,
    ).strip()

# --- headline sanitizer: strip source/time fragments from titles ---
TIME_TOK = r'(?:\d{1,2}\s*(?:mins?|minutes?|hrs?|hours?|day|days)\s*ago|yesterday|today)'
LEAD_SOURCE_RE = re.compile(r'^\s*[A-Za-z][\w& ]{2,40}\s*[·•\-\|]\s*')  # e.g., 'Mint ·'
TRAIL_TIME_RE  = re.compile(rf'\s*[·•\-\|]\s*{TIME_TOK}\s*$', re.I)
PARENS_TIME_RE = re.compile(rf'\((?:\s*{TIME_TOK}\s*)\)', re.I)
TIME_RE        = re.compile(rf'\b{TIME_TOK}\b', re.I)

def sanitize_headline(h: str) -> str:
    t = (h or "").strip()
    t = LEAD_SOURCE_RE.sub("", t)
    t = TRAIL_TIME_RE.sub("", t)
    t = PARENS_TIME_RE.sub("", t)
    t = TIME_RE.sub("", t)
    t = re.sub(r'\s{2,}', ' ', t).strip(" -·•|")
    return t

def news_variants(url: str):
    base = stock_base(url); seen = set()
    for u in (base, urljoin(base, "market-news")):
        if u not in seen:
            seen.add(u); yield u

def read_firms(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("firm_name") or "").strip()
            url = (row.get("url") or "").strip()
            if name and url: yield name, url

def atomic_write_rows(rows, out_csv: Path) -> Path:
    """Write rows atomically. Returns the final path written (out_csv or fallback)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")

    # write to a temp file first
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_name","headline","article_url","rel_time_raw","published_at","scraped_at","source_url"])
        for it in rows:
            w.writerow([
                it["firm_name"], it["headline"], it["article_url"], it["rel_time_raw"],
                it["published_at"], it["scraped_at"], it["source_url"]
            ])

    # atomically replace; if locked, fall back to a timestamped file
    try:
        os.replace(tmp, out_csv)
        final_path = out_csv
        print(f"[OK] Wrote {len(rows)} rows to {out_csv}")
    except PermissionError:
        ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        alt = out_csv.with_name(f"{out_csv.stem}_{ts}{out_csv.suffix}")
        os.replace(tmp, alt)
        final_path = alt
        print(f"[WARN] {out_csv} was locked. Wrote fallback file: {alt}")

    return final_path



def scrape_once(firms_csv: Path, out_csv: Path, headless: bool = True):
    if not firms_csv.exists():
        raise FileNotFoundError(f"firms.csv not found at {firms_csv}")
    items = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ))
        page = context.new_page()

        for firm, url in read_firms(firms_csv):
            firm_hours = []  # primary (mins/hrs/today)
            firm_yday = []  # fallback (yesterday/1 day ago)

            for candidate in news_variants(url):
                resp = page.goto(candidate, wait_until="domcontentloaded", timeout=30000)
                status = resp.status if resp else None
                print(f"[visit] {candidate} -> HTTP {status}")
                if status and status >= 400:
                    continue

                # light scroll to trigger lazy-loading
                for _ in range(6):
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(300)

                try:
                    page.wait_for_selector(NEWS_SELECTOR, timeout=6000)
                except Exception:
                    continue

                row_handles = page.query_selector_all(NEWS_SELECTOR)
                seen = kept_any = old = no_time = 0

                for row in row_handles:
                    seen += 1

                    # headline + link
                    a = row.query_selector("a")
                    if a:
                        headline = " ".join((a.inner_text() or "").split())
                        href = a.get_attribute("href")
                    else:
                        headline = " ".join((row.inner_text() or "").split())
                        href = None
                    if not headline:
                        continue

                    # OPTIONAL: clean headline to drop "Source · X ago" prefixes
                    headline = clean_headline(headline)
                    headline = sanitize_headline(headline)

                    mtxt = meta_text(row)
                    if not mtxt:
                        no_time += 1
                        continue

                    # priority: hours/mins/today
                    m1 = REL_HOURS_TODAY_RE.search(mtxt)
                    if m1:
                        rel = m1.group(0)
                        dt = dt_from_hours_token(rel)
                        if not dt:
                            old += 1
                            continue
                        # construct URL
                        if not href:
                            article_url = candidate
                        else:
                            if "://" in href:
                                article_url = href
                            elif href.startswith("/"):
                                article_url = urljoin(candidate, href)
                            elif href.startswith("#"):
                                article_url = candidate
                            else:
                                article_url = urljoin(candidate, href)
                        firm_hours.append({
                            "firm_name": firm,
                            "headline": headline,
                            "article_url": article_url,
                            "rel_time_raw": rel,
                            "published_at": dt.isoformat(),
                            "scraped_at": datetime.now(TZ).isoformat(),
                            "source_url": candidate,
                        })
                        kept_any += 1
                        continue

                    # fallback: yesterday / 1 day ago
                    m2 = REL_YDAY_RE.search(mtxt)
                    if m2:
                        rel = m2.group(0)
                        dt = dt_from_yday_token(rel)
                        if not href:
                            article_url = candidate
                        else:
                            if "://" in href:
                                article_url = href
                            elif href.startswith("/"):
                                article_url = urljoin(candidate, href)
                            elif href.startswith("#"):
                                article_url = candidate
                            else:
                                article_url = urljoin(candidate, href)
                        firm_yday.append({
                            "firm_name": firm,
                            "headline": headline,
                            "article_url": article_url,
                            "rel_time_raw": rel,
                            "published_at": dt.isoformat(),
                            "scraped_at": datetime.now(TZ).isoformat(),
                            "source_url": candidate,
                        })
                        kept_any += 1
                        continue

                    # stats: if meta had some other time (2 days/week), count as old
                    if REL_ANY_RE.search(mtxt):
                        old += 1
                    else:
                        no_time += 1

                print(f"[page] rows seen={seen}, kept_any={kept_any}, old={old}, no_time={no_time} @ {candidate}")

            # decide what to keep for this firm
            chosen = firm_hours if firm_hours else firm_yday
            print(f"[firm] {firm}: hours={len(firm_hours)} yday={len(firm_yday)} -> kept={len(chosen)}")
            items.extend(chosen)

        context.close();
        browser.close()

    # de-dupe
    dedup = {}
    for it in items:
        key = (it["firm_name"].strip(), it["headline"].strip())
        if key not in dedup:
            dedup[key] = it

    # optional: deterministic order
    rows = sorted(dedup.values(), key=lambda x: (x["firm_name"].lower(), x["headline"].lower()))
    news_written = atomic_write_rows(rows, out_csv)

    # 2) score + email using the actual file
    scores_csv = news_written.parent / "scores_today.csv"
    try:
        df_scores = compute_scores_and_write(Path(firms_csv), news_written, scores_csv)
        send_scores_email(df_scores, scores_csv)
    except Exception as e:
        print("[WARN] scoring/email failed:", e)


# --- RULES: housekeeping neutralizer & expansion/growth nudge ---
HOUSEKEEPING_RE = re.compile(
    r'\b('
    r'trading window|board (?:meeting|to consider)|results (?:on|date)|'
    r'AGM|annual general meeting|record date|meeting on|'
    r'dividend (?:record )?date|earnings call|conference call|'
    r'investor presentation|intimation|notice of|press release'
    r')\b',
    re.IGNORECASE
)

EXPANSION_RE = re.compile(
    r'\b('
    r'joint venture|partnership|partners?|mou|invests?|investment|'
    r'launch(?:es|ed)?|expand(?:s|ed|ing)?|expansion|opens|commission(?:s|ed)?|'
    r'hires|to employ|plants?|facility|capacity|capex|'
    r'wins order|bags order|secures (?:order|deal|contract)|'
    r'raises|funding|acquire(?:s|d)?|stake|merger'
    r')\b',
    re.IGNORECASE
)

GROWTH_RE = re.compile(
    r'\b('
    r'growth|improv(?:e|ed|es|ing)|strong|robust|record|'
    r'gains?|market share|surge|accelerat(?:e|ed|es|ing)'
    r')\b',
    re.IGNORECASE
)

# Amount / invest cues
AMOUNT_RE = re.compile(r'(₹|rs\.?|inr|\$|usd|eur|£)?\s?\d{1,3}(?:[,\d]{3})*(?:\s?(?:cr|crore|mn|million|bn|billion))', re.I)
INVEST_VERB_RE = re.compile(r'\b(invests?|investment|to invest|capex|set up|build|commission(?:s|ed)?|facility|plant)\b', re.I)

# Banking growth cues
GROWTH_BANK_RE = re.compile(r'\b(improved spending growth|strong growth|spending trends?|market share gains?)\b', re.I)

# Ownership transfer (often over-scored if no amount)
OWNERSHIP_XFER_RE = re.compile(r'\b(fully owns|stake transfer|transferred its stake|now owns)\b', re.I)


# Indirect / mandate-style positives (banks as advisors, bookrunners)
SECONDARY_FIN_SERV_RE = re.compile(
    r'\b(lead (?:manager|banker|arranger)|bookrunner|joint bookrunner|advisor|adviser|'
    r'mandate(?:d)?|managing the issue)\b', re.I)

# Technical momentum cues
TECH_MOMENTUM_RE = re.compile(
    r'\b(swing high|breakout|bullish momentum|52[-\s]?week high|multi[-\s]?year high)\b', re.I)

# Partner/elite program cues (prestige, not direct revenue)
PRESTIGE_PARTNER_RE = re.compile(
    r'\b(inner circle|top\s*\d+% partner|elite partner|strategic partner program)\b', re.I)


def is_housekeeping(title: str) -> bool:
    return bool(HOUSEKEEPING_RE.search(title or ""))

def apply_expansion_nudge(title: str, s: float) -> float:
    t = title or ""
    has_amount = bool(AMOUNT_RE.search(t))
    has_invest = bool(INVEST_VERB_RE.search(t))
    has_expand = bool(EXPANSION_RE.search(t))
    has_growth = bool(GROWTH_RE.search(t))
    owns_xfer  = bool(OWNERSHIP_XFER_RE.search(t))

    # 1) Clear expansion/invest/growth → ensure positive
    if has_expand or has_invest or has_growth:
        s = max(s, -0.3)
        base_min = 2.0 if (has_amount or has_invest) else 1.5
        if s < base_min: s = base_min

    # 2) Ownership transfer (no amount) → mild+ but capped
    if owns_xfer and not has_amount:
        s = max(s, 1.5)
        s = min(s, 3.0)

    # 3) Banking/advisory mandates → cap enthusiasm (indirect revenue)
    if SECONDARY_FIN_SERV_RE.search(t):
        s = max(s, 1.5)   # still positive
        s = min(s, 3.0)   # avoid 4–5 spikes

    # 4) Technical momentum (chart signals) → allow strong but cap at 4
    if TECH_MOMENTUM_RE.search(t):
        s = max(s, 2.5)
        s = min(s, 4.0)

    # 5) Prestige partner programs (no $ value) → cap at ~3.5
    if PRESTIGE_PARTNER_RE.search(t) and not has_amount:
        s = max(s, 2.0)
        s = min(s, 3.5)

    return round(max(-4.0, min(4.0, s)), 3)


def apply_industry_specific_nudges(title: str, s: float) -> float:
    t = title or ""
    # Banking: growth / share gains should not be negative
    if GROWTH_BANK_RE.search(t):
        s = max(s, 1.5)
    return round(s, 3)


import pandas as pd


# ---------- Compute per-firm scores & write CSV ----------
def compute_scores_and_write(firms_csv: Path, news_csv: Path, out_scores_csv: Path) -> pd.DataFrame:
    """
    OpenAI-only scoring path.
    Reads today's news CSV, gets per-headline scores via OpenAI,
    averages per firm, fills 0 for firms with no news, writes scores_today.csv, returns DataFrame.
    Also writes data/last_scorer.txt as a breadcrumb.
    """
    # tiny logger
    try:
        log
    except NameError:
        def log(msg: str):
            print(f"[SENTI] {msg}")

    # Load firm universe
    firms_df = pd.read_csv(firms_csv, usecols=["firm_name"])
    firms_df["firm_name"] = firms_df["firm_name"].astype(str)

    if not news_csv.exists():
        out = firms_df.copy()
        out["score"] = 0.0
        atomic_write_rows_df(out, out_scores_csv)
        # marker
        from datetime import datetime
        stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        marker = out_scores_csv.parent / "last_scorer.txt"
        with open(marker, "w", encoding="utf-8") as f:
            f.write(
                "scorer=OpenAI\n"
                f"model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}\n"
                "headlines_scored=0\n"
                f"written={out_scores_csv.name}\n"
                f"time={stamp}\n"
            )
        log(f"Wrote marker: {marker}")
        log(f"[OK] scored {len(out)} firms -> {out_scores_csv}")
        return out

    news_df = pd.read_csv(news_csv)
    if news_df.empty:
        out = firms_df.copy()
        out["score"] = 0.0
        atomic_write_rows_df(out, out_scores_csv)
        # marker
        from datetime import datetime
        stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        marker = out_scores_csv.parent / "last_scorer.txt"
        with open(marker, "w", encoding="utf-8") as f:
            f.write(
                "scorer=OpenAI\n"
                f"model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}\n"
                "headlines_scored=0\n"
                f"written={out_scores_csv.name}\n"
                f"time={stamp}\n"
            )
        log(f"Wrote marker: {marker}")
        log(f"[OK] scored {len(out)} firms -> {out_scores_csv}")
        return out

    # clean headlines
    news_df["headline"] = news_df["headline"].fillna("").astype(str)

    # --- FORCE OpenAI path (no fallback) ---
    log("compute_scores_and_write -> OpenAI-only path")
    log(f"Scoring {len(news_df)} headlines via OpenAI...")
    scores = score_headlines_openai(
        list(news_df["headline"]),
        batch_size=40,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    log(f"OpenAI returned {len(scores)} scores (first 5: {scores[:5]})")

    # attach & ensure numeric Series
    news_df["headline_score"] = pd.to_numeric(pd.Series(scores, index=news_df.index), errors="coerce").fillna(0.0)

    # aggregate per firm
    agg = (
        news_df.groupby("firm_name", as_index=False)["headline_score"]
               .mean()
               .rename(columns={"headline_score": "score"})
    )
    out = firms_df.merge(agg, on="firm_name", how="left")
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)

    # seatbelts: clip → shrink → snap → clip
    out["score"] = out["score"].clip(-5, 5)
    out["score"] = 0.9 * out["score"]                    # gentle shrink
    out["score"] = (out["score"] * 2).round() / 2.0      # snap to 0.5 steps
    out["score"] = out["score"].clip(-5, 5).round(2)

    # write CSV atomically
    atomic_write_rows_df(out, out_scores_csv)
    log(f"[OK] scored {len(out)} firms -> {out_scores_csv}")

    # marker file to prove OpenAI was used
    from datetime import datetime
    stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    marker = out_scores_csv.parent / "last_scorer.txt"
    with open(marker, "w", encoding="utf-8") as f:
        f.write(
            "scorer=OpenAI\n"
            f"model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}\n"
            f"headlines_scored={len(news_df)}\n"
            f"written={out_scores_csv.name}\n"
            f"time={stamp}\n"
        )
    log(f"Wrote marker: {marker}")

    return out




def atomic_write_rows_df(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    try:
        os.replace(tmp, out_csv)
    except PermissionError:
        ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        alt = out_csv.with_name(f"{out_csv.stem}_{ts}{out_csv.suffix}")
        os.replace(tmp, alt)
        print(f"[WARN] {out_csv} was locked. Wrote fallback: {alt}")

# ---------- Email helper (SMTP) ----------
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_scores_email(df, scores_csv: Path):
    api_key = os.environ.get("SENDGRID_API_KEY")
    mail_from = os.environ.get("MAIL_FROM")
    mail_to = os.environ.get("MAIL_TO")
    if not (api_key and mail_from and mail_to):
        print("[WARN] SENDGRID_API_KEY / MAIL_FROM / MAIL_TO not set; skipping email.")
        return

    date_str = datetime.now(TZ).strftime("%Y-%m-%d")
    html_table = df.to_html(index=False)
    message = Mail(
        from_email=mail_from,
        to_emails=mail_to,
        subject=f"Daily sentiment scores — {date_str}",
        html_content=f"<p>Firm sentiment scores for <b>{date_str}</b> (−5↔+5):</p>{html_table}<p>CSV attached.</p>",
    )

    # attach the CSV
    data = scores_csv.read_bytes()
    encoded = base64.b64encode(data).decode()
    attachment = Attachment(
        FileContent(encoded),
        FileName("scores_today.csv"),
        FileType("text/csv"),
        Disposition("attachment"),
    )
    message.attachment = attachment

    sg = SendGridAPIClient(api_key)
    sg.send(message)
    print(f"[OK] emailed scores to {mail_to} via SendGrid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--firms", type=str, default=str((Path(__file__).parent / "firms.csv").resolve()))
    parser.add_argument("--out", type=str, default=str((Path(__file__).parent / "data" / "todays_news.csv").resolve()))
    parser.add_argument("--headless", type=lambda s: s.lower() not in {"0","false","no"}, default=True)
    args = parser.parse_args()
    scrape_once(Path(args.firms), Path(args.out), headless=args.headless)
