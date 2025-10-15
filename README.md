# AI News Sentiment Analyser — README

A compact guide to run the pipeline end-to-end: what it does, what each main script is responsible for, what outputs you get, the environment you must set up, and the exact execution order.

---

## 1) What the Pipeline Does (in order)

1. **Scrape** latest headlines for selected Indian stocks from **Groww** news pages.
2. **Clean & Deduplicate** rows and normalize relative times (e.g., “2 hours ago”) to timestamps.
3. **Score Sentiment** of each headline using **OpenAI** (range: −5 … +5) with a polarity label.
4. **Persist** results to a CSV log for the day (append-only, idempotent).
5. **Notify (Optional)** via email if new headlines cross a score threshold.

Pipeline summary: **Scrape → Clean/Dedupe → Score → Save CSV → Email (optional)**

---

## 2) Main Code Responsibilities

> Adjust file names if your repo uses slightly different paths. The responsibilities below match the canonical layout.

- `src/main.py`  
  Orchestrates the full run (scrape → score → save → email). Exposes CLI flags like `--once`.

- `src/config.py`  
  Loads environment variables (`.env`), constants (paths, thresholds, timezone), and the ticker list (from `.env` or `tickers.txt`).

- `src/scraping/groww_playwright_scrape.py`  
  Uses **Playwright (Chromium)** to visit Groww pages, scroll, and extract rows:  
  `firm_name, headline, article_url, rel_time_raw, source_url`.

- `src/scraping/parse.py`  
  Utilities to clean text, parse “X mins/hours ago” into timestamps, and construct dedupe keys.

- `src/scoring/score_headlines_openai.py`  
  Sends headlines to **OpenAI** and returns `headline_score ∈ [−5, +5]` plus `sentiment_label` (`negative/neutral/positive`).

- `src/notify/mailer.py`  
  Sends a summary email (HTML + plaintext) grouped by firm with links and scores.

- `src/notify/formatter.py`  
  Formats the email body (sections, headings, table-like lists).

- `src/utils/logger.py`  
  Configures rotating file logs (INFO by default).

- `src/utils/timeutils.py`  
  Timezone helpers (defaults to `Asia/Kolkata`) and ISO conversions.

---

## 3) Outputs You Get

- **CSV (daily append):** `data/todays_news.csv`  
  Columns:
  - `scraped_at` (UTC ISO)
  - `published_at` (best-effort from relative time)
  - `firm_name`
  - `headline`
  - `article_url`
  - `source_url` (Groww listing)
  - `rel_time_raw`
  - `headline_score` (−5 … +5)
  - `sentiment_label` (`negative`/`neutral`/`positive`)
  - `model_name` (OpenAI model)
  - `run_id` (UUID of the job)

- **Logs:** `logs/run.log` (rotating)
- **Email (optional):** summary of new headlines crossing the configured score threshold.

> **Deduplication key:** `(firm_name, headline)` ensures repeated runs don’t re-insert the same row.

---

## 4) Environment Setup (do this once)

### Prerequisites
- **Python** 3.10+
- **Playwright** with **Chromium** browser
- **OpenAI API key**
- **SMTP** account (e.g., Gmail with an **App Password**)

### Create virtual environment & install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium

