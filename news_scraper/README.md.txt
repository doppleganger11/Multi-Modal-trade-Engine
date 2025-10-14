# Groww Trade Engine

Daily pipeline to scrape news (Playwright), score sentiment (OpenAI),
pull prices (Yahoo Finance), compute signals, build a portfolio, and output orders.

## Quick start
```bash
conda env create -f environment.yml
conda activate growwtrade
copy .env.example .env   # Windows
# edit .env with your keys (OPENAI_API_KEY=...)
python groww_playwright_scrape.py --firms "..\firms.csv" --out ".\data\todays_news.csv" --headless True
python prices_pipeline.py --firms "..\firms.csv"
python build_weekly_adv.py
python signals_pipeline.py
python portfolio_engine.py --capital 10000
python order_build.py --capital 10000 --max_name_frac 0.15
