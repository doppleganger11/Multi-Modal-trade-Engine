Multi-Modal Trade Engine
A lightweight, end-to-end pipeline for algorithmic trading based on momentum, mean-reversion, and news sentiment signals for a defined universe of Indian equities.

How It Works
This engine automates the entire process from data collection to order generation in a sequential pipeline:

Scrapes the latest news headlines for a fixed universe of Indian equities from Groww pages.

Scores each headline for sentiment using the OpenAI API (on a scale of -5 to +5).

Fetches daily historical prices and volumes via Yahoo Finance.

Builds weekly Average Daily Volume (ADV) and Average Daily Value (ADVVAL) as liquidity guardrails.

Computes trading signals by blending momentum, mean-reversion, and the news sentiment scores.

Constructs a target portfolio using Kelly/mean-variance style weighting, subject to liquidity caps.

Outputs final share orders sized according to available capital and ADV constraints.

Repository Contents
This repository contains the core scripts and configuration files that drive the trading engine.

Configuration
firms.csv: Defines the universe of equities for the engine. It must contain firm_name, url, and either a ticker or yf_ticker.

Execution Scripts
groww_playwright_scrape.py

Scrapes news cards from Groww for each firm, deduplicates them, and scores each headline using OpenAI.

Output: data/todays_news.csv

prices_pipeline.py

Performs an incremental daily fetch of price and volume data from Yahoo Finance, with NSE/BSE fallbacks and robust retries. It only appends new data.

Output: data/prices_daily.csv

build_weekly_adv.py

Aggregates the daily price data into weekly ADV (shares) and ADVVAL (shares × close) guardrails, using Friday week-ends.

Output: data/weekly_adv.csv

signals_pipeline.py

Builds trading signals from prices (e.g., 20-day momentum, mean-reversion, volatility) and blends them with the headline sentiment from todays_news.csv.

Output: data/signals_today.csv

portfolio_engine.py

Converts the generated signals into target portfolio weights, inspired by Kelly/mean-variance. It applies liquidity caps using ADV/ADVVAL, enforces per-name limits, and normalizes weights to 100%.

Output: data/portfolio_today.csv

order_build.py

Translates the final portfolio weights into concrete share orders for a given capital, honoring ADV caps and rounding.

Output: data/orders_today.csv

Outputs
The primary outputs of the pipeline are stored in the data/ directory as CSV files, providing full transparency at each step.

data/todays_news.csv: Contains the scraped headlines and their corresponding sentiment scores.

data/prices_daily.csv: A long-format panel of daily price and volume data for each equity.

data/weekly_adv.csv: Contains weekly average daily volume and ADVVAL.

data/signals_today.csv: The generated model signals (alpha, sigma, etc.) for each firm.

data/portfolio_today.csv: The target portfolio weights with notes on liquidity constraints.

data/orders_today.csv: The final share counts and notional value per firm, ready for execution.

Getting Started
Prerequisites
Python 3.8+

An OpenAI API Key

Installation
Clone the repository:

git clone [https://github.com/your-username/multi-modal-trade-engine.git](https://github.com/your-username/multi-modal-trade-engine.git)
cd multi-modal-trade-engine

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Set up your environment variables (e.g., in a .env file):

OPENAI_API_KEY='your_api_key_here'

Usage
Run the main pipeline script (assuming you create one to orchestrate the steps):

python main.py

Or run the scripts individually in the correct order.
