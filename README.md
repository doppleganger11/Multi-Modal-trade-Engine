# Multi-Modal-trade-Engine
A lightweight, end-to-end pipeline that:

-scrapes latest news headlines for a fixed universe of Indian equities (Groww pages),
-scores each headline via OpenAI sentiment (−5…+5),
-fetches daily prices & volumes via Yahoo Finance,
-builds weekly ADV/ADVVAL guardrails,
-computes signals (momentum / mean-reversion + sentiment blend),
-constructs a portfolio (Kelly/mean-variance style weights with liquidity caps),
-outputs share orders sized to capital and ADV constraints.
