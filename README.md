# Crypto Price Predictor

An end-to-end crypto analytics pipeline in Python that scrapes live market data for the top 100
cryptocurrencies, persists it to PostgreSQL, and uses a stacked LSTM neural network to forecast future
price trajectories.

The project has two halves: a **data pipeline** that keeps a local database of live market data, and a
**forecasting engine** that trains a recurrent neural network on a year of historical prices and
projects a coin's price 1–60 days into the future.

---

## Highlights

- **Live data ingestion** — Scrapes the top 100 coins from [CoinMarketCap](https://coinmarketcap.com/)
  with Playwright (headless Chromium), collecting price, market cap, and 24h trading volume.
- **Persistent storage** — Every scrape is written to a **PostgreSQL** database using `psycopg2` batch
  upserts (`execute_values` + `ON CONFLICT`), so the table always reflects the latest snapshot while
  retaining a full price history.
- **Historical data** — Pulls **365 days** of price history per coin from the
  [CoinGecko API](https://www.coingecko.com/en/api) `market_chart` endpoint.
- **LSTM forecasting** — A three-layer stacked LSTM network, trained with MinMaxScaler-normalized
  inputs, that predicts a coin's price anywhere from 1 to 60 days ahead.
- **Full ML lifecycle** — Feature engineering, train/test splitting, scaling, model training,
  evaluation, and inverse-transformed predictions rendered as actual-vs-predicted charts.
- **Desktop GUI** — A Tkinter table view of the scraped top 100 with inputs for coin and horizon,
  so the whole pipeline is usable without touching the command line.

## Model Performance

Evaluated on held-out data using **Mean Absolute Percentage Error (MAPE)**:

| Forecast horizon | MAPE |
| ---------------- | ---- |
| 1–3 days         | ~15% |
| ~30 days         | ~30% |

Error grows with horizon length, which is expected for time-series forecasting: near-term structure
carries more signal than long-range trend. Treat these as directional projections, not price
guarantees — crypto markets are not predictable to that precision.

## Tech Stack

| Layer         | Technology                                        |
| ------------- | ------------------------------------------------- |
| Scraping      | Playwright (Chromium), Python                      |
| Storage       | PostgreSQL, `psycopg2`, `execute_values`           |
| Data          | `pandas`, `numpy`, `requests`                      |
| ML            | TensorFlow / Keras, `scikit-learn` (MinMaxScaler)  |
| Visualization | `matplotlib`                                       |
| Interface     | `tkinter` / `ttk`                                  |

## Architecture

```
CoinMarketCap ──▶ Playwright scrape ──▶ PostgreSQL (crypto table, upserted)
                                            │
                                            ▼
                                      Tkinter top-100 view
                                            │
CoinGecko API ──▶ 365d price history ──▶ MinMaxScaler
                                            │
                                            ▼
                          sliding windows (60-day lookback)
                                            │
                                            ▼
                       3× LSTM(50) + Dropout + Dense(1)
                                            │
                                            ▼
                          prediction ──▶ inverse_transform ──▶ chart
```

### Forecasting model

- **Input**: 60-day sliding window of daily prices, reshaped to `(samples, 60, 1)`.
- **Architecture**: `LSTM(50, return_sequences)` → `Dropout(0.2)` → `LSTM(50, return_sequences)` →
  `Dropout(0.2)` → `LSTM(50)` → `Dropout(0.2)` → `Dense(1)`.
- **Compilation**: Adam optimizer, mean squared error loss.
- **Training**: 25 epochs, batch size 32.
- **Scaling**: `MinMaxScaler` fitted on the training range only, so the test window is scaled without
  leaking statistics from the future; predictions are inverse-transformed back to USD before plotting.
- **Horizon**: the target offset is parameterized, so a single architecture supports 1–60 day targets.

## Data Model

The `crypto` table stores one row per coin, updated on each run:

| Column            | Type            | Description                          |
| ----------------- | --------------- | ------------------------------------ |
| `id`              | text (PK)       | CoinMarketCap rank/identifier        |
| `name`            | text            | Coin name                            |
| `symbol`          | text            | Ticker symbol                        |
| `price_usd`       | numeric         | Current price in USD                 |
| `market_cap_usd`  | bigint          | Market capitalization in USD         |
| `volume_24h_usd`  | bigint          | 24-hour trading volume in USD        |
| `scrape_date`     | timestamp       | Timestamp of the snapshot            |

Primary key on `id` with an upsert keeps the scraper idempotent — re-running refreshes values in
place instead of duplicating rows.

## Getting Started

### Requirements

- Python 3.10+
- PostgreSQL running locally
- A CoinMarketCap / CoinGecko connection (both APIs are free-tier; CoinGecko rate-limits unauthenticated
  requests, so space out repeated runs)

### Install

```bash
git clone https://github.com/dasuddybuddy/crypto-price-predictor.git
cd crypto-price-predictor

python -m venv path
source path/bin/activate        # Windows: path\Scripts\activate

pip install playwright tensorflow pandas numpy requests matplotlib scikit-learn pandas-datareader psycopg2-binary
playwright install chromium
```

### Database

Create the table the scraper writes into:

```sql
CREATE TABLE crypto (
    id             TEXT PRIMARY KEY,
    name           TEXT,
    symbol         TEXT,
    price_usd      NUMERIC,
    market_cap_usd BIGINT,
    volume_24h_usd BIGINT,
    scrape_date    TIMESTAMP
);
```

Then set your credentials in `app.py` (or, better, load them from environment variables):

```python
pgconn = psycopg2.connect(
    host=os.getenv("PGHOST", "localhost"),
    database=os.getenv("PGDATABASE", "postgres"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
)
```

### Run

```bash
python app.py
```

A Chromium window opens and scrolls CoinMarketCap to load the full top-100 table, the results are
upserted into PostgreSQL, and the Tkinter view launches. From there, enter a coin name (e.g.
`bitcoin`) and a horizon between 1 and 60 to train a model and view the predicted-vs-actual chart.

## Usage

```python
from app import crypto_prediction

crypto_prediction("bitcoin", 7)   # forecast BTC 7 days out
crypto_prediction("ethereum", 30) # forecast ETH 30 days out
```

## Project Structure

```
.
├── app.py            # scraper, DB layer, LSTM model, and Tkinter UI
├── crypto_table.sql  # helper query against the crypto table
├── path/             # local virtual environment (git-ignored)
└── README.md
```

## Future Work

- Move the Tkinter UI to a web dashboard (Streamlit or FastAPI) for remote access
- Add scheduled scrapes so the database builds a continuous time series rather than snapshots
- Layer in technical indicators (RSI, MACD, moving averages) as additional model features
- Walk-forward / time-series cross-validation instead of a single train/test split
- Serve trained models with joblib and add a "predict tonight's close" batch job
- Expand features beyond price: volume, market cap, and on-chain activity

## Notes

- The CoinMarketCap DOM selectors are tied to the current page layout and will need updating if the
  site changes its markup.
- Financial data comes from public endpoints and is provided for educational purposes only. Nothing
  here is investment advice.
