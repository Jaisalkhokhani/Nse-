import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="NSE Stock Research",
    page_icon="📈",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")
today = datetime.now(IST).date()

st.title("📈 NSE Stock Research")
st.caption(
    f"India date: {today} | Research and paper trading only. "
    "Signals are not guaranteed predictions."
)
st.warning(
    "Prices may be delayed. This app does not place orders. "
    "A technical score is NOT a probability of profit."
)


@st.cache_data(ttl=86400, show_spinner=False)
def company_list(day):
    url = (
        "https://archives.nseindia.com/content/"
        "equities/EQUITY_L.csv"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=25,
    )
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = frame.columns.str.strip()
    frame = frame.rename(
        columns={"NAME OF COMPANY": "Company", "SYMBOL": "Symbol"}
    )
    if "SERIES" in frame:
        frame = frame[
            frame["SERIES"].astype(str).str.strip().eq("EQ")
        ]
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip()
    return frame[["Company", "Symbol"]].drop_duplicates()


@st.cache_data(ttl=86400, show_spinner=False)
def nifty500(day):
    urls = [
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/"
        "ind_nifty500list.csv",
    ]
    errors = []
    for url in urls:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=25,
            )
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            frame.columns = frame.columns.str.strip()
            if "Symbol" not in frame:
                raise ValueError("Symbol column not found")
            return frame
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Nifty 500 list unavailable: " + "; ".join(errors))


@st.cache_data(ttl=900, show_spinner=False)
def prices(symbol, period="2y", interval="1d"):
    frame = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"No price data returned for {symbol}")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.loc[
        :, ~frame.columns.duplicated()
    ].dropna(subset=["Open", "High", "Low", "Close"])

    # Exclude today's daily candle; it may still be incomplete.
    if interval == "1d":
        frame = frame[
            np.array(frame.index.date) < datetime.now(IST).date()
        ]
    else:
        # Retain only completed intraday candles.
        minutes = int(interval.replace("m", ""))
        index = pd.DatetimeIndex(frame.index)
        if index.tz is None:
            index = index.tz_localize(IST)
        now = pd.Timestamp.now(tz=IST)
        frame = frame[
            index + pd.Timedelta(minutes=minutes) <= now
        ]

    if frame.empty:
        raise ValueError("No completed candles available")
    return frame


def indicators(frame):
    d = frame.copy()
    d["EMA20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = d.Close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = d.Close.ewm(span=200, adjust=False).mean()

    change = d.Close.diff()
    gains = change.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()
    losses = (-change.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    ratio = gains / losses.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + ratio)
    d.loc[(losses == 0) & (gains > 0), "RSI"] = 100
    d.loc[(losses == 0) & (gains == 0), "RSI"] = 50

    previous_close = d.Close.shift(1)
    ranges = pd.concat(
        [
            d.High - d.Low,
            (d.High - previous_close).abs(),
            (d.Low - previous_close).abs(),
        ],
        axis=1,
    )
    d["ATR"] = ranges.max(axis=1).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    d["VolumeRatio"] = (
        d.Volume
        / d.Volume.rolling(20).mean().shift(1).replace(0, np.nan)
    )
    d["PriorHigh20"] = d.High.rolling(20).max().shift(1)

    d["BuySetup"] = (
        (d.Close > d.EMA50)
        & (d.EMA50 > d.EMA200)
        & (d.Close > d.PriorHigh20)
        & d.RSI.between(55, 75)
        & (d.VolumeRatio >= 1.2)
    )
    d["ExitSetup"] = d.Close < d.EMA20
    return d


def technical_score(row):
    points = 0
    points += 25 if row.Close > row.EMA50 else 0
    points += 20 if row.EMA50 > row.EMA200 else 0
    points += 25 if row.Close > row.PriorHigh20 else 0
    points += 15 if 55 <= row.RSI <= 75 else 0
    points += 15 if row.VolumeRatio >= 1.2 else 0
    return points


def trade_plan(entry, atr, capital, risk_percent):
    stop = entry - 2 * atr
    if not np.isfinite(atr) or atr <= 0 or stop <= 0:
        return None

    risk_per_share = entry - stop
    risk_quantity = int(
        capital * risk_percent / 100 / risk_per_share
    )
    affordable_quantity = int(capital / entry)
    quantity = max(0, min(risk_quantity, affordable_quantity))

    return {
        "Reference price": round(entry, 2),
        "Illustrative stop": round(stop, 2),
        "Illustrative target": round(entry + 4 * atr, 2),
        "Quantity": quantity,
        "Capital required": round(quantity * entry, 2),
        "Planned stop risk": round(quantity * risk_per_share, 2),
    }


@st.cache_data(ttl=900, show_spinner=False)
def recent_news(company):
    response = requests.get(
        "https://news.google.com/rss/search",
        params={
            "q": f'"{company}" stock India when:3d',
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        },
        timeout=20,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    return [
        {
            "Headline": item.findtext("title", ""),
            "Published": item.findtext("pubDate", ""),
            "Link": item.findtext("link", ""),
        }
        for item in root.findall(".//item")[:12]
    ]


def historical_test(d, capital, cost_bps):
    # Long-only, one position at a time.
    # Yesterday's close signal executes at today's open.
    # ATR stop and target are fixed when entering.
    cash = float(capital)
    position = None
    trades = []
    equity = []
    cost_rate = cost_bps / 10000

    for i in range(201, len(d)):
        bar = d.iloc[i]
        previous = d.iloc[i - 1]
        date = d.index[i]

        if position is None and bool(previous.BuySetup):
            entry = float(bar.Open)
            atr = float(previous.ATR)
            plan = trade_plan(entry, atr, cash, 1.0)
            if plan:
                quantity = min(
                    plan["Quantity"],
                    int(cash / (entry * (1 + cost_rate))),
                )
                if quantity > 0:
                    position = {
                        "Entry date": date,
                        "Entry": entry,
                        "Quantity": quantity,
                        "Stop": entry - 2 * atr,
                        "Target": entry + 4 * atr,
                        "Entry cost": quantity * entry * cost_rate,
                    }
                    cash -= quantity * entry * (1 + cost_rate)

        if position is not None:
            exit_price = None
            reason = ""
            p = position

            if bar.Open <= p["Stop"]:
                exit_price, reason = float(bar.Open), "Gap below stop"
            elif bar.Open >= p["Target"]:
                exit_price, reason = float(bar.Open), "Gap above target"
            elif bool(previous.ExitSetup):
                exit_price, reason = float(bar.Open), "Trend exit"
            elif bar.Low <= p["Stop"]:
                # If stop and target both touch, assume stop first.
                exit_price, reason = p["Stop"], "Stop"
            elif bar.High >= p["Target"]:
                exit_price, reason = p["Target"], "Target"

            if exit_price is not None:
                proceeds = p["Quantity"] * exit_price
                exit_cost = proceeds * cost_rate
                cash += proceeds - exit_cost
                pnl = (
                    p["Quantity"] * (exit_price - p["Entry"])
                    - p["Entry cost"]
                    - exit_cost
                )
                trades.append(
                    {
                        **p,
                        "Exit date": date,
                        "Exit": exit_price,
                        "Net P&L": pnl,
                        "Reason": reason,
                    }
                )
                position = None

        mark = cash
        if position is not None:
            mark += position["Quantity"] * float(bar.Close)
        equity.append({"Date": date, "Equity": mark})

    return pd.DataFrame(trades), pd.DataFrame(equity), position


try:
    companies = company_list(str(today))
except Exception as exc:
    st.warning(
        "Full NSE company list is unavailable. "
        "Using a small fallback list, not the full market."
    )
    companies = pd.DataFrame(
        {
            "Company": [
                "Reliance Industries",
                "Tata Consultancy Services",
                "HDFC Bank",
                "Infosys",
                "State Bank of India",
            ],
            "Symbol": [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"
            ],
        }
    )

labels = (
    companies.Company.astype(str)
    + " | "
    + companies.Symbol.astype(str)
).tolist()
lookup = dict(zip(labels, companies.Symbol))
name_lookup = dict(zip(companies.Symbol, companies.Company))

with st.sidebar:
    st.header("Research settings")
    capital = st.number_input(
        "Available capital (INR)",
        min_value=1000,
        value=100000,
        step=1000,
    )
    risk_percent = st.slider(
        "Planned risk per trade (%)", 0.25, 2.0, 0.5, 0.25
    )
    st.caption(
        "Position sizing is for one trade only. "
        "Gap losses can exceed planned stop risk."
    )

    if st.button("Clear cached data"):
        st.cache_data.clear()
        st.rerun()

tabs = st.tabs(
    [
        "Daily watchlist",
        "Stock analysis & news",
        "Backtest",
        "Intraday",
        "Paper journal",
    ]
)

with tabs[0]:
    st.subheader("Daily qualifying watchlist")
    st.write(
        "Uses completed daily candles. Scanning runs when this "
        "page opens with the option below enabled. It does not "
        "run in the background while the app is closed."
    )

    auto = st.checkbox("Scan automatically when opened", value=True)
    limit = st.select_slider(
        "Nifty 500 stocks to examine",
        options=[25, 50, 100, 250, 500],
        value=25,
    )
    st.caption(
        "Partial scans use an alphabetical subset, not the "
        "best stocks in the entire market. Select 500 for full "
        "Nifty 500 coverage. BSE-only shares are not included."
    )

    @st.cache_data(ttl=86400, show_spinner=False)
    def scan(day, count):
        members = nifty500(day)
        symbols = sorted(
            members.Symbol.astype(str).str.strip().unique()
        )[:count]
        benchmark = indicators(prices("^NSEI"))
        benchmark_up = (
            benchmark.Close.iloc[-1] > benchmark.EMA50.iloc[-1]
        )

        rows = []
        failures = []
        for symbol in symbols:
            try:
                d = indicators(prices(symbol + ".NS"))
                if len(d) < 220:
                    raise ValueError("Insufficient price history")
                r = d.iloc[-1]
                if bool(r.BuySetup) and benchmark_up:
                    rows.append(
                        {
                            "Company": name_lookup.get(symbol, symbol),
                            "Symbol": symbol,
                            "Technical score / 100": technical_score(r),
                            "Last completed session": str(d.index[-1].date()),
                            "Reference close": round(float(r.Close), 2),
                            "RSI": round(float(r.RSI), 1),
                            "Volume ratio": round(float(r.VolumeRatio), 2),
                            "News status": "NOT REVIEWED",
                        }
                    )
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")

        if rows:
            rows = sorted(
                rows,
                key=lambda x: (
                    x["Technical score / 100"],
                    x["Volume ratio"],
                ),
                reverse=True,
            )[:10]
        return rows, failures, len(symbols), bool(benchmark_up)

    if auto or st.button("Run daily scan"):
        with st.spinner("Downloading and screening stocks..."):
            try:
                rows, failures, examined, market_up = scan(
                    str(today), limit
                )
                st.write(
                    f"Examined: {examined} | "
                    f"Data failures: {len(failures)} | "
                    f"Nifty above EMA50: {market_up}"
                )
                if rows:
                    result = pd.DataFrame(rows)
                    st.dataframe(result, hide_index=True)
                    st.download_button(
                        "Download watchlist",
                        result.to_csv(index=False),
                        "watchlist.csv",
                        "text/csv",
                    )
                else:
                    st.info(
                        "No stocks passed the filters. "
                        "The app will not invent 5–10 picks."
                    )
                if failures:
                    with st.expander("Data failures"):
                        st.write(failures)
                st.caption(
                    "Watchlist only—not buy orders. Check latest "
                    "prices, results announcements and news first."
                )
            except Exception as exc:
                st.error(f"Scan unavailable: {exc}")

with tabs[1]:
    selection = st.selectbox("Search company", labels, key="company")
    symbol = lookup[selection]
    manual = st.text_input(
        "Or enter an NSE symbol",
        placeholder="Example: RELIANCE",
    ).strip().upper()
    if manual:
        symbol = manual.replace(".NS", "")

    if st.button("Analyse stock"):
        try:
            d = indicators(prices(symbol + ".NS"))
            if len(d) < 220:
                raise ValueError("At least 220 sessions are required")
            r = d.iloc[-1]
            st.write(f"Latest completed daily candle: {d.index[-1]}")
            if bool(r.BuySetup):
                st.success("BUY SETUP — pending news and price review")
            elif bool(r.ExitSetup):
                st.warning(
                    "EXIT REVIEW — close is below EMA20. "
                    "Not an instruction to short-sell."
                )
            else:
                st.info("WAIT — no qualifying new entry")

            st.metric(
                "Technical score—not probability",
                f"{technical_score(r)}/100",
            )
            st.line_chart(
                d[["Close", "EMA20", "EMA50", "EMA200"]].tail(200)
            )
            plan = trade_plan(
                float(r.Close),
                float(r.ATR),
                capital,
                risk_percent,
            )
            if plan:
                st.write(plan)
                st.caption(
                    "Illustrative plan based on the historical "
                    "close, not a live executable quote. "
                    "Recalculate using your actual entry price."
                )

            st.subheader("Recent news — manual review required")
            try:
                articles = recent_news(
                    name_lookup.get(symbol, symbol)
                )
                if not articles:
                    st.warning(
                        "No headlines returned. This does NOT "
                        "mean there is no negative news."
                    )
                for article in articles:
                    st.write(article["Published"])
                    st.link_button(
                        article["Headline"], article["Link"]
                    )
            except Exception as exc:
                st.warning(
                    f"News unavailable; safety check incomplete: {exc}"
                )
        except Exception as exc:
            st.error(str(exc))

with tabs[2]:
    st.subheader("Historical rules-based simulation")
    st.caption(
        "Not an ML validation. Uses the same stock-level entry "
        "rules, but excludes historical news and Nifty filtering. "
        "Results must not be presented as the full app's performance."
    )
    back_symbol = st.text_input(
        "NSE symbol for backtest", "RELIANCE"
    ).strip().upper().replace(".NS", "")
    cost = st.number_input(
        "Cost per side (basis points)",
        min_value=0.0,
        value=15.0,
        step=1.0,
    )
    st.caption(
        "15 basis points = 0.15% on entry and 0.15% on exit. "
        "This is an adjustable cost assumption, not a tax estimate."
    )

    if st.button("Run historical simulation"):
        try:
            d = indicators(prices(back_symbol + ".NS", "5y"))
            if len(d) < 260:
                raise ValueError("Insufficient history")
            trades, curve, open_position = historical_test(
                d, capital, cost
            )
            if curve.empty:
                st.info("No simulation results")
            else:
                st.line_chart(curve.set_index("Date")["Equity"])
                end_equity = float(curve.Equity.iloc[-1])
                peak = curve.Equity.cummax().clip(lower=capital)
                drawdown = curve.Equity / peak - 1
                st.write(
                    {
                        "Marked-to-market return %": round(
                            (end_equity / capital - 1) * 100, 2
                        ),
                        "Maximum drawdown %": round(
                            float(drawdown.min()) * 100, 2
                        ),
                        "Closed trades": len(trades),
                        "Open position at end": open_position is not None,
                    }
                )
                if not trades.empty:
                    wins = trades["Net P&L"] > 0
                    st.metric(
                        "Closed-trade win rate",
                        f"{wins.mean() * 100:.1f}%",
                    )
                    st.dataframe(trades, hide_index=True)
                    st.download_button(
                        "Download simulated trades",
                        trades.to_csv(index=False),
                        "backtest_trades.csv",
                        "text/csv",
                    )
                st.warning(
                    "Historical simulation only. No claim of "
                    "predictive accuracy. Stop fills can be worse "
                    "than simulated; corporate actions, liquidity, "
                    "parameter selection and survivorship can "
                    "affect results."
                )
        except Exception as exc:
            st.error(str(exc))

with tabs[3]:
    st.subheader("Completed-candle intraday research")
    intraday_symbol = st.text_input(
        "Intraday NSE symbol", "RELIANCE"
    ).strip().upper().replace(".NS", "")

    if st.button("Check 15-minute trend"):
        try:
            d = prices(
                intraday_symbol + ".NS", period="5d", interval="15m"
            ).copy()
            d["EMA9"] = d.Close.ewm(span=9, adjust=False).mean()
            d["EMA21"] = d.Close.ewm(span=21, adjust=False).mean()
            r = d.iloc[-1]
            now = datetime.now(IST)
            last_date = d.index[-1].date()
            hours = (
                now.weekday() < 5
                and (9, 15) <= (now.hour, now.minute) < (15, 30)
                and last_date == now.date()
            )
            st.write(f"Latest completed candle: {d.index[-1]}")
            if not hours:
                st.warning(
                    "No current-session signal: market closed "
                    "or data stale. Holiday calendar not verified."
                )
            elif r.Close > r.EMA9 > r.EMA21:
                st.success("BULLISH TREND — research signal only")
            elif r.Close < r.EMA9 < r.EMA21:
                st.warning("BEARISH TREND — review existing long")
            else:
                st.info("WAIT — mixed trend")
            st.line_chart(d[["Close", "EMA9", "EMA21"]].tail(80))
            st.caption(
                "Not streaming live prices. Cached for up to "
                "15 minutes. This module is not the daily strategy."
            )
        except Exception as exc:
            st.error(str(exc))

with tabs[4]:
    st.subheader("Session-only paper journal")
    st.caption(
        "Private to this browser session, not permanent storage. "
        "Download your CSV before closing or refreshing the session."
    )
    if "journal" not in st.session_state:
        st.session_state.journal = []

    with st.form("paper_trade"):
        paper_symbol = st.text_input("Paper-trade symbol", "RELIANCE")
        quantity = st.number_input("Shares", min_value=1, value=1)
        entry = st.number_input(
            "Entry price", min_value=0.01, value=100.0
        )
        exit_price = st.number_input(
            "Exit price", min_value=0.01, value=100.0
        )
        notes = st.text_input("Notes")
        submitted = st.form_submit_button("Record closed paper trade")

    if submitted:
        st.session_state.journal.append(
            {
                "Date": str(today),
                "Symbol": paper_symbol.upper(),
                "Quantity": quantity,
                "Entry": entry,
                "Exit": exit_price,
                "Gross P&L before costs": quantity * (exit_price - entry),
                "Notes": notes,
            }
        )

    if st.session_state.journal:
        journal = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal, hide_index=True)
        st.download_button(
            "Download paper journal",
            journal.to_csv(index=False),
            "paper_journal.csv",
            "text/csv",
        )
