
import requests
import time
import pandas as pd

# ==============================
# GET API KEYS FROM SETX
# ==============================

import os

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# ==============================
# BOT SETTINGS
# ==============================

SYMBOL = "SOFI"

TRADE_SIZE = 200

TAKE_PROFIT = 0.008   # 0.8%
STOP_LOSS = 0.004     # 0.4%

SPREAD_LIMIT = 0.5

CHECK_INTERVAL = 20


# ==============================
# GET CURRENT QUOTE
# ==============================

def get_quote(symbol):

    url = f"{DATA_URL}/v2/stocks/{symbol}/quotes/latest"

    r = requests.get(url, headers=HEADERS)

    data = r.json()

    bid = data["quote"]["bp"]
    ask = data["quote"]["ap"]

    return bid, ask


# ==============================
# CALCULATE SPREAD
# ==============================

def spread_percent(bid, ask):

    mid = (bid + ask) / 2

    spread = (ask - bid) / mid * 100

    return spread


# ==============================
# GET LAST BARS
# ==============================

def get_bars(symbol):

    url = f"{DATA_URL}/v2/stocks/{symbol}/bars?timeframe=5Min&limit=50"

    r = requests.get(url, headers=HEADERS)

    data = r.json()["bars"]

    df = pd.DataFrame(data)

    return df


# ==============================
# CALCULATE EMA
# ==============================

def calculate_ema(df):

    df["ema9"] = df["c"].ewm(span=9).mean()

    df["ema21"] = df["c"].ewm(span=21).mean()

    return df


# ==============================
# BUY ORDER
# ==============================

def buy(symbol, price):

    qty = int(TRADE_SIZE / price)

    order = {

        "symbol": symbol,
        "qty": qty,
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": round(price,2)

    }

    r = requests.post(BASE_URL+"/v2/orders",json=order,headers=HEADERS)

    print("BUY ORDER:", r.json())

    return price, qty


# ==============================
# SELL ORDER
# ==============================

def sell(symbol, price, qty):

    order = {

        "symbol": symbol,
        "qty": qty,
        "side": "sell",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": round(price,2)

    }

    r = requests.post(BASE_URL+"/v2/orders",json=order,headers=HEADERS)

    print("SELL ORDER:", r.json())


# ==============================
# MAIN BOT
# ==============================

position = False
entry_price = 0
shares = 0

while True:

    try:

        bid, ask = get_quote(SYMBOL)

        spread = spread_percent(bid, ask)

        print("PRICE:", ask, "SPREAD:", spread)

        if spread > SPREAD_LIMIT:

            print("Spread too high")

            time.sleep(CHECK_INTERVAL)

            continue

        bars = get_bars(SYMBOL)

        bars = calculate_ema(bars)

        ema9 = bars["ema9"].iloc[-1]
        ema21 = bars["ema21"].iloc[-1]

        price = bars["c"].iloc[-1]

        # ==========================
        # BUY SIGNAL
        # ==========================

        if not position:

            if ema9 > ema21:

                entry_price, shares = buy(SYMBOL, ask)

                position = True

        # ==========================
        # SELL SIGNAL
        # ==========================

        else:

            if price >= entry_price * (1 + TAKE_PROFIT):

                sell(SYMBOL, bid, shares)

                position = False

            elif price <= entry_price * (1 - STOP_LOSS):

                sell(SYMBOL, bid, shares)

                position = False

        time.sleep(CHECK_INTERVAL)

    except Exception as e:

        print("ERROR:", e)

        time.sleep(10)