import pandas as pd
from exchange import get_exchange

exchange = get_exchange()

def get_top_symbols(limit=20):

    try:

        markets = exchange.fetch_markets()

        perps = []

        for market in markets:

            if (
                market.get("swap")
                and market.get("quote") == "USDT"
                and market.get("active")
            ):
                perps.append(market)

        perps.sort(
            key=lambda x: float(
                x.get("info", {}).get("turnover24h", 0)
            ),
            reverse=True
        )

        return [x["symbol"] for x in perps[:limit]]

    except Exception as e:

        print(f"Top Symbol Error: {e}")

        return [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT"
        ]


def get_ohlcv(symbol, timeframe, limit=200):

    try:

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        return df

    except Exception as e:

        print(f"{symbol} OHLCV Error: {e}")

        return None
