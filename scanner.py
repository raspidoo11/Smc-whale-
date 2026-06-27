import logging
import pandas as pd

from exchange import get_exchange

logger = logging.getLogger(__name__)
exchange = get_exchange()


def get_top_symbols(limit=20):
    try:
        logger.info("Loading Bybit markets...")

        markets = exchange.fetch_markets()

        perps = []

        for market in markets:
            if (
                market.get("swap")
                and market.get("quote") == "USDT"
                and market.get("active")
            ):
                perps.append(market)

        logger.info(
            f"Found {len(perps)} active USDT perpetuals"
        )

        perps.sort(
            key=lambda x: float(
                x.get("info", {}).get("turnover24h")
                or x.get("info", {}).get("volume24h")
                or 0
            ),
            reverse=True
        )

        symbols = [
            x["symbol"]
            for x in perps[:limit]
        ]

        logger.info(
            f"Selected top {len(symbols)} symbols"
        )

        return symbols

    except Exception as e:
        logger.exception(
            f"Top Symbol Error: {e}"
        )

        return [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
            "DOGE/USDT:USDT"
        ]


def get_ohlcv(
    symbol,
    timeframe,
    limit=200
):
    try:
        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not candles:
            logger.warning(
                f"{symbol} returned no candles"
            )
            return None

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

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        df = df.dropna()

        if len(df) < 50:
            logger.warning(
                f"{symbol} only returned "
                f"{len(df)} candles"
            )
            return None

        logger.info(
            f"{symbol} {timeframe} "
            f"candles loaded: {len(df)}"
        )

        return df

    except Exception as e:
        logger.exception(
            f"{symbol} OHLCV Error: {e}"
        )
        return None
