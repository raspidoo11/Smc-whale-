import ccxt

def get_exchange():

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap"
        }
    })

    exchange.load_markets()

    return exchange
