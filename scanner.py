import pandas as pd
import logging
import time
from exchange import get_exchange
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

exchange = get_exchange()

# ==========================================================
# FILTERS
# ==========================================================

# Meme coin patterns to filter OUT
MEME_PATTERNS = {
    'BABY', 'SAFE', 'MOON', 'LUNC', 'INU', 'COIN', 'TOKEN',
    '1000', '10000', 'ELON', 'DOGE', 'SHIB', 'PEPE', 'FLOKI',
    'X', 'V2', 'OLD', 'CUMROCKET', 'PUSSY', 'CUMMIES', 'RIBBIT'
}

# Tier 1: Always include (top tier coins)
TIER1 = {'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'LINK'}

# Stablecoins to block (stable vs stable pairs)
STABLECOINS = {'USDC', 'USDT', 'FDUSD', 'USDE', 'TUSD', 'BUSD', 'DAI', 'PYUSD', 'GUSD', 'USDM'}

# Cache for symbol list (update every 5 minutes)
SYMBOL_CACHE = {
    'symbols': [],
    'timestamp': None,
    'cache_duration': 300  # 5 minutes
}


def is_stablecoin_pair(symbol: str) -> bool:
    """
    Detect stablecoin vs stablecoin pairs.
    Blocks: USDCUSDT, USDTUSDC, FDUSDUSDT, USDEUSDT, TUSDUSDT, etc.
    """
    s = symbol.upper().replace("/", "").replace(":", "").replace("-", "")
    
    # Direct blocklist for known bad pairs
    blocked_pairs = {
        "USDCUSDT", "USDTUSDC", "FDUSDUSDT", "USDEUSDT",
        "TUSDUSDT", "BUSDUSDT", "USDTUSDT", "USDCUSDC"
    }
    if s in blocked_pairs:
        return True

    # General detection
    for stable in STABLECOINS:
        if s.endswith(stable):
            base = s[:-len(stable)]
            if base in STABLECOINS:
                return True
    return False


def is_meme_coin(symbol):
    """Check if symbol matches meme coin patterns"""
    base = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '').replace('FDUSD', '')
    
    for pattern in MEME_PATTERNS:
        if pattern in base:
            return True
    return False


def get_live_symbols(limit=30):
    """
    Fetch LIVE trading pairs from Bybit
    Filter by volume, quality, meme coins, and stablecoin pairs
    Cache results for 5 minutes
    """
    
    # Check cache
    if SYMBOL_CACHE['timestamp']:
        age = (datetime.now() - SYMBOL_CACHE['timestamp']).total_seconds()
        if age < SYMBOL_CACHE['cache_duration']:
            logger.info(f"Using cached symbols ({len(SYMBOL_CACHE['symbols'])} coins)")
            return SYMBOL_CACHE['symbols'][:limit]
    
    try:
        logger.info("🔍 Fetching live trading pairs from Bybit...")
        
        markets = exchange.load_markets()
        
        # Filter USDT pairs only
        usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('USDT')]
        
        # ==========================================================
        # STABLECOIN FILTER (NEW)
        # ==========================================================
        before_count = len(usdt_pairs)
        usdt_pairs = [s for s in usdt_pairs if not is_stablecoin_pair(s)]
        after_count = len(usdt_pairs)
        
        if before_count != after_count:
            logger.info(f"🛡️ Filtered out {before_count - after_count} stablecoin pairs")
        
        logger.info(f"Found {len(usdt_pairs)} quality USDT trading pairs (after stablecoin filter)")
        
        # Get volume data
        volume_data = []
        
        for symbol in usdt_pairs:
            try:
                time.sleep(0.05)  # Rate limit
                
                ticker = exchange.fetch_ticker(symbol)
                volume = ticker.get('quoteVolume', 0)
                
                # Quality filters
                if volume < 500000:  # Min $500k volume
                    continue
                
                # Skip meme coins
                if is_meme_coin(symbol):
                    logger.debug(f"Skipping meme coin: {symbol}")
                    continue
                
                volume_data.append({
                    'symbol': symbol,
                    'volume': volume,
                    'price': ticker.get('last', 0),
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0)
                })
            
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue
        
        # Sort by volume (descending)
        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        
        # Get top coins
        top_symbols = [item['symbol'] for item in volume_data[:limit]]
        
        # Ensure Tier 1 coins are included
        for coin in TIER1:
            symbol = f"{coin}USDT"
            if symbol not in top_symbols and symbol in [m['symbol'] for m in volume_data]:
                top_symbols.insert(0, symbol)
        
        top_symbols = top_symbols[:limit]
        
        logger.info(f"✅ Fetched {len(top_symbols)} quality coins from Bybit")
        logger.info(f"Top 5: {', '.join(top_symbols[:5])}")
        
        # Cache results
        SYMBOL_CACHE['symbols'] = top_symbols
        SYMBOL_CACHE['timestamp'] = datetime.now()
        
        return top_symbols
    
    except Exception as e:
        logger.error(f"Failed to fetch live symbols: {e}")
        if SYMBOL_CACHE['symbols']:
            logger.info("Using cached symbols (fetch failed)")
            return SYMBOL_CACHE['symbols'][:limit]
        return []


def get_ohlcv(symbol, timeframe, limit):
    """Fetch OHLCV data with error handling and rate limiting"""
    try:
        time.sleep(0.1)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            logger.warning(f"No OHLCV data for {symbol} {timeframe}")
            return None
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        df = df[(df['volume'] > 0) & (df['close'] > 0)]
        
        if len(df) < limit * 0.8:
            logger.warning(f"Insufficient data for {symbol}: {len(df)}/{limit} candles")
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol} {timeframe}: {e}")
        return None


def validate_symbol(symbol):
    """Check if symbol is tradeable"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        volume = ticker.get('quoteVolume', 0)
        
        if volume < 500000:
            return False
        if ticker.get('bid', 0) <= 0 or ticker.get('ask', 0) <= 0:
            return False
        
        # Also reject stablecoin pairs here as a safety net
        if is_stablecoin_pair(symbol):
            return False
            
        return True
    
    except Exception as e:
        logger.debug(f"Symbol validation failed for {symbol}: {e}")
        return False


def get_market_data(symbols):
    """Get market data for multiple symbols with rate limiting"""
    data = {}
    
    for symbol in symbols:
        try:
            time.sleep(0.1)
            ticker = exchange.fetch_ticker(symbol)
            
            data[symbol] = {
                'price': ticker.get('last', 0),
                'volume': ticker.get('quoteVolume', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'change': ticker.get('percentage', 0)
            }
        except Exception as e:
            logger.debug(f"Market data fetch failed for {symbol}: {e}")
            continue
    
    return data


def refresh_symbols():
    """Manually refresh symbol cache"""
    SYMBOL_CACHE['timestamp'] = None
    return get_live_symbols()


logger.info("✅ Scanner initialized - fetching LIVE coins from Bybit (with stablecoin + meme filter)")
