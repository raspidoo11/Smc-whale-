import pandas as pd
import logging
import time
from exchange import get_exchange
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

exchange = get_exchange()

# Meme coin patterns to filter OUT
MEME_PATTERNS = {
    'BABY', 'SAFE', 'MOON', 'LUNC', 'INU', 'COIN', 'TOKEN',
    '1000', '10000', 'ELON', 'DOGE', 'SHIB', 'PEPE', 'FLOKI',
    'X', 'V2', 'OLD', 'CUMROCKET', 'PUSSY', 'CUMMIES'
}

# Tier 1: Always include (top tier coins)
TIER1 = {'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'LINK'}

# Cache for symbol list (update every 5 minutes)
SYMBOL_CACHE = {
    'symbols': [],
    'timestamp': None,
    'cache_duration': 300  # 5 minutes
}

def is_meme_coin(symbol):
    """Check if symbol matches meme coin patterns"""
    base = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '').replace('USDT', '')
    
    # Check blacklist patterns
    for pattern in MEME_PATTERNS:
        if pattern in base:
            return True
    
    return False

def get_live_symbols(limit=30):
    """
    Fetch LIVE trading pairs from Bybit
    Filter by volume and quality
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
        
        # Get all trading pairs
        markets = exchange.load_markets()
        
        # Filter USDT pairs only
        usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('USDT')]
        
        logger.info(f"Found {len(usdt_pairs)} USDT trading pairs")
        
        # Get volume data
        volume_data = []
        
        for symbol in usdt_pairs:
            try:
                time.sleep(0.05)  # Rate limit: 0.05s between API calls
                
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
        tier1_symbols = [f"{coin}USDT" for coin in TIER1 if f"{coin}USDT" in top_symbols]
        if len(tier1_symbols) < len(TIER1):
            # Add missing Tier 1 coins
            for coin in TIER1:
                symbol = f"{coin}USDT"
                if symbol not in top_symbols:
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
        # Fallback to cache if available
        if SYMBOL_CACHE['symbols']:
            logger.info("Using cached symbols (fetch failed)")
            return SYMBOL_CACHE['symbols'][:limit]
        return []

def get_ohlcv(symbol, timeframe, limit):
    """
    Fetch OHLCV data with error handling
    Implements rate limiting to avoid API blocks
    """
    try:
        # Add small delay to avoid rate limiting
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
        
        # Clean data
        df = df[(df['volume'] > 0) & (df['close'] > 0)]
        
        if len(df) < limit * 0.8:  # Need at least 80% of requested data
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
        if volume < 500000:  # Min $500k volume
            return False
        
        bid = ticker.get('bid', 0)
        ask = ticker.get('ask', 0)
        if bid <= 0 or ask <= 0:
            return False
        
        return True
    
    except Exception as e:
        logger.debug(f"Symbol validation failed for {symbol}: {e}")
        return False

def get_market_data(symbols):
    """
    Get market data for multiple symbols
    Rate limited to 0.1s per symbol
    """
    data = {}
    
    for symbol in symbols:
        try:
            time.sleep(0.1)  # Rate limit: 0.1s between API calls
            
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

logger.info("✅ Scanner initialized - fetching LIVE coins from Bybit")
