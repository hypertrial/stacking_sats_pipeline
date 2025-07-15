"""
Yahoo Finance data loader for stock, ETF, and cryptocurrency data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Import yfinance with fallback error handling
try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance is required for YFinanceLoader. Install it with: pip install yfinance"
    ) from None

# Logging configuration
# ---------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class YFinanceLoader:
    """Loader for Yahoo Finance data supporting multiple symbols."""

    DEFAULT_FILENAME = "yfinance_data.csv"

    # Popular symbol categories
    STOCKS = {
        # Tech Giants
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc. Class A",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "AMD": "Advanced Micro Devices Inc.",
        "INTC": "Intel Corporation",
        # Traditional Finance
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corporation",
        "WFC": "Wells Fargo & Company",
        "GS": "Goldman Sachs Group Inc.",
        "MS": "Morgan Stanley",
        "C": "Citigroup Inc.",
        "BRK-B": "Berkshire Hathaway Inc. Class B",
        "V": "Visa Inc.",
        "MA": "Mastercard Incorporated",
        "AXP": "American Express Company",
        # Energy
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        "SLB": "Schlumberger Limited",
        # Healthcare
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "UNH": "UnitedHealth Group Incorporated",
        "ABBV": "AbbVie Inc.",
        # Consumer
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble Company",
        "KO": "Coca-Cola Company",
        "PEP": "PepsiCo Inc.",
        "MCD": "McDonald's Corporation",
        # Industrial
        "BA": "Boeing Company",
        "CAT": "Caterpillar Inc.",
        "GE": "General Electric Company",
        "MMM": "3M Company",
    }

    CRYPTO = {
        # Major Cryptocurrencies
        "BTC-USD": "Bitcoin USD",
        "ETH-USD": "Ethereum USD",
        "BNB-USD": "Binance Coin USD",
        "XRP-USD": "XRP USD",
        "ADA-USD": "Cardano USD",
        "DOGE-USD": "Dogecoin USD",
        "SOL-USD": "Solana USD",
        "TRX-USD": "TRON USD",
        "AVAX-USD": "Avalanche USD",
        "DOT-USD": "Polkadot USD",
        # Layer 2 & DeFi
        "MATIC-USD": "Polygon USD",
        "LINK-USD": "Chainlink USD",
        "UNI-USD": "Uniswap USD",
        "AAVE-USD": "Aave USD",
        "COMP-USD": "Compound USD",
        # Legacy Crypto
        "LTC-USD": "Litecoin USD",
        "BCH-USD": "Bitcoin Cash USD",
        "ETC-USD": "Ethereum Classic USD",
        "XLM-USD": "Stellar USD",
        "XMR-USD": "Monero USD",
    }

    INDICES = {
        # US Stock Indices
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^RUT": "Russell 2000",
        "^NYA": "NYSE Composite",
        # Volatility & Fear
        "^VIX": "CBOE Volatility Index",
        "^VXN": "NASDAQ Volatility Index",
        # Interest Rates & Bonds
        "^TNX": "10-Year Treasury Yield",
        "^FVX": "5-Year Treasury Yield",
        "^TYX": "30-Year Treasury Yield",
        "^IRX": "3-Month Treasury Bill Yield",
        # International Indices
        "^FTSE": "FTSE 100 (UK)",
        "^GDAXI": "DAX (Germany)",
        "^FCHI": "CAC 40 (France)",
        "^N225": "Nikkei 225 (Japan)",
        "^HSI": "Hang Seng (Hong Kong)",
        "000001.SS": "Shanghai Composite (China)",
        # Commodities Indices
        "^GSCI": "Goldman Sachs Commodity Index",
        "DJP": "DJ UBS Commodity Index",
    }

    ETFS = {
        # Broad Market ETFs
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        # Precious Metals & Commodities
        "GLD": "SPDR Gold Trust",
        "SLV": "iShares Silver Trust",
        "PPLT": "Aberdeen Standard Platinum Shares ETF",
        "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        "DBA": "Invesco DB Agriculture Fund",
        # Bonds & Fixed Income
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "AGG": "iShares Core US Aggregate Bond ETF",
        "HYG": "iShares iBoxx High Yield Corporate Bond ETF",
        "TIP": "iShares TIPS Bond ETF",
        # Sector ETFs
        "XLF": "Financial Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLK": "Technology Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund",
        "XLI": "Industrial Select Sector SPDR Fund",
        "XLU": "Utilities Select Sector SPDR Fund",
        "XLP": "Consumer Staples Select Sector SPDR Fund",
        "XLY": "Consumer Discretionary Select Sector SPDR Fund",
        "XLB": "Materials Select Sector SPDR Fund",
        # Thematic & Growth ETFs
        "ARKK": "ARK Innovation ETF",
        "ARKQ": "ARK Autonomous Technology & Robotics ETF",
        "ARKW": "ARK Next Generation Internet ETF",
        "ICLN": "iShares Global Clean Energy ETF",
        "BOTZ": "Global X Robotics & Artificial Intelligence ETF",
        # Cryptocurrency ETFs
        "BITO": "ProShares Bitcoin Strategy ETF",
        "GBTC": "Grayscale Bitcoin Trust",
        "ETHE": "Grayscale Ethereum Trust",
        # International ETFs
        "EFA": "iShares MSCI EAFE ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "FXI": "iShares China Large-Cap ETF",
        "EWJ": "iShares MSCI Japan ETF",
        "EWG": "iShares MSCI Germany ETF",
        # Real Estate
        "VNQ": "Vanguard Real Estate ETF",
        "SCHH": "Schwab US REIT ETF",
        # Currency ETFs
        "UUP": "Invesco DB US Dollar Index Bullish Fund",
        "FXE": "Invesco CurrencyShares Euro Trust",
        "FXY": "Invesco CurrencyShares Japanese Yen Trust",
    }

    COMMODITIES = {
        # Precious Metals
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "PL=F": "Platinum Futures",
        "PA=F": "Palladium Futures",
        # Energy
        "CL=F": "Crude Oil Futures",
        "BZ=F": "Brent Crude Oil Futures",
        "NG=F": "Natural Gas Futures",
        "RB=F": "RBOB Gasoline Futures",
        "HO=F": "Heating Oil Futures",
        # Agricultural
        "ZC=F": "Corn Futures",
        "ZS=F": "Soybean Futures",
        "ZW=F": "Wheat Futures",
        "KC=F": "Coffee Futures",
        "CC=F": "Cocoa Futures",
        "SB=F": "Sugar Futures",
        "CT=F": "Cotton Futures",
        # Base Metals
        "HG=F": "Copper Futures",
        "ALI=F": "Aluminum Futures",
        # Livestock
        "LE=F": "Live Cattle Futures",
        "GF=F": "Feeder Cattle Futures",
        "HE=F": "Lean Hogs Futures",
    }

    # Forex pairs
    FOREX = {
        # Major Currency Pairs
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD",
        "USDCHF=X": "USD/CHF",
        "NZDUSD=X": "NZD/USD",
        # Cross Pairs
        "EURGBP=X": "EUR/GBP",
        "EURJPY=X": "EUR/JPY",
        "GBPJPY=X": "GBP/JPY",
        # Emerging Market Currencies
        "USDCNY=X": "USD/CNY (Chinese Yuan)",
        "USDINR=X": "USD/INR (Indian Rupee)",
        "USDBRL=X": "USD/BRL (Brazilian Real)",
        "USDMXN=X": "USD/MXN (Mexican Peso)",
        "USDZAR=X": "USD/ZAR (South African Rand)",
    }

    # All symbols combined
    ALL_SYMBOLS = {**STOCKS, **CRYPTO, **INDICES, **ETFS, **COMMODITIES, **FOREX}

    # Legacy symbol for backward compatibility
    SYMBOL = "BTC-USD"

    # Default interesting symbols for comprehensive market data
    DEFAULT_SYMBOLS = [
        # Crypto (primary focus)
        "BTC-USD",
        "ETH-USD",
        # Gold & Precious Metals
        "GLD",
        "GC=F",
        "SLV",
        # Major Stocks
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        # Traditional Finance
        "JPM",
        "BAC",
        "GS",
        "BRK-B",
        # Energy & Commodities
        "XLE",
        "CL=F",
        "USO",
        # International
        "EFA",
        "EEM",
        # Bonds
        "TLT",
        "AGG",
        # Currency
        "EURUSD=X",
        "USDJPY=X",
        # Volatility
        "^VIX",
    ]

    def __init__(
        self,
        data_dir: str | Path | None = None,
        symbols: list[str] | None = None,
        period: str = "max",
        interval: str = "1d",
    ):
        """
        Initialize Yahoo Finance loader.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory to store/load CSV files. If None, uses current file's parent directory.
        symbols : List[str], optional
            List of Yahoo Finance symbols to load. If None, defaults to BTC-USD.
        period : str, default "max"
            Time period to download. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval : str, default "1d"
            Data interval. Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)

        # Set symbols - default to BTC-USD for backward compatibility
        self.symbols = symbols or [self.SYMBOL]
        self.period = period
        self.interval = interval

        # Validate symbols
        if not isinstance(self.symbols, list):
            raise ValueError("symbols must be a list of strings")

        # Log unknown symbols as warnings (not errors since Yahoo Finance supports
        # many symbols)
        unknown_symbols = [sym for sym in self.symbols if sym not in self.ALL_SYMBOLS]
        if unknown_symbols:
            logging.warning(
                "Unknown symbols (may still be valid): %s. Known symbols: %s",
                unknown_symbols,
                list(self.ALL_SYMBOLS.keys()),
            )

    def load_from_web(self) -> pd.DataFrame:
        """
        Download Yahoo Finance data directly into memory.

        Returns
        -------
        pd.DataFrame
            DataFrame with financial data, indexed by datetime, with columns like
            'AAPL_Close', 'AAPL_Volume', etc.
        """
        logging.info(
            "Downloading %d symbols from Yahoo Finance: %s (period=%s, interval=%s)",
            len(self.symbols),
            self.symbols,
            self.period,
            self.interval,
        )

        frames = []

        for symbol in self.symbols:
            logging.info("Fetching symbol: %s", symbol)

            try:
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=self.period, interval=self.interval)

                if data.empty:
                    logging.warning("No data returned for symbol %s", symbol)
                    continue

                # Reset index to get datetime as column, then set it back
                data = data.reset_index()

                # Handle different datetime column names
                date_col = None
                for col in ["Date", "Datetime", "date", "datetime"]:
                    if col in data.columns:
                        date_col = col
                        break

                if date_col is None:
                    logging.warning("No date column found for symbol %s", symbol)
                    continue

                # Convert to UTC timezone-aware datetime
                data[date_col] = pd.to_datetime(data[date_col])

                # Handle timezone conversion first
                if data[date_col].dt.tz is None:
                    # If no timezone, localize to UTC
                    data[date_col] = data[date_col].dt.tz_localize("UTC")
                else:
                    # If already timezone-aware, convert to UTC
                    data[date_col] = data[date_col].dt.tz_convert("UTC")

                # Normalize to midnight UTC AFTER timezone conversion to avoid hour shifts
                data[date_col] = data[date_col].dt.normalize()

                # Set as index
                data.set_index(date_col, inplace=True)
                data.index.name = "time"

                # Add symbol suffix to columns
                data.columns = [f"{symbol}_{col}" for col in data.columns]

                # Remove duplicates and sort
                data = data.loc[~data.index.duplicated(keep="last")].sort_index()

                frames.append(data)
                logging.info("Loaded symbol %s (%d rows)", symbol, len(data))

            except Exception as e:
                logging.error("Failed to download symbol %s: %s", symbol, e)
                continue

        if not frames:
            raise ValueError("No valid data retrieved for any symbol")

        # Combine all symbols into a single DataFrame
        combined_df = pd.concat(frames, axis=1, sort=True)

        # Forward fill missing values to handle holidays/weekends
        combined_df = combined_df.ffill()

        logging.info(
            "Combined Yahoo Finance data loaded (%d rows, %d symbols)",
            len(combined_df),
            len(frames),
        )
        self._validate_data(combined_df)

        return combined_df

    def load_symbols(self, symbols: list[str]) -> pd.DataFrame:
        """
        Load specific symbols (convenience method).

        Parameters
        ----------
        symbols : List[str]
            List of Yahoo Finance symbols to load.

        Returns
        -------
        pd.DataFrame
            DataFrame with requested symbols data.
        """
        original_symbols = self.symbols
        self.symbols = symbols
        try:
            return self.load_from_web()
        finally:
            self.symbols = original_symbols

    def load_category(self, category: str) -> pd.DataFrame:
        """
        Load all symbols from a specific category.

        Parameters
        ----------
        category : str
            Category name: 'stocks', 'crypto', 'indices', 'etfs', 'commodities', 'forex'

        Returns
        -------
        pd.DataFrame
            DataFrame with category symbols data.
        """
        category_map = {
            "stocks": self.STOCKS,
            "crypto": self.CRYPTO,
            "indices": self.INDICES,
            "etfs": self.ETFS,
            "commodities": self.COMMODITIES,
            "forex": self.FOREX,
        }

        if category not in category_map:
            raise ValueError(
                f"Unknown category: {category}. Available: {list(category_map.keys())}"
            )

        return self.load_symbols(list(category_map[category].keys()))

    def extract_to_csv(self, local_path: str | Path | None = None) -> Path:
        """
        Download Yahoo Finance data and store them locally as CSV.

        Parameters
        ----------
        local_path : str or Path, optional
            Destination CSV path. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        Path
            Path where the data was saved.
        """
        if local_path is None:
            local_path = self.data_dir / self.DEFAULT_FILENAME
        else:
            local_path = Path(local_path)

        # Use the in-memory loader and save to CSV
        df = self.load_from_web()
        df.to_csv(local_path)
        logging.info("Saved Yahoo Finance data ➜ %s", local_path)

        return local_path

    def extract_to_parquet(self, local_path: str | Path | None = None) -> Path:
        """
        Download Yahoo Finance data and store them locally as Parquet.

        Parameters
        ----------
        local_path : str or Path, optional
            Destination Parquet path. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        Path
            Path where the data was saved.
        """
        if local_path is None:
            # Change extension from .csv to .parquet
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            local_path = self.data_dir / parquet_filename
        else:
            local_path = Path(local_path)

        # Use the in-memory loader and save to Parquet
        df = self.load_from_web()
        df.to_parquet(local_path)
        logging.info("Saved Yahoo Finance data ➜ %s", local_path)

        return local_path

    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load Yahoo Finance data from a local CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the CSV file. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with financial data, indexed by datetime.
        """
        if path is None:
            path = self.data_dir / self.DEFAULT_FILENAME
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "Yahoo Finance data file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_csv(path)

        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        # Convert naive datetime index to UTC timezone-aware and normalize to midnight
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Normalize to midnight UTC to ensure daily-level data consistency
        df.index = df.index.normalize()

        self._validate_data(df)
        return df

    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load Yahoo Finance data from a local Parquet file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the Parquet file. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with financial data, indexed by datetime.
        """
        if path is None:
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            path = self.data_dir / parquet_filename
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "Yahoo Finance Parquet file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_parquet(path)

        df = pd.read_parquet(path)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        # Convert naive datetime index to UTC timezone-aware and normalize to midnight
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Normalize to midnight UTC to ensure daily-level data consistency
        df.index = df.index.normalize()

        self._validate_data(df)
        return df

    def load(
        self,
        use_memory: bool = True,
        path: str | Path | None = None,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Load Yahoo Finance data either from memory (web) or from a local file.

        Parameters
        ----------
        use_memory : bool, default True
            If True, loads data directly from web into memory.
            If False, loads from local file (downloads if doesn't exist).
        path : str or Path, optional
            Path to the file. Only used if use_memory=False.
        file_format : str, default "csv"
            File format to use when use_memory=False. Options: "csv", "parquet".

        Returns
        -------
        pd.DataFrame
            DataFrame with financial data, indexed by datetime.
        """
        if use_memory:
            logging.info("Loading Yahoo Finance data directly from web...")
            return self.load_from_web()
        else:
            if file_format.lower() == "parquet":
                return self.load_from_parquet(path)
            else:
                return self.load_from_file(path)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Basic sanity‑check on the Yahoo Finance dataframe.
        """
        if df.empty:
            raise ValueError("Yahoo Finance dataframe is empty")

        # Check for at least one price column (Close, Open, High, Low, Adj Close)
        price_cols = [
            col
            for col in df.columns
            if any(
                price_type in col for price_type in ["Close", "Open", "High", "Low", "Adj Close"]
            )
        ]
        if not price_cols:
            raise ValueError("No price columns found in Yahoo Finance data")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware")

        # Check if timezone is UTC (accept both pytz.UTC and pandas UTC)
        if str(df.index.tz) != "UTC":
            raise ValueError("DatetimeIndex must be in UTC timezone")

    @classmethod
    def get_available_symbols(cls) -> dict[str, str]:
        """Get all available symbols with descriptions."""
        return cls.ALL_SYMBOLS.copy()

    @classmethod
    def get_symbols_by_category(cls, category: str) -> dict[str, str]:
        """Get symbols for a specific category."""
        category_map = {
            "stocks": cls.STOCKS,
            "crypto": cls.CRYPTO,
            "indices": cls.INDICES,
            "etfs": cls.ETFS,
            "commodities": cls.COMMODITIES,
            "forex": cls.FOREX,
        }

        if category not in category_map:
            raise ValueError(
                f"Unknown category: {category}. Available: {list(category_map.keys())}"
            )

        return category_map[category].copy()


# Convenience functions for backward compatibility
def load_btc_data_from_web() -> pd.DataFrame:
    """Load Yahoo Finance BTC data from web (backward compatibility)."""
    loader = YFinanceLoader(symbols=["BTC-USD"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "BTC-USD_Close" in df.columns:
        df = df.rename(columns={"BTC-USD_Close": "PriceUSD"})
    return df


def extract_btc_data_to_csv(local_path: str | Path | None = None) -> None:
    """Extract Yahoo Finance BTC data to CSV (backward compatibility)."""
    loader = YFinanceLoader(symbols=["BTC-USD"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "BTC-USD_Close" in df.columns:
        df = df.rename(columns={"BTC-USD_Close": "PriceUSD"})

    if local_path is None:
        local_path = loader.data_dir / "btc_yfinance.csv"
    else:
        local_path = Path(local_path)

    df.to_csv(local_path)
    logging.info("Saved Yahoo Finance BTC data ➜ %s", local_path)


def extract_btc_data_to_parquet(local_path: str | Path | None = None) -> None:
    """Extract Yahoo Finance BTC data to Parquet (backward compatibility)."""
    loader = YFinanceLoader(symbols=["BTC-USD"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "BTC-USD_Close" in df.columns:
        df = df.rename(columns={"BTC-USD_Close": "PriceUSD"})

    if local_path is None:
        local_path = loader.data_dir / "btc_yfinance.parquet"
    else:
        local_path = Path(local_path)

    df.to_parquet(local_path)
    logging.info("Saved Yahoo Finance BTC data ➜ %s", local_path)


def load_stock_data_from_web(symbol: str = "AAPL") -> pd.DataFrame:
    """Load Yahoo Finance stock data from web."""
    loader = YFinanceLoader(symbols=[symbol])
    return loader.load_from_web()


def load_crypto_data_from_web(symbols: list[str] | None = None) -> pd.DataFrame:
    """Load Yahoo Finance crypto data from web."""
    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD"]
    loader = YFinanceLoader(symbols=symbols)
    return loader.load_from_web()
