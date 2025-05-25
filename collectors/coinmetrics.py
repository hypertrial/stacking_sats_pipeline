# ---------------------------
# Extract BTC data from CoinMetrics and save locally
# ---------------------------
import pandas as pd 
import logging
import os
from datetime import datetime
import requests
from io import StringIO
import argparse
import sys
from typing import Optional, Dict, Any, Union, List, Tuple

# Get logger
logger = logging.getLogger(__name__)

def extract_data(url: str = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv") -> Optional[pd.DataFrame]:
    """
    Extract Bitcoin data from CoinMetrics GitHub repository
    
    Args:
        url: URL to the Bitcoin CSV data
        
    Returns:
        DataFrame containing Bitcoin data or None if extraction failed
    """
    try:
        logger.info(f"Downloading BTC data from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        logger.info("Parsing CSV content")
        btc_df = pd.read_csv(StringIO(response.text), low_memory=False)
        
        # Process timestamps
        btc_df['time'] = pd.to_datetime(btc_df['time']).dt.normalize()
        btc_df['time'] = btc_df['time'].dt.tz_localize(None)
        btc_df.set_index('time', inplace=True)
        
        logger.info(f"Successfully extracted {len(btc_df)} rows of BTC data")
        return btc_df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def save_data(df: Optional[pd.DataFrame], csv_path: Optional[str] = None, parquet_path: Optional[str] = None) -> bool:
    """
    Save Bitcoin data to CSV and/or Parquet format
    
    Args:
        df: DataFrame containing Bitcoin data
        csv_path: Path to save CSV file (None to skip)
        parquet_path: Path to save Parquet file (None to skip)
        
    Returns:
        True if at least one file was saved successfully, False otherwise
    """
    if df is None or df.empty:
        logger.error("No data to save")
        return False
    
    success = False
    
    if csv_path:
        try:
            logger.info(f"Saving data to CSV: {csv_path}")
            df.to_csv(csv_path)
            logger.info(f"Successfully saved CSV to {csv_path}")
            success = True
        except Exception as e:
            logger.error(f"Error saving CSV file: {e}")
    
    if parquet_path:
        try:
            logger.info(f"Saving data to Parquet: {parquet_path}")
            df.to_parquet(parquet_path)
            logger.info(f"Successfully saved Parquet to {parquet_path}")
            success = True
        except ImportError:
            logger.error("pyarrow or fastparquet package is required to save Parquet files")
            logger.error("Install with: pip install pyarrow")
        except Exception as e:
            logger.error(f"Error saving Parquet file: {e}")
    
    return success

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download Bitcoin data from CoinMetrics')
    parser.add_argument('--csv', dest='csv_path', default='btc_data.csv',
                        help='Path to save CSV file (default: btc_data.csv, set to "none" to skip)')
    parser.add_argument('--parquet', dest='parquet_path', default='btc_data.parquet',
                        help='Path to save Parquet file (default: btc_data.parquet, set to "none" to skip)')
    parser.add_argument('--url', dest='url', 
                        default='https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv',
                        help='URL to Bitcoin data CSV')
    
    args = parser.parse_args()
    
    # Handle "none" string to convert to None
    if args.csv_path.lower() == "none":
        args.csv_path = None
    if args.parquet_path.lower() == "none":
        args.parquet_path = None
        
    return args

def main() -> int:
    """Main function to run the script"""
    args = parse_arguments()
    
    logger.info("Starting Bitcoin data extraction")
    btc_df = extract_data(args.url)
    
    if btc_df is not None:
        save_data(btc_df, args.csv_path, args.parquet_path)
        logger.info(f"Data shape: {btc_df.shape}")
        logger.info(f"Date range: {btc_df.index.min()} to {btc_df.index.max()}")
    else:
        logger.error("Failed to extract Bitcoin data")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)