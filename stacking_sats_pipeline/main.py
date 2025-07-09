# Import all necessary functions and constants
import argparse
import os
from pathlib import Path

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    # Look for .env file in current working directory first, then package directory
    env_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent / ".env",  # Package directory
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    pass

from .config import BACKTEST_END, BACKTEST_START
from .data.data_loader import MultiSourceDataLoader


def extract_all_data(file_format: str = "csv", output_dir: str | Path | None = None) -> None:
    """
    Extract all available data sources to a single merged CSV or Parquet file.

    Parameters
    ----------
    file_format : str, default "csv"
        File format to extract data to. Options: "csv", "parquet".
    output_dir : str or Path, optional
        Directory to save file to. If None, saves to current directory.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"EXTRACTING ALL DATA TO MERGED {file_format.upper()} FILE")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Determine available sources
    loader = MultiSourceDataLoader()
    available_sources = loader.get_available_sources()

    # Always try to include these sources if available
    sources_to_load = []
    if "coinmetrics" in available_sources:
        sources_to_load.append("coinmetrics")
        print("✅ Bitcoin price data (CoinMetrics) - available")
    else:
        print("❌ Bitcoin price data (CoinMetrics) - not available")

    if "feargreed" in available_sources:
        sources_to_load.append("feargreed")
        print("✅ Fear & Greed Index data - available")
    else:
        print("❌ Fear & Greed Index data - not available")

    # Check for FRED API key
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        print("⚠️  FRED_API_KEY environment variable not found.")
        print("   To include FRED data, set your API key: export FRED_API_KEY=your_key_here")
        print("   Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    else:
        if "fred" in available_sources:
            sources_to_load.append("fred")
            print("✅ U.S. Dollar Index data (FRED) - available")
        else:
            print("❌ U.S. Dollar Index data (FRED) - not available")

    if not sources_to_load:
        print("❌ No data sources are available for extraction.")
        return

    try:
        print(f"\n📊 Loading and merging data from {len(sources_to_load)} sources...")

        # Load and merge all available data sources
        merged_df = loader.load_and_merge(sources_to_load, use_memory=True)

        # Filter data to match config date ranges
        print(f"🗓️  Filtering data to config date range: {BACKTEST_START} to {BACKTEST_END}")
        start_date = BACKTEST_START
        end_date = BACKTEST_END

        # Filter the dataframe to the specified date range
        original_shape = merged_df.shape
        merged_df = merged_df.loc[start_date:end_date]
        filtered_shape = merged_df.shape

        # Handle formatting safely for both real data and mocked data
        try:
            original_rows = int(original_shape[0])
            filtered_rows = int(filtered_shape[0])
            rows_removed = original_rows - filtered_rows
            print(
                f"📊 Data filtered: {original_rows:,} → {filtered_rows:,} rows "
                f"({rows_removed:,} rows removed)"
            )
        except (TypeError, ValueError):
            # Fallback for tests or unexpected data types
            print(f"📊 Data filtered: {original_shape[0]} → {filtered_shape[0]} rows")

        # Determine output filename
        if file_format.lower() == "parquet":
            output_file = output_dir / "merged_crypto_data.parquet"
            merged_df.to_parquet(output_file)
        else:
            output_file = output_dir / "merged_crypto_data.csv"
            merged_df.to_csv(output_file)

        # Calculate file size
        file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB

        print(f"\n{'=' * 60}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ Successfully merged {len(sources_to_load)} data sources:")
        for source in sources_to_load:
            print(f"   • {source}")
        print(f"\n📁 Output file: {output_file.name} ({file_size:.1f} MB)")
        print(f"📊 Format: {file_format.upper()}")
        try:
            rows = int(merged_df.shape[0])
            cols = int(merged_df.shape[1])
            print(f"📈 Data shape: {rows:,} rows × {cols} columns")
        except (TypeError, ValueError):
            print(f"📈 Data shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
        print(f"📅 Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"\n✅ All data extracted to: {output_dir}")

    except Exception as e:
        print(f"❌ Failed to extract and merge data: {e}")
        raise


def main():
    """
    Main function to extract and merge data from multiple sources.
    """
    parser = argparse.ArgumentParser(
        description="Extract and merge cryptocurrency and financial data from multiple sources"
    )
    parser.add_argument(
        "--extract-data",
        choices=["csv", "parquet"],
        default="csv",
        help="Extract all data sources to specified format (default: csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for extracted data files (default: current directory)",
    )

    args = parser.parse_args()

    # Handle data extraction
    extract_all_data(file_format=args.extract_data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
