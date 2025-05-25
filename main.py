#!/usr/bin/env python
# ---------------------------
# Main script to run all data collectors and merge results
# ---------------------------
import os
import importlib.util
import pandas as pd
import logging
from datetime import datetime
import sys
from typing import Dict, Optional, Callable, Any, List, Union, TypeVar, Tuple, cast
from types import ModuleType

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration in case other modules configure logging
)
logger = logging.getLogger(__name__)

# Type definitions
ExtractFunc = Callable[[], Optional[pd.DataFrame]]

def load_collector_module(file_path: str) -> Optional[ModuleType]:
    """
    Load a Python module from file path
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Loaded module or None if loading failed
    """
    try:
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading module {file_path}: {e}")
        return None

def find_data_extraction_function(module: ModuleType) -> Optional[ExtractFunc]:
    """
    Find the function that extracts data in a collector module
    
    Args:
        module: The loaded Python module
        
    Returns:
        The function that returns a DataFrame or None if not found
    """
    # Look for the standard extract_data function first
    extract_func = getattr(module, 'extract_data', None)
    
    # For backward compatibility, check for alternative names
    if extract_func is None:
        for func_name in ['extract_btc_data', 'get_data', 'collect_data', 'fetch_data']:
            extract_func = getattr(module, func_name, None)
            if extract_func is not None:
                break
    
    return cast(Optional[ExtractFunc], extract_func)

def run_collectors(collectors_dir: str = 'collectors') -> Dict[str, pd.DataFrame]:
    """
    Run all collector scripts in the specified directory
    
    Args:
        collectors_dir: Directory containing collector scripts
        
    Returns:
        Dictionary mapping collector names to DataFrames
    """
    results: Dict[str, pd.DataFrame] = {}
    
    # Ensure the collectors directory exists
    collectors_dir = os.path.abspath(collectors_dir)
    if not os.path.exists(collectors_dir):
        logger.error(f"Collectors directory '{collectors_dir}' not found")
        return results
    
    # Find all Python files in the collectors directory
    collector_files: List[str] = [
        os.path.join(collectors_dir, f) for f in os.listdir(collectors_dir)
        if f.endswith('.py') and not f.startswith('__')
    ]
    
    logger.info(f"Found {len(collector_files)} collector scripts")
    
    # Run each collector
    for file_path in collector_files:
        collector_name = os.path.basename(file_path).replace('.py', '')
        logger.info(f"Running collector: {collector_name}")
        
        # Load the module
        module = load_collector_module(file_path)
        if not module:
            continue
        
        # Find the data extraction function
        extract_func = find_data_extraction_function(module)
        if not extract_func:
            logger.warning(f"Could not find data extraction function in {collector_name}")
            continue
        
        # Call the extraction function
        try:
            data = extract_func()
            if isinstance(data, pd.DataFrame) and not data.empty:
                logger.info(f"Successfully collected data from {collector_name}: {len(data)} rows")
                results[collector_name] = data
            else:
                logger.warning(f"Collector {collector_name} did not return valid data")
        except Exception as e:
            logger.error(f"Error running collector {collector_name}: {e}")
    
    return results

def merge_data(data_dict: Dict[str, pd.DataFrame], output_file: str = 'stacking_sats_data.parquet') -> bool:
    """
    Merge all collected data into a single parquet file
    
    Args:
        data_dict: Dictionary mapping collector names to DataFrames
        output_file: Path to save the merged data
        
    Returns:
        True if successful, False otherwise
    """
    if not data_dict:
        logger.error("No data to merge")
        return False
    
    try:
        # Create a directory for the merged data if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Add source column to each DataFrame
        merged_dfs: List[pd.DataFrame] = []
        for source, df in data_dict.items():
            df_copy = df.copy()
            if not df_copy.index.name:
                df_copy.index.name = 'time'
            df_copy = df_copy.reset_index()
            df_copy['source'] = source
            merged_dfs.append(df_copy)
        
        # Concatenate all DataFrames
        merged_data = pd.concat(merged_dfs, axis=0, ignore_index=True)
        
        # Save to parquet
        logger.info(f"Saving merged data to {output_file}")
        merged_data.to_parquet(output_file)
        
        logger.info(f"Successfully saved merged data: {len(merged_data)} rows from {len(data_dict)} sources")
        
        # Print column information at the end
        print("\nColumn information for merged data:")
        print(f"Total columns: {len(merged_data.columns)}")
        print(f"Columns: {list(merged_data.columns)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main() -> int:
    """Main function to run all collectors and merge data"""
    # Ensure data directory exists
    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(data_dir, f"merged_data_{timestamp}.parquet")
    
    logger.info("Starting data collection process")
    logger.info(f"Data will be saved to: {output_file}")
    
    # Run all collectors
    data_dict = run_collectors()
    
    if not data_dict:
        logger.error("No data collected from any source")
        return 1
    
    # Merge and save data
    success = merge_data(data_dict, output_file)
    
    if success:
        logger.info(f"Data collection completed successfully")
        return 0
    else:
        logger.error("Data collection failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 