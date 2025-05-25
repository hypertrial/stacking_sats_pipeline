#!/usr/bin/env python
"""
Script to run static type checking using mypy
"""
import os
import sys
import subprocess
import argparse
from typing import List, Tuple, Dict, Optional, Any

def run_mypy(files_or_dirs: List[str], config_file: Optional[str] = None) -> Tuple[bool, str]:
    """
    Run mypy on specified files or directories
    
    Args:
        files_or_dirs: List of files or directories to check
        config_file: Path to mypy config file
        
    Returns:
        Tuple of (success, output)
    """
    cmd = ["mypy"]
    
    # Add config file if specified
    if config_file:
        cmd.extend(["--config-file", config_file])
    
    # Add files/directories to check
    cmd.extend(files_or_dirs)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, f"Error running mypy: {e}"

def find_config_file() -> Optional[str]:
    """
    Find mypy config file in common locations
    
    Returns:
        Path to config file or None if not found
    """
    config_paths = [
        "mypy.ini",
        ".mypy.ini",
        "pyproject.toml",
        os.path.join("config", "mypy.ini"),
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            return path
    
    return None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run mypy type checking")
    parser.add_argument(
        "paths", 
        nargs="*", 
        default=["main.py", "collectors"],
        help="Files or directories to check (default: main.py and collectors)"
    )
    parser.add_argument(
        "--config", 
        "-c", 
        dest="config_file",
        help="Path to mypy config file (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Show more detailed output"
    )
    
    return parser.parse_args()

def main() -> int:
    """
    Main function to run type checking
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Use specified config file or try to find one
    config_file = args.config_file or find_config_file()
    
    if args.verbose and config_file:
        print(f"Using mypy config file: {config_file}")
    
    # Run mypy
    success, output = run_mypy(args.paths, config_file)
    
    # Print output
    print(output)
    
    # Print summary
    if success:
        print("\n✅ Type checking passed!")
        return 0
    else:
        print("\n❌ Type checking failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 