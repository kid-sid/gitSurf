import os
import sys
import json
from src.tools.symbol_extractor import SymbolExtractor

def build_local_symbol_index():
    print("Building symbol index for local project...")
    extractor = SymbolExtractor(cache_dir=".cache/symbols")
    # Index the current directory
    symbols = extractor.extract_from_directory(".", force_rebuild=True)
    print(f"Successfully indexed {len(symbols)} files.")
    
    index_path = os.path.abspath(".cache/symbols/symbol_index.json")
    if os.path.exists(index_path):
        print(f"Symbol index saved to: {index_path}")
    else:
        print("Error: Symbol index file not found after extraction.")

if __name__ == "__main__":
    # Ensure src is in path
    sys.path.append(os.getcwd())
    build_local_symbol_index()
