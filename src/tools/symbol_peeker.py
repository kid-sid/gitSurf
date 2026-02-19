import os
import sys
from typing import List, Dict, Optional

# Add root directory to sys.path to allow relative imports when run directly
try:
    from src.tools.symbol_extractor import SymbolExtractor
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.tools.symbol_extractor import SymbolExtractor

class SymbolPeeker:
    """
    Provides "Peek Definition" (F12) functionality.
    Extracts the actual source code block for a given symbol (class/function).
    """

    def __init__(self, symbol_index: Dict[str, List[Dict]], root_path: str):
        self.symbol_index = symbol_index
        self.root_path = os.path.abspath(root_path)

    def peek_symbol(self, symbol_name: str) -> List[Dict]:
        """
        Finds all occurrences of a symbol name and returns their code blocks.
        """
        results = []
        for rel_path, symbols in self.symbol_index.items():
            for sym in symbols:
                # Top level functions/classes
                if sym.get("name") == symbol_name:
                    content = self._read_lines(rel_path, sym["start_line"], sym["end_line"])
                    if content:
                        results.append({
                            "file": rel_path,
                            "type": sym["type"],
                            "name": sym["name"],
                            "start_line": sym["start_line"],
                            "end_line": sym["end_line"],
                            "content": content
                        })
                
                # Check methods within classes
                if sym.get("type") == "class":
                    for method in sym.get("methods", []):
                        if method.get("name") == symbol_name:
                            content = self._read_lines(rel_path, method["start_line"], method["end_line"])
                            if content:
                                results.append({
                                    "file": rel_path,
                                    "type": "method",
                                    "name": f"{sym['name']}.{method['name']}",
                                    "start_line": method["start_line"],
                                    "end_line": method["end_line"],
                                    "content": content
                                })
        return results

    def _read_lines(self, rel_path: str, start: int, end: int) -> Optional[str]:
        """Reads specific lines from a file."""
        abs_path = os.path.join(self.root_path, rel_path)
        if not os.path.exists(abs_path):
            return None
        
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                # AST/Regex line numbers are usually 1-indexed
                target_lines = lines[start-1:end]
                return "".join(target_lines)
        except Exception as e:
            print(f"[SymbolPeeker] Error reading {rel_path}: {e}")
            return None

if __name__ == "__main__":
    # Internal Test Mode
    import json
    # Use relative pathing to find the cache
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache_path = os.path.join(base_dir, ".cache", "symbols", "symbol_index.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        peeker = SymbolPeeker(index, base_dir)
        # Try to peek at SymbolExtractor as a test
        matches = peeker.peek_symbol("SymbolExtractor")
        if matches:
            for m in matches:
                print(f"\n--- PEEK: {m['file']} ({m['start_line']}-{m['end_line']}) ---")
                print(m['content'][:500] + ("..." if len(m['content']) > 500 else ""))
        else:
            print("No matches found for 'SymbolExtractor'.")
    else:
        print(f"Cache not found at: {cache_path}. Run gitSurf first to index symbols.")
