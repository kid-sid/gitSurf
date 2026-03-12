import subprocess
import shutil
import os
from typing import List, Dict, Optional
import json

class SearchTool:
    def is_available(self) -> bool:
        return shutil.which(self.executable_path) is not None

    def __init__(self, executable_path: str = "rg"):
        self.executable_path = executable_path
        
        if not self.is_available():
            print(f"Warning: '{self.executable_path}' not found in PATH.")

    def search(self, query: str, search_path: str = ".", extra_args: Optional[List[str]] = None) -> List[Dict]:
        """
        Executes ripgrep with the given query in the search_path.
        Returns a list of results.
        """
        if not self.is_available():
            raise FileNotFoundError(f"ripgrep executable '{self.executable_path}' not found.")

        cmd = [self.executable_path, "--json", "-i", query, search_path]
        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        except Exception as e:
            return [{"error": str(e)}]

        parsed_results = []
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    match_data = data["data"]
                    file_path = match_data["path"]["text"].replace("\\", "/")
                    parsed_results.append({
                        "file": file_path,
                        "line_number": match_data["line_number"],
                        "content": match_data["lines"]["text"].strip()
                    })
            except json.JSONDecodeError:
                continue
        
        return parsed_results

    def search_and_chunk(self, query: str, search_path: str = ".", context_lines: int = 10) -> List[Dict]:
        """
        Search using ripgrep and return results with context lines, as cohesive chunks.
        """
        matches = self.search(query, search_path)
        chunks = []
        seen_chunks = set() # (file, start_line, end_line)

        for match in matches:
            file_path = match["file"]
            line_num = match["line_number"]
            
            start_line = max(1, line_num - context_lines)
            end_line = line_num + context_lines 
            
            chunk_id = (file_path, start_line, end_line)
            if chunk_id in seen_chunks:
                continue
            
            seen_chunks.add(chunk_id)
            
            try:
                abs_path = os.path.join(search_path, file_path)
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    actual_end = min(len(lines), end_line)
                    chunk_lines = lines[start_line-1:actual_end]
                    content = "".join(chunk_lines).strip()
                    
                    chunks.append({
                        "file": file_path,
                        "start_line": start_line,
                        "end_line": actual_end,
                        "content": f"File: {file_path} (lines {start_line}-{actual_end})\n\n{content}",
                        "source": "keyword",
                        "symbol": "",
                    })
            except Exception:
                continue
                
        return chunks