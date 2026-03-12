import os
from typing import Optional

class FileEditorTool:
    """
    Allows the agent to read, write, and modify files.
    Includes safety checks to prevent path traversal outside the root directory.
    """

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def _get_abs_path(self, rel_path: str) -> str:
        """Returns the absolute path, ensuring it's within the root_path to prevent path traversal."""
        abs_path = os.path.abspath(os.path.join(self.root_path, rel_path))
        if not abs_path.startswith(self.root_path):
            raise ValueError(f"Path restricted: {rel_path} resolves outside of root directory.")
        return abs_path

    def read_file(self, rel_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Reads a file, optionally by line range (1-indexed)."""
        try:
            abs_path = self._get_abs_path(rel_path)
            if not os.path.exists(abs_path):
                return f"[Error] File not found: {rel_path}"
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if start_line is not None and end_line is not None:
                # 1-indexed to 0-indexed
                target_lines = lines[start_line - 1 : end_line]
                return "".join(target_lines)
            
            return "".join(lines)
        except Exception as e:
            return f"[Error] Expected to read file: {e}"

    def write_file(self, rel_path: str, content: str) -> str:
        """Overwrites or creates a new file."""
        try:
            abs_path = self._get_abs_path(rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"[Success] Wrote file: {rel_path}"
        except Exception as e:
            return f"[Error] Failed to write file: {e}"

    def replace_in_file(self, rel_path: str, target: str, replacement: str) -> str:
        """Replaces exactly `target` with `replacement` in the file."""
        try:
            abs_path = self._get_abs_path(rel_path)
            if not os.path.exists(abs_path):
                return f"[Error] File not found: {rel_path}"
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if target not in content:
                return f"[Error] Target string not found in {rel_path}"
                
            new_content = content.replace(target, replacement)
            
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return f"[Success] Replaced content in {rel_path}"
        except Exception as e:
            return f"[Error] Failed to replace in file: {e}"

    def delete_file(self, rel_path: str) -> str:
        """Deletes a file from the file system."""
        try:
            abs_path = self._get_abs_path(rel_path)
            if not os.path.exists(abs_path):
                return f"[Error] File not found: {rel_path}"
                
            os.remove(abs_path)
            return f"[Success] Deleted file: {rel_path}"
        except Exception as e:
            return f"[Error] Failed to delete file: {e}"

if __name__ == "__main__":
    # Simple internal test
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tool = FileEditorTool(base_dir)
    test_file = "test_editor.txt"
    
    print(tool.write_file(test_file, "Hello World\nLine 2\nLine 3\n"))
    print(tool.read_file(test_file))
    print(tool.replace_in_file(test_file, "World", "Agent"))
    print(tool.read_file(test_file))
    
    # Cleanup
    try:
        os.remove(os.path.join(base_dir, test_file))
        print(f"[Cleanup] Removed {test_file}")
    except:
        pass
