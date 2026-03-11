from typing import List, Dict
import re
import os
import sys
import json
from openai import OpenAI
from src.prompts import (
    refine_query_prompt,
    identify_relevant_files_prompt,
    generate_search_queries_prompt,
    analyze_project_context_prompt,
    generate_questions_prompt,
    github_search_query_prompt,
    decide_action_prompt
)

class LLMClient:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.api_key = None
        self.client = None
        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                print("Warning: OPENAI_API_KEY not set. LLM calls will fail.")
        
    def refine_user_query(self, user_question: str, project_context: str = "", file_structure: str = "") -> Dict:
        """
        Refines a vague user query into a technical information need.
        Returns a dict with 'intent', 'refined_question', 'keywords', and 'is_action_request'.
        """
        if self.provider == "mock":
            return {
                "intent": "Search for code related to the question.",
                "refined_question": user_question,
                "keywords": user_question.split(),
                "is_action_request": False
            }

        prompt = refine_query_prompt(user_question, project_context, file_structure)

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a helpful assistant. Return ONLY valid JSON."},
                              {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content.strip()
                
                # Robust JSON extraction
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    try:
                        data = json.loads(json_str)
                        # Ensure the key exists even if the LLM misses it
                        if "is_action_request" not in data:
                            data["is_action_request"] = False
                        return data
                    except json.JSONDecodeError:
                        pass # Fall through to default
                        
            except Exception as e:
                print(f"[Query Expansion] Warning: Failed to refine query: {e}")
        
        return {
            "intent": "General code search",
            "refined_question": user_question,
            "keywords": [],
            "is_action_request": False
        }

    def identify_relevant_files(self, user_question: str, file_structure: str, symbol_minimap: Dict = None) -> List[str]:
        """
        Skeleton-first analysis: Given a project file tree and an optional symbol minimap,
        identify the files most likely to contain the answer.
        """
        if self.provider == "mock":
            return []

        minimap_hint = ""
        if symbol_minimap:
            # Create a condensed version of the minimap for the prompt
            parts = []
            for path, entry in list(symbol_minimap.items())[:50]: # Cap to 50 files for prompt space
                # Handle both old format (list) and new format (dict with symbols/keywords)
                if isinstance(entry, list):
                    symbols = entry
                    file_keywords = []
                else:
                    symbols = entry.get("symbols", [])
                    file_keywords = entry.get("keywords", [])
                
                sym_strs = []
                for s in symbols[:10]: # Cap symbols per file
                    sig = s.get("signature", "")
                    doc = f" - {s['doc']}" if s.get("doc") else ""
                    sym_strs.append(f"{s['name']}{sig}{doc}")
                
                kw_str = f"  Keywords: {', '.join(file_keywords)}" if file_keywords else ""
                parts.append(f"### {path}\n" + "\n".join(f"  * {ss}" for ss in sym_strs) + (f"\n{kw_str}" if kw_str else ""))
            
            minimap_hint = f"\nSymbol MiniMap (Classes, Functions, Signatures, Keywords):\n" + "\n".join(parts) + "\n"

        prompt = identify_relevant_files_prompt(user_question, file_structure, minimap_hint)

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a helpful assistant. Return ONLY valid JSON."},
                              {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content.strip()
                
                # Robust JSON extraction
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        files = json.loads(json_str)
                        if isinstance(files, list):
                            print(f"[Skeleton] Identified {len(files)} relevant files: {files}")
                            return files[:8]
                    except json.JSONDecodeError as e:
                        print(f"[Skeleton] Warning: JSON Decode Error: {e}")
                else:
                    print(f"[Skeleton] Warning: No JSON list found in response: {content[:100]}...")
            except Exception as e:
                print(f"[Skeleton] Warning: Could not parse file list: {e}. Raw content: {content[:100]}...")
                return []

        return []

    def generate_search_queries(self, user_question: str, tool: str = "ripgrep", history: List[Dict] = None, project_context: str = "", file_structure: str = "") -> List[str]:
        """
        Generates search queries based on the user's question and optional history.
        Now accepts file_structure to generate more targeted queries.
        """
        if self.provider == "mock":
            words = user_question.split()
            return [w for w in words if len(w) > 3]

        # Format history for prompt
        history_str = ""
        if history:
            history_str = "Conversation History:\n" + "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history]) + "\n"

        structure_hint = ""
        if file_structure:
            structure_hint = f"""\nProject File Structure:
```
{file_structure[:2000]}
```
Use this structure to generate targeted queries. For example, if you see 'services/auth_service.py', search for function names or patterns likely in that file.\n"""
            
        if tool == "github":
             prompt = github_search_query_prompt(user_question)
        else:
             prompt = generate_search_queries_prompt(user_question, project_context, structure_hint, history_str)

        if self.provider == "openai" and self.client:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            content = content.replace("```", "").strip()
            return [line.strip() for line in content.splitlines() if line.strip()]

        return []

    def decide_action(self, user_question: str, context: str, project_structure: str = "", history: List[Dict] = None, available_tools: str = "") -> Dict:
        """
        Determines the next action in an agentic loop (either a tool call or a final answer).
        """
        if self.provider == "mock":
            return {"action": "final_answer", "content": "[Mock Final Answer from Action Loop]"}

        history_str = ""
        if history:
            history_str = "Conversation History:\n"
            for msg in history:
                history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = decide_action_prompt(user_question, context, history_str, project_structure, available_tools)

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", # Using full gpt-4o for tool reasoning
                    messages=[{"role": "system", "content": "You are an autonomous AI. Return ONLY a valid JSON object."},
                              {"role": "user", "content": prompt}]
                )
                
                content = response.choices[0].message.content.strip()
                # Robust JSON extraction
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    return json.loads(json_str)
                else:
                    return {"action": "final_answer", "content": content}
                    
            except Exception as e:
                print(f"[Action Loop] Error: {e}")
                return {"action": "final_answer", "content": f"Encountered error in loop: {e}"}

        return {"action": "final_answer", "content": "LLM Provider not configured."}

    def analyze_project_context(self, readme_content: str) -> str:
        """
        Analyzes the README content to extract key project details for search context.
        """
        if not readme_content:
            return ""
            
        if self.provider == "mock":
            return "Mock Project Context"

        prompt = analyze_project_context_prompt(readme_content)

        if self.provider == "openai" and self.client:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        return ""

    def generate_questions(self, context: str, num: int = 5) -> str:
        """
        Generates sample questions and answers based on the project context.
        """
        if self.provider == "mock":
            return "1. **Question**: What is this? \n   - **Answer**: A mock project."

        # Truncate context to safe limit (approx 15k tokens) to avoid errors
        # GPT-4o has 128k context, but we want to be cost-effective and safe.
        safe_context = context[:50000] 

        prompt = generate_questions_prompt(safe_context, num)

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating questions: {e}"

        return "LLM Provider not configured."
