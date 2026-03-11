from typing import List, Dict, Optional

def refine_query_prompt(user_question: str, project_context: str, file_structure: str) -> str:
    return f"""You are a senior software architect. A developer has asked a question or given a command about a codebase. 
Your goal is to translate this request into a structured "Information Need" that can guide a search engine, OR identify if it's a direct action command.

Project Context:
{project_context}

File Structure:
```
{file_structure[:2000]}
```

User Request: "{user_question}"

TASK:
1. Identify the **Technical Intent** (e.g., "Persistence layer implementation", "Service initialization flow").
2. Formulate a **Refined Question** that is more descriptive and technical.
3. Suggest 5-10 **Technical Keywords** or likely symbol names (classes/functions) to search for.
4. Determine if the user is asking a direct action command (e.g., "Create a file", "Run a test", "Fix this bug") rather than a search query (e.g., "Where is the auth logic?"). Set `is_action_request` to true if it is an action.

Return ONLY a JSON object with this structure:
{{
  "intent": "string",
  "refined_question": "string",
  "keywords": ["list", "of", "strings"],
  "is_action_request": boolean
}}"""

def identify_relevant_files_prompt(user_question: str, file_structure: str, minimap_hint: str) -> str:
    return f"""You are an expert developer. Given this project file structure and symbol metadata, identify which files are MOST LIKELY to contain the answer to the user's question.

Project Structure:
```
{file_structure}
```
{minimap_hint}

Question: {user_question}

RULES:
- Return 3-8 file paths that are most relevant
- Prioritize source code files (.py, .js, .ts) over docs/tests
- Use the Symbol MiniMap to be surgical. If a function signature or docstring matches the question's intent, ALWAYS include that file.
- For questions about configuration, constants, or specific model names/ports, ALWAYS include 'config.py' or equivalent config files.
- For questions about data storage, caching, or history, ALWAYS include relevant service files (e.g., 'services/redis.py', 'services/database.py') even if the feature sounds missing.
- For questions about security/auth, include auth-related files  
- For questions about features, include the main app file AND relevant service files
- For questions about CORS, middleware, or server config, ALWAYS include main.py

Return ONLY JSON array of file paths, no explanation. Example:
["services/auth_service.py", "routes/auth_router.py", "config.py"]"""

def generate_search_queries_prompt(user_question: str, project_context: str, structure_hint: str, history_str: str) -> str:
    return f"""
You are an expert developer assistant. Your task is to generate 5-10 search queries to find code relevant to the user's question.
Target tool: ripgrep (regex supported).

Project Context (Summary):
{project_context}
{structure_hint}
Strategies:
1.  **Simple Keywords**: Start with broad, single-word terms (e.g., 'platform', 'linux', 'windows', 'support').
2.  **Code Patterns**: Search for class names, function definitions, variable assignments related to the question.
3.  **Exact Matches**: Look for exact string literals if the user asks for a specific message or error.
4.  **Synonyms**: Include synonyms and related terms (e.g., for 'brute force' also search 'login_attempt', 'lockout', 'blocked').
5.  **Configuration Patterns**: For config questions, search for middleware, env vars, constants (e.g., 'CORSMiddleware', 'allow_origins').
6.  **Avoid Complex Regex**: Do NOT use complex regex (like `.*`) unless searching for a strict code pattern. Prefer simple substrings.

Format: Return ONLY the search queries, one per line. No bullets, no numbering.

{history_str}
Question: {user_question}
"""

def github_search_query_prompt(user_question: str) -> str:
    return f"Search query for {user_question}"

def answer_question_prompt(user_question: str, context: str, history_str: str) -> str:
    return f"""
You are an expert developer assistant. Answer the user's question based strictly on the provided codebase context and the conversation history.
If the answer is not in the context, say so. Do not hallucinate.

{history_str}
Question: {user_question}

Context:
{context}
"""

def answer_code_question_prompt(user_question: str, context: str, history_str: str, structure_section: str, skeleton_section: str, graph_section: str) -> str:
    return f"""You are an expert code analyst. Answer the user's question using the provided code context, call graph, and project structure.

ANALYSIS STRATEGY:
1. **Targeted Files First**: Start by analyzing the code from the specifically targeted files — these were identified as most relevant.
2. **Context Awareness**: Understand where each code snippet fits in the project architecture.
3. **Function Relationships**: Use the call graph to trace which functions interact with each other.
4. **Call Chain Tracing**: When a function is relevant, identify what it calls (downstream) and what calls it (upstream).
5. **Root Cause Identification**: If the question is about a bug or issue, trace the dependency chain to find the actual source — not just the symptom.
6. **Look for Stubs/Mocks/TODOs/Dead Code**: 
   - Carefully check for stub implementations, TODO comments, hardcoded test values, or mock returns.
   - Even if a feature is marked as 'missing' in a README, check if service methods exist but are not 'wired up' (called) in the routes.
7. **Find Hardcoded Constants**: For questions about specific values (model names, ports, expirations), look for the actual string or integer definitions in config files, not just the setting variable name.

RULES:
- Always cite the file path and line numbers for your answer.
- If you find a function is causing an issue, trace its callers and callees to explain the full impact.
- If the answer is not in the context, say so clearly.
- Do NOT hallucinate code that doesn't exist in the context.
- When analyzing security or configuration, examine the ACTUAL values in code (not docs or comments about what they should be).

{history_str}
{structure_section}
{skeleton_section}
{graph_section}

Code Context:
{context}

Question: {user_question}"""

def analyze_project_context_prompt(readme_content: str) -> str:
    return f"""
You are an expert developer assistant. Analyze the following README content and provide a concise summary to help with code search.
Focus on:
1. Core Functionality & Purpose
2. Key Architectural Components (if mentioned)
3. Important Terminology or Jargon
4. Folder Structure hints (if mentioned)

Keep it under 200 words.

README Content:
{readme_content[:1000]} 
"""

def generate_questions_prompt(context: str, num: int) -> str:
    return f"""
You are an expert developer assistant. 
Based on the following project codebase dump, generate {num} insightful technical questions that a new developer might ask to understand the architecture, key features, or usage of this system.
For each question, provide a brief, accurate answer derived ONLY from the provided context.

Format:
1. **Question**: [Question text]
   - **Answer**: [Brief answer]

Context (Truncated):
{context}
"""

def decide_action_prompt(user_question: str, context: str, history_str: str, project_structure: str, available_tools: str) -> str:
    return f"""You are an autonomous AI software engineer. The user has asked a question or requested a task.
Based on the provided codebase context and the current state, you must decide what to do next.

AVAILABLE TOOLS:
{available_tools}

CURRENT CONTEXT:
{context[:20000]}

PROJECT STRUCTURE:
```
{project_structure[:5000]}
```

CONVERSATION HISTORY:
{history_str}

USER REQUEST: {user_question}

TASK:
You must choose exactly one of two options:
1. **Take an Action**: If you need to edit a file, read a file to get more context, etc. You must use one of the available tools.
2. **Final Answer**: If you have completed the user's request, or have gathered enough information to answer the question perfectly without further actions.

You must respond with ONLY a valid JSON object. 

Format for taking an action:
{{
  "action": "tool_call",
  "tool": "<tool_name_from_available_tools>",
  "method": "<method_name>",
  "args": {{
     "param1": "value1"
  }},
  "thought": "I need to read this file because..."
}}

Format for final answer:
{{
  "action": "final_answer",
  "content": "Here is the final answer to your question...",
  "thought": "I have all the context needed, I will now answer."
}}
"""

def verify_answer_prompt(question: str, answer: str, context: str) -> str:
    return f"""
You are an expert technical auditor. Your task is to verify an answer provided by an AI assistant based strictly on the provided codebase context.

User Question: {question}
AI Answer: {answer}

Codebase Context:
{context[:10000]}

Evaluate the answer based on the following criteria:
1. **Factual Accuracy**: Is the answer correct according to the context?
2. **Hallucination**: Does the answer mention things not found in the context?
3. **Completeness**: Does it fully address the user's question?

Format your response as a JSON object with the following keys:
- "verdict": "PASS", "FAIL", or "PARTIAL"
- "confidence_score": (float between 0 and 1)
- "reasoning": (concise explanation of your evaluation)
- "suggested_correction": (optional, if FAIL or PARTIAL)

JSON Response:
"""
