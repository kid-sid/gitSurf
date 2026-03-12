from typing import List, Dict, Optional

def refine_query_prompt(user_question: str, project_context: str, file_structure: str) -> str:
    return f"""Analyze the user's request about the codebase and return ONLY a JSON object.

<project_context>
{project_context}
</project_context>

<file_structure>
{file_structure[:2000]}
</file_structure>

<user_request>{user_question}</user_request>

TASK:
1. Extract the technical "intent".
2. Formulate a "refined_question".
3. Extract 5-10 technical "keywords" (e.g., class names, functions).
4. Set "is_action_request" to true ONLY if the user is asking to perform an action (edit, run, delete) rather than search.

EXPECTED JSON FORMAT:
{{
  "intent": "string",
  "refined_question": "string",
  "keywords": ["str1", "str2"],
  "is_action_request": boolean
}}"""

def identify_relevant_files_prompt(user_question: str, file_structure: str, minimap_hint: str) -> str:
    return f"""Identify 3-8 files most likely to answer the user's question. Return ONLY a JSON array of strings.

<file_structure>
{file_structure}
</file_structure>

{minimap_hint}

<question>{user_question}</question>

RULES:
- Return ONLY top 3-8 exact file paths.
- Prioritize source code (.py, .js, .ts).
- If config/constants requested, include config.py.
- If data/history requested, include relevant service files regardless of features.

EXPECTED JSON FORMAT:
["path1.py", "path2.js"]"""

def generate_search_queries_prompt(user_question: str, project_context: str, structure_hint: str, history_str: str) -> str:
    return f"""Generate 5-10 ripgrep search queries for the user's question. Return ONLY the queries, one per line.

<context>
{project_context}
{structure_hint}
</context>

{history_str}
<question>{user_question}</question>

RULES:
- Use simple keywords and obvious code patterns (e.g., class names).
- Avoid complex regex (`.*`).
- Include synonyms.
- One line per query. No numbering or markdown."""

def github_search_query_prompt(user_question: str) -> str:
    return f"Search query for {user_question}"

def answer_question_prompt(user_question: str, context: str, history_str: str) -> str:
    return f"""Answer the user's question strictly using the provided codebase context. Do not hallucinate.

{history_str}
<question>{user_question}</question>

<context>
{context}
</context>"""

def answer_code_question_prompt(user_question: str, context: str, history_str: str, structure_section: str, skeleton_section: str, graph_section: str) -> str:
    return f"""Answer the user's question strictly using the provided codebase context.

RULES:
- Cite file paths and line numbers.
- Trace call graphs to identify root causes.
- Check actual values in code, not just comments.
- Do not hallucinate code.

{history_str}
{structure_section}
{skeleton_section}
{graph_section}

<context>
{context}
</context>

<question>{user_question}</question>"""

def analyze_project_context_prompt(readme_content: str) -> str:
    return f"""Summarize this README into < 200 words focusing on Purpose, Key Components, and Jargon.

<readme>
{readme_content[:1500]}
</readme>"""

def generate_questions_prompt(context: str, num: int) -> str:
    return f"""Generate {num} insightful technical questions for a new developer based on the context.

FORMAT:
1. **Question**: [text]
   - **Answer**: [brief answer from context ONLY]

<context>
{context}
</context>"""

def decide_action_prompt(user_question: str, context: str, history_str: str, project_structure: str, available_tools: str) -> str:
    return f"""You are an autonomous AI software engineer. Decide the next action. Return ONLY valid JSON.

<available_tools>
{available_tools}
</available_tools>

<project_structure>
{project_structure[:5000]}
</project_structure>

<current_context>
{context[:20000]}
</current_context>

{history_str}
<user_request>{user_question}</user_request>

TASK:
Choose exactly one option:
1. action: "tool_call" (if you need to edit, read, or search)
2. action: "final_answer" (if you have gathered enough info to fulfill the request)

TOOL CALL JSON FORMAT:
{{
  "action": "tool_call",
  "tool": "ToolName",
  "method": "method_name",
  "args": {{"param": "value"}},
  "thought": "Reasoning here."
}}

FINAL ANSWER JSON FORMAT:
{{
  "action": "final_answer",
  "content": "Final response text here.",
  "thought": "Reasoning here."
}}"""

def verify_answer_prompt(question: str, answer: str, context: str) -> str:
    return f"""Verify the AI answer strictly against the codebase context. Return ONLY valid JSON.

<question>{question}</question>
<ai_answer>{answer}</ai_answer>

<context>
{context[:10000]}
</context>

EXPECTED JSON FORMAT:
{{
  "verdict": "PASS" | "FAIL" | "PARTIAL",
  "confidence_score": 0.9,
  "reasoning": "Explanation here",
  "suggested_correction": "Required if FAIL/PARTIAL"
}}"""
