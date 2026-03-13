"""
main.py — Thin CLI entrypoint for GitSurf.

All heavy pipeline logic lives in src/orchestrator.py.
"""

import argparse
import sys
import os
import shutil

from dotenv import load_dotenv
load_dotenv()

from src.llm_client import LLMClient
from src.history_manager import HistoryManager
from src.verifier import AnswerVerifier
from src.tools.file_editor_tool import FileEditorTool
from src.tools.search_tool import SearchTool
from src.tools.web_tool import WebSearchTool
from src.tools.repo_manager import RepoManager
from src.tools.markdown_repo_manager import MarkdownRepoManager
from src.orchestrator import run_code_aware_pipeline, run_local_pipeline, PipelineContext

# Force UTF-8 for stdout/stderr on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


AVAILABLE_TOOLS = """
Tool: FileEditorTool
Description: Read, write, modify, or delete files inside the project directory.
Methods:
  - read_file(rel_path, start_line=None, end_line=None)
  - write_file(rel_path, content)
  - replace_in_file(rel_path, target, replacement)
  - delete_file(rel_path)

Tool: SearchTool
Description: Search for text patterns in the codebase using ripgrep.
Methods:
  - search(query, search_path=".")
  - search_and_chunk(query, search_path=".", context_lines=10)

Tool: WebSearchTool
Description: Search the web or fetch URL content for documentation/errors.
Methods:
  - search(query, num_results=5)
  - fetch_url(url)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitSurf — Agentic Code Search & Edit Tool")
    parser.add_argument("question", nargs="?", help="Question or command for the agent.")
    parser.add_argument("--path", default=".", help="Root path for local search.")
    parser.add_argument("--github-repo", help="GitHub repo (e.g. 'owner/repo').")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai).")
    parser.add_argument("--reset", action="store_true", help="Reset conversation history.")
    parser.add_argument("--clone", action="store_true", help="Use full git clone (legacy).")
    parser.add_argument("--suggest", action="store_true", help="Suggest questions for the repo.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force-rebuild FAISS index.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip answer verification.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete all cached data.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive CLI REPL.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Cache clearing
    if args.clear_cache:
        cache_dir = os.path.abspath(".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"[Cache] Cleared: {cache_dir}")
        else:
            print("[Cache] No cache directory found.")
        if not args.question and not args.interactive:
            return

    # Core components
    llm = LLMClient(provider=args.provider)
    history_mgr = HistoryManager()

    if args.reset:
        history_mgr.clear_history()
        print("Conversation history reset.")
        if not args.question and not args.interactive:
            return

    is_interactive = args.interactive or (not args.question and not args.suggest)

    if not is_interactive and not args.question and not args.suggest:
        parser.print_help()
        return

    search_path = args.path
    project_context = ""

    # GitHub Repo Setup
    if args.github_repo:
        if args.clone:
            print(f"Mode: GitHub Search ({args.github_repo}) - Git Clone")
            repo_mgr = RepoManager()
            try:
                search_path = repo_mgr.sync_repo(args.github_repo)
            except Exception as e:
                print(f"Error syncing repo: {e}")
                sys.exit(1)
        else:
            print(f"Mode: GitHub Search ({args.github_repo}) - Markdown Cache")
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                print("Error: GITHUB_TOKEN required.")
                sys.exit(1)

            repo_mgr = MarkdownRepoManager(token=token, cache_dir=".cache")

            if args.suggest:
                cached_path = repo_mgr.get_cache_path(args.github_repo)
                context = ""
                if cached_path:
                    codebase_md = os.path.join(cached_path, "full_codebase.md")
                    with open(codebase_md, "r", encoding="utf-8") as f:
                        context = f.read()
                else:
                    context = repo_mgr.fetch_readme(args.github_repo)
                if context:
                    print(llm.generate_questions(context))
                return

            cached_path = repo_mgr.get_cache_path(args.github_repo)
            if cached_path:
                readme_part = repo_mgr.get_local_context(args.github_repo)
                if readme_part:
                    project_context = llm.analyze_project_context(readme_part)
                search_path = cached_path
            else:
                readme_content = repo_mgr.fetch_readme(args.github_repo)
                if readme_content:
                    project_context = llm.analyze_project_context(readme_content)
                search_path = repo_mgr.sync_repo(args.github_repo)

    # Agent Tools
    file_editor = FileEditorTool(root_path=search_path)
    searcher = SearchTool()
    web_tool = WebSearchTool()
    is_code_search = args.github_repo is not None
    pipeline_ctx = PipelineContext(search_path=search_path, rebuild_index=args.rebuild_index)

    # Tool registry: maps tool names to instances for the PRAR action loop
    agent_tools = {
        "FileEditorTool": file_editor,
        "SearchTool": searcher,
        "WebSearchTool": web_tool,
    }

    def process_query(current_question: str):
        print(f"\nAnalyzing question: '{current_question}'...")
        history_context = history_mgr.get_recent_context()

        if is_code_search:
            answer, full_context = run_code_aware_pipeline(
                question=current_question,
                search_path=search_path,
                llm=llm,
                project_context=project_context,
                available_tools=AVAILABLE_TOOLS,
                tools=agent_tools,
                history=history_context,
                rebuild_index=args.rebuild_index,
                ctx=pipeline_ctx,
            )
        else:
            answer, full_context = run_local_pipeline(
                question=current_question,
                search_path=search_path,
                llm=llm,
                project_context=project_context,
                available_tools=AVAILABLE_TOOLS,
                tools=agent_tools,
                history=history_context,
                rebuild_index=args.rebuild_index,
                ctx=pipeline_ctx,
            )

        # Verification (shared)
        verification_summary = ""
        if not args.skip_verify and llm.client:
            step_label = "[Step 8/8]" if is_code_search else "[Step 6/6]"
            print(f"{step_label} Verifying Answer...")
            verifier = AnswerVerifier(client=llm.client)
            v_result = verifier.verify(current_question, answer, full_context)
            verdict = v_result.get("verdict", "UNKNOWN")
            reasoning = v_result.get("reasoning", "")
            verification_summary = f"\n[Verification Verdict: {verdict}]\nReasoning: {reasoning}"
            if v_result.get("suggested_correction"):
                verification_summary += f"\nNote: {v_result['suggested_correction']}"
        elif not args.skip_verify and not llm.client:
            verification_summary = "\n[Verification skipped: No API key configured]"

        print("\n=== FINAL ANSWER ===\n")
        print(answer)
        if verification_summary:
            print(verification_summary)

        history_mgr.add_interaction(current_question, answer)

    # Run Pipeline
    if is_interactive:
        print("\n=== GitSurf Interactive Mode ===")
        print("Type 'exit', 'quit', or press Ctrl+C to quit.\n")
        if args.question:
            process_query(args.question)
        while True:
            try:
                user_input = input("gitSurf> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting interactive mode.")
                    break
                process_query(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
    else:
        if args.question:
            process_query(args.question)


if __name__ == "__main__":
    main()