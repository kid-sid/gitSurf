import argparse
import sys
import os
import shutil
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from src.tools.search_tool import SearchTool
from src.tools.repo_manager import RepoManager
from src.tools.markdown_repo_manager import MarkdownRepoManager
from src.tools.vector_search_tool import VectorSearchTool
from src.tools.bm25_search_tool import BM25SearchTool
from src.embeddings import EmbeddingClient
from src.llm_client import LLMClient
from src.history_manager import HistoryManager
from src.reranker import CrossEncoderReranker
from src.verifier import AnswerVerifier

# Force UTF-8 for stdout/stderr (fixes Windows console encoding issues)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Standard RRF algorithm to merge multiple ranked lists.
    Each list is expected to be sorted by relevance.
    """
    scores = {} # (file, start, end) -> score
    doc_map = {} # (file, start, end) -> chunk_data
    
    for results in results_lists:
        for rank, doc in enumerate(results):
            # Use 1-based ranking for RRF
            key = (doc["file"], doc["start_line"], doc["end_line"])
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + (rank + 1))
            
    # Sort by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    merged_results = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc["rrf_score"] = scores[key]
        merged_results.append(doc)
    
    return merged_results

def main():
    parser = argparse.ArgumentParser(description="Agentic Search Tool")
    parser.add_argument("question", nargs="?", help="The question you want to ask about the codebase.")
    parser.add_argument("--path", default=".", help="Root path to search in (for local search).")
    parser.add_argument("--github-repo", help="GitHub repo to search (e.g., 'owner/repo'). Overrides local path.")
    parser.add_argument("--provider", default="openai", help="LLM Provider to use (default: openai).")
    parser.add_argument("--reset", action="store_true", help="Reset conversation history.")
    parser.add_argument("--clone", action="store_true", help="Use full git clone (legacy behavior). Default is markdown cache.")
    parser.add_argument("--suggest", action="store_true", help="Generate sample questions based on the codebase context.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild the FAISS vector index.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip answer verification.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete all cached indexes and cloned repos.")
    
    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        cache_dir = os.path.abspath(".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"[Cache] Cleared all cached data from: {cache_dir}")
        else:
            print("[Cache] No cache directory found.")
        if not args.question:
            return

    # 1. Initialize Components
    llm = LLMClient(provider=args.provider)
    history_mgr = HistoryManager()
    
    if args.reset:
        history_mgr.clear_history()
        print("Conversation history reset.")
        if not args.question:
            return

    if not args.question:
        parser.print_help()
        return

    print(f"Analyzing question: '{args.question}'...")
    
    history_context = history_mgr.get_recent_context()
    search_path = args.path
    project_context = ""
    
    # --- GitHub Repo Management ---
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
                    with open(os.path.join(cached_path, "full_codebase.md"), "r", encoding="utf-8") as f:
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
    
    # Initialize Agent Tools
    from src.tools.file_editor_tool import FileEditorTool
    file_editor = FileEditorTool(root_path=search_path)
    
    available_tools = """
    Tool: FileEditorTool
    Description: Allows you to read, write, and modify files within the project. Only use files within the project.
    Methods:
      - read_file(rel_path, start_line=None, end_line=None): Reads a file or specific lines (1-indexed). Returns string.
      - write_file(rel_path, content): Creates or overwrites a file. Returns string.
      - replace_in_file(rel_path, target, replacement): Replaces an exact string block with another. Returns string.
    """

    # --- Determine search mode ---
    is_code_search = args.github_repo is not None

    if is_code_search:
        # ============================================================
        #  CODE-AWARE PIPELINE (GitHub Repos) — 7 Steps
        # ============================================================
        print("\n[Code-Aware Pipeline]")

        # Step 1: Load Project Skeleton
        print("[Step 1/8] Loading Project Skeleton...")
        project_structure = ""
        structure_path = os.path.join(search_path, "project_structure.txt")
        if os.path.exists(structure_path):
            try:
                with open(structure_path, "r", encoding="utf-8") as f:
                    project_structure = f.read()
                print(f"   Loaded file tree ({len(project_structure.splitlines())} entries)")
            except Exception:
                pass
            
        # Step 1b: Load Symbol MiniMap
        symbol_minimap = {}
        minimap_path = os.path.join(search_path, "symbol_minimap.json")
        if os.path.exists(minimap_path):
            try:
                with open(minimap_path, "r", encoding="utf-8") as f:
                    symbol_minimap = json.load(f)
                print(f"   Loaded Symbol MiniMap for {len(symbol_minimap)} files")
            except Exception:
                pass

        # Step 2: Query Refinement — Bridge the gap between vague questions and code
        print("[Step 2/8] Refining Query (Technical Intent)...")
        refined_data = llm.refine_user_query(args.question, project_context=project_context, file_structure=project_structure)
        query_to_use = refined_data.get("refined_question", args.question)
        technical_intent = refined_data.get("intent", "General search")
        expansion_keywords = refined_data.get("keywords", [])
        is_action_request = refined_data.get("is_action_request", False)
        
        print(f"   Intent: {technical_intent}")
        if query_to_use != args.question:
            print(f"   Refined Question: {query_to_use}")

        if is_action_request:
            print(f"   [Fast-Path] Action command detected. Skipping search pipeline (Steps 3-7).")
            top_chunks = []
            targeted_chunks = []
            call_graph_context = ""
            skeleton_context = ""
        else:
            # Step 3: Skeleton Analysis — LLM identifies relevant files
            print("[Step 3/8] Skeleton Analysis (identifying relevant files)...")
            targeted_files = []
            skeleton_context = ""
            if project_structure:
                targeted_files = llm.identify_relevant_files(query_to_use, project_structure, symbol_minimap=symbol_minimap)
                if targeted_files:
                    skeleton_context = "Targeted files:\n" + "\n".join(f"  - {f}" for f in targeted_files)

            # Step 4: Targeted File Retrieval — read full content of identified files
            print("[Step 4/8] Targeted File Retrieval...")
            from src.tools.targeted_retriever import TargetedRetriever
            targeted_retriever = TargetedRetriever(cache_path=search_path)
            targeted_chunks = []
            if targeted_files:
                targeted_chunks = targeted_retriever.retrieve_files(targeted_files)
                print(f"   Retrieved {len(targeted_chunks)} targeted file(s)")
            else:
                print("   No targeted files identified, relying on search only")

            # Step 5: Symbol Extraction + Call Graph
            print("[Step 5/8] Code Analysis (Symbols + Call Graph)...")
            from src.tools.symbol_extractor import SymbolExtractor
            from src.tools.call_graph import CallGraph

            sym_extractor = SymbolExtractor(cache_dir=os.path.join(".cache", "symbols"))
            symbol_index = sym_extractor.extract_from_directory(search_path, force_rebuild=args.rebuild_index)

            call_graph = CallGraph(cache_dir=os.path.join(".cache", "call_graph"))
            call_graph.build_from_symbols(symbol_index, force_rebuild=args.rebuild_index)

            # Step 6: Triple-Hybrid Search (guided by skeleton)
            print("[Step 6/8] Triple-Hybrid Search (Skeleton-Guided)...")

            emb_client = EmbeddingClient()
            vector_tool = VectorSearchTool(
                embedding_client=emb_client,
                cache_dir=os.path.join(".cache", "vector_index")
            )
            vector_tool.build_index_with_symbols(search_path, symbol_index, force_rebuild=args.rebuild_index)
            vector_results = vector_tool.search(query_to_use, top_k=20)

            bm25_tool = BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
            bm25_tool.build_index(vector_tool.metadata, force_rebuild=args.rebuild_index)
            bm25_results = bm25_tool.search(query_to_use, top_k=20)

            searcher = SearchTool()
            queries = llm.generate_search_queries(
                query_to_use, tool="ripgrep",
                project_context=project_context,
                file_structure=project_structure
            )
            # Add technical keywords to ripgrep search
            if expansion_keywords:
                queries = expansion_keywords[:3] + queries
                
            keyword_chunks = []
            for q in queries[:5]:
                keyword_chunks.extend(searcher.search_and_chunk(q, search_path))

            print("   Applying Reciprocal Rank Fusion (RRF)...")
            search_candidates = reciprocal_rank_fusion([vector_results, bm25_results, keyword_chunks])

            # Step 7: Merge Targeted + Search, Deduplicate, Rerank
            print("[Step 7/8] Merging + Reranking...")

            # Targeted chunks - these are GOLD. Keep them all.
            print(f"   [Orchestrator] Keeping {len(targeted_chunks)} targeted chunks.")
            
            # Rerank search candidates only
            reranker = CrossEncoderReranker()
            # Calculate how many search results we can fit
            slots_remaining = 10 - len(targeted_chunks)
            if slots_remaining < 3: slots_remaining = 3 
            
            reranked_search = reranker.rerank(query_to_use, search_candidates, top_k=slots_remaining)
            
            # Combine
            top_chunks = list(targeted_chunks)
            seen_paths = {c['file'] for c in targeted_chunks}
            
            for chunk in reranked_search:
                if chunk['file'] not in seen_paths:
                    top_chunks.append(chunk)
                    seen_paths.add(chunk['file'])
                
            print(f"   Selected top {len(top_chunks)} chunks (Targeted: {len(targeted_chunks)}, Search: {len(top_chunks)-len(targeted_chunks)})")

            # Expand context: for matched symbols, get their call graph info
            graph_context_parts = []
            seen_symbols = set()
            for chunk in top_chunks:
                symbol_name = chunk.get("symbol", "")
                if symbol_name and symbol_name not in seen_symbols:
                    seen_symbols.add(symbol_name)
                    ctx = call_graph.get_context_for_function(symbol_name, depth=2)
                    if ctx and "No call graph data" not in ctx:
                        graph_context_parts.append(ctx)

            call_graph_context = "\n\n---\n\n".join(graph_context_parts) if graph_context_parts else ""

        # Step 8: Agentic Action Loop
        print("[Step 8/8] Agentic Action Loop (Synthesizing/Editing)...")
        full_context = "\n\n---\n\n".join([c["content"] for c in top_chunks])
        
        iteration = 0
        max_iterations = 5
        answer = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"   [Iteration {iteration}/{max_iterations}] Thinking...")
            
            # Use call_graph and skeleton as part of context for decide_action in github mode
            combined_context = f"{skeleton_context}\n\n{call_graph_context}\n\n{full_context}"
            
            decision = llm.decide_action(
                args.question, 
                combined_context, 
                project_structure=project_structure, 
                history=history_context, 
                available_tools=available_tools
            )
            
            action_type = decision.get("action")
            thought = decision.get("thought", "No thought provided.")
            print(f"   Thought: {thought}")
            
            if action_type == "final_answer":
                answer = decision.get("content", "No content provided.")
                break
            elif action_type == "tool_call":
                tool = decision.get("tool")
                method = decision.get("method")
                kwargs = decision.get("args", {})
                
                print(f"   Action -> {tool}.{method}({kwargs})")
                
                observation = ""
                if tool == "FileEditorTool":
                    if method == "read_file":
                        observation = file_editor.read_file(**kwargs)
                    elif method == "write_file":
                        observation = file_editor.write_file(**kwargs)
                    elif method == "replace_in_file":
                        observation = file_editor.replace_in_file(**kwargs)
                    else:
                        observation = f"[Error] Unknown method: {method}"
                else:
                    observation = f"[Error] Unknown tool: {tool}"
                    
                print(f"   Observation: {observation[:100]}...\n")
                
                # Append observation to context so the next iteration sees it
                action_str = f"Action taken: {tool}.{method}({kwargs})\nObservation: {observation}"
                full_context += f"\n\n--- ACTION LOG ---\n{action_str}"
            else:
                print(f"   [Warning] Unknown action type: {action_type}")
                answer = "Error: Invalid agent action."
                break
                
        if answer is None:
            answer = "Error: Agent reached maximum iterations without giving a final answer."

    else:
        # ============================================================
        #  GENERAL PIPELINE (Local File System)
        # ============================================================
        print("\n[General Search Pipeline]")
        refined_data = llm.refine_user_query(args.question, project_context=project_context, file_structure="[Local Search Tree Not Injected]")
        is_action_request = refined_data.get("is_action_request", False)
        
        if is_action_request:
            print(f"   [Fast-Path] Action command detected. Skipping search pipeline (Steps 1-2).")
            top_chunks = []
            project_structure = "[Local search tree not injected by default]"
        else:
            print("[Step 1/4] Triple-Hybrid Search (Keyword + Semantic + Statistical)...")

            # (Local File system skips building the file structure string right now, but we can provide a small one)
            project_structure = "[Local search tree not injected by default]"
            
            emb_client = EmbeddingClient()
            vector_tool = VectorSearchTool(
                embedding_client=emb_client,
                cache_dir=os.path.join(".cache", "vector_index")
            )
            vector_tool.build_index(search_path, force_rebuild=args.rebuild_index)
            vector_results = vector_tool.search(args.question, top_k=20)

            bm25_tool = BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
            bm25_tool.build_index(vector_tool.metadata, force_rebuild=args.rebuild_index)
            bm25_results = bm25_tool.search(args.question, top_k=20)

            searcher = SearchTool()
            queries = llm.generate_search_queries(args.question, tool="ripgrep", project_context=project_context)
            keyword_chunks = []
            for q in queries[:3]:
                keyword_chunks.extend(searcher.search_and_chunk(q, search_path))

            print("   Applying Reciprocal Rank Fusion (RRF)...")
            deduped_candidates = reciprocal_rank_fusion([vector_results, bm25_results, keyword_chunks])
            print(f"   Collected {len(deduped_candidates)} unique candidate chunks.")

            print("[Step 2/4] Reranking Chunks (Local BERT Cross-Encoder)...")
            reranker = CrossEncoderReranker()
            top_chunks = reranker.rerank(args.question, deduped_candidates, top_k=5)
            print(f"   Selected top {len(top_chunks)} chunks.")

        print("[Step 3/4] Agentic Action Loop (Synthesizing/Editing)...")
        full_context = "\n\n---\n\n".join([c["content"] for c in top_chunks])
        
        iteration = 0
        max_iterations = 5
        answer = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"   [Iteration {iteration}/{max_iterations}] Thinking...")
            
            decision = llm.decide_action(
                args.question, 
                full_context, 
                project_structure=project_structure, 
                history=history_context, 
                available_tools=available_tools
            )
            
            action_type = decision.get("action")
            thought = decision.get("thought", "No thought provided.")
            print(f"   Thought: {thought}")
            
            if action_type == "final_answer":
                answer = decision.get("content", "No content provided.")
                break
            elif action_type == "tool_call":
                tool = decision.get("tool")
                method = decision.get("method")
                kwargs = decision.get("args", {})
                
                print(f"   Action -> {tool}.{method}({kwargs})")
                
                observation = ""
                if tool == "FileEditorTool":
                    if method == "read_file":
                        observation = file_editor.read_file(**kwargs)
                    elif method == "write_file":
                        observation = file_editor.write_file(**kwargs)
                    elif method == "replace_in_file":
                        observation = file_editor.replace_in_file(**kwargs)
                    else:
                        observation = f"[Error] Unknown method: {method}"
                else:
                    observation = f"[Error] Unknown tool: {tool}"
                    
                print(f"   Observation: {observation[:100]}...\n")
                
                action_str = f"Action taken: {tool}.{method}({kwargs})\nObservation: {observation}"
                full_context += f"\n\n--- ACTION LOG ---\n{action_str}"
            else:
                print(f"   [Warning] Unknown action type: {action_type}")
                answer = "Error: Invalid agent action."
                break
                
        if answer is None:
            answer = "Error: Agent reached maximum iterations without giving a final answer."

    # --- Verification (shared by both pipelines) ---
    verification_summary = ""
    if not args.skip_verify:
        step_label = "[Step 8/8]" if is_code_search else "[Step 4/4]"
        print(f"{step_label} Verifying Answer...")
        verifier = AnswerVerifier(client=llm.client)
        v_result = verifier.verify(args.question, answer, full_context)

        verdict = v_result.get("verdict", "UNKNOWN")
        reasoning = v_result.get("reasoning", "")

        verification_summary = f"\n[Verification Verdict: {verdict}]\nReasoning: {reasoning}"
        if v_result.get("suggested_correction"):
            verification_summary += f"\nNote: {v_result['suggested_correction']}"

    print("\n=== FINAL ANSWER ===\n")
    print(answer)
    if verification_summary:
        print(verification_summary)

    history_mgr.add_interaction(args.question, answer)

if __name__ == "__main__":
    main()