"""
Orchestrator: Coordinates pipelines and the shared ReAct action loop.

  - run_code_aware_pipeline: GitHub repo search (8-step skeleton + dual graph pipeline)
  - run_local_pipeline: Local file system search (4-step pipeline)
  - execute_action_loop: Shared ReAct (Reason + Act) loop used by both pipelines
"""

import os
import json
from typing import List, Dict, Optional

from src.tools.search_tool import SearchTool
from src.tools.vector_search_tool import VectorSearchTool
from src.tools.bm25_search_tool import BM25SearchTool
from src.tools.targeted_retriever import TargetedRetriever
from src.tools.symbol_extractor import SymbolExtractor
from src.tools.call_graph import CallGraph
from src.tools.file_editor_tool import FileEditorTool
from src.embeddings import EmbeddingClient
from src.reranker import CrossEncoderReranker


# ─────────────────────────────────────────────
#  Helper: Reciprocal Rank Fusion
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Standard RRF algorithm to merge multiple ranked result lists."""
    scores: Dict = {}
    doc_map: Dict = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = (doc["file"], doc["start_line"], doc["end_line"])
            scores.setdefault(key, 0.0)
            doc_map[key] = doc
            scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    merged = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc["rrf_score"] = scores[key]
        merged.append(doc)
    return merged


# ─────────────────────────────────────────────
#  Shared: ReAct Action Loop
# ─────────────────────────────────────────────
def execute_action_loop(
    question: str,
    initial_context: str,
    llm,
    file_editor: FileEditorTool,
    available_tools: str,
    project_structure: str = "",
    extra_context_prefix: str = "",
    history=None,
    max_iterations: int = 5,
) -> str:
    """
    Shared ReAct loop. Lets the LLM choose between tool calls and a final answer.
    Returns the final answer string.
    """
    full_context = initial_context
    answer = None

    for iteration in range(1, max_iterations + 1):
        print(f"   [Iteration {iteration}/{max_iterations}] Thinking...")

        context_for_llm = (f"{extra_context_prefix}\n\n{full_context}".strip()
                           if extra_context_prefix else full_context)

        decision = llm.decide_action(
            question,
            context_for_llm,
            project_structure=project_structure,
            history=history,
            available_tools=available_tools,
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

            if tool == "FileEditorTool":
                fn = getattr(file_editor, method, None)
                if fn:
                    observation = fn(**kwargs)
                else:
                    observation = f"[Error] Unknown method: {method}"
            else:
                observation = f"[Error] Unknown tool: {tool}"

            print(f"   Observation: {observation[:120]}...\n")
            action_str = f"Action taken: {tool}.{method}({kwargs})\nObservation: {observation}"
            full_context += f"\n\n--- ACTION LOG ---\n{action_str}"

        else:
            print(f"   [Warning] Unknown action type: {action_type}")
            answer = "Error: Invalid agent action."
            break

    if answer is None:
        answer = "Error: Agent reached maximum iterations without giving a final answer."
    return answer


# ─────────────────────────────────────────────
#  Pipeline: Code-Aware (GitHub repos)
# ─────────────────────────────────────────────
def run_code_aware_pipeline(
    question: str,
    search_path: str,
    llm,
    project_context: str,
    available_tools: str,
    file_editor: FileEditorTool,
    history=None,
    rebuild_index: bool = False,
) -> tuple:
    """
    8-step code-aware pipeline for GitHub repos.
    Returns (answer: str, full_context: str).
    """
    print("\n[Code-Aware Pipeline]")

    # Step 1: Load Project Skeleton + Symbol MiniMap
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

    symbol_minimap: Dict = {}
    minimap_path = os.path.join(search_path, "symbol_minimap.json")
    if os.path.exists(minimap_path):
        try:
            with open(minimap_path, "r", encoding="utf-8") as f:
                symbol_minimap = json.load(f)
            print(f"   Loaded Symbol MiniMap for {len(symbol_minimap)} files")
        except Exception:
            pass

    # Step 2: Query Refinement
    print("[Step 2/8] Refining Query (Technical Intent)...")
    refined_data = llm.refine_user_query(
        question, project_context=project_context, file_structure=project_structure
    )
    query_to_use = refined_data.get("refined_question", question)
    technical_intent = refined_data.get("intent", "General search")
    expansion_keywords = refined_data.get("keywords", [])
    is_action_request = refined_data.get("is_action_request", False)

    print(f"   Intent: {technical_intent}")
    if query_to_use != question:
        print(f"   Refined Question: {query_to_use}")

    top_chunks: List[Dict] = []
    call_graph_context = ""
    skeleton_context = ""

    if is_action_request:
        print("   [Fast-Path] Action command detected. Skipping search pipeline (Steps 3-7).")
    else:
        # Step 3: Skeleton Analysis
        print("[Step 3/8] Skeleton Analysis (identifying relevant files)...")
        targeted_files: List[str] = []
        if project_structure:
            targeted_files = llm.identify_relevant_files(
                query_to_use, project_structure, symbol_minimap=symbol_minimap
            )
            if targeted_files:
                skeleton_context = "Targeted files:\n" + "\n".join(
                    f"  - {f}" for f in targeted_files
                )

        # Step 4: Targeted File Retrieval
        print("[Step 4/8] Targeted File Retrieval...")
        targeted_retriever = TargetedRetriever(cache_path=search_path)
        targeted_chunks: List[Dict] = []
        if targeted_files:
            targeted_chunks = targeted_retriever.retrieve_files(targeted_files)
            print(f"   Retrieved {len(targeted_chunks)} targeted file(s)")
        else:
            print("   No targeted files identified, relying on search only")

        # Step 5: Symbol Extraction + Call Graph
        print("[Step 5/8] Code Analysis (Symbols + Call Graph)...")
        sym_extractor = SymbolExtractor(cache_dir=os.path.join(".cache", "symbols"))
        symbol_index = sym_extractor.extract_from_directory(
            search_path, force_rebuild=rebuild_index
        )
        cg = CallGraph(cache_dir=os.path.join(".cache", "call_graph"))
        cg.build_from_symbols(symbol_index, force_rebuild=rebuild_index)

        # Step 6: Triple-Hybrid Search
        print("[Step 6/8] Triple-Hybrid Search (Skeleton-Guided)...")
        emb_client = EmbeddingClient()
        vector_tool = VectorSearchTool(
            embedding_client=emb_client, cache_dir=os.path.join(".cache", "vector_index")
        )
        vector_tool.build_index_with_symbols(
            search_path, symbol_index, force_rebuild=rebuild_index
        )
        vector_results = vector_tool.search(query_to_use, top_k=20)

        bm25_tool = BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
        bm25_tool.build_index(vector_tool.metadata, force_rebuild=rebuild_index)
        bm25_results = bm25_tool.search(query_to_use, top_k=20)

        searcher = SearchTool()
        queries = llm.generate_search_queries(
            query_to_use, tool="ripgrep",
            project_context=project_context,
            file_structure=project_structure,
        )
        if expansion_keywords:
            queries = expansion_keywords[:3] + queries
        keyword_chunks: List[Dict] = []
        for q in queries[:5]:
            keyword_chunks.extend(searcher.search_and_chunk(q, search_path))

        print("   Applying Reciprocal Rank Fusion (RRF)...")
        search_candidates = reciprocal_rank_fusion(
            [vector_results, bm25_results, keyword_chunks]
        )

        # Step 7: Merge + Rerank
        print("[Step 7/8] Merging + Reranking...")
        print(f"   [Orchestrator] Keeping {len(targeted_chunks)} targeted chunks.")
        reranker = CrossEncoderReranker()
        slots_remaining = max(10 - len(targeted_chunks), 3)
        reranked_search = reranker.rerank(query_to_use, search_candidates, top_k=slots_remaining)

        top_chunks = list(targeted_chunks)
        seen_paths = {c["file"] for c in targeted_chunks}
        for chunk in reranked_search:
            if chunk["file"] not in seen_paths:
                top_chunks.append(chunk)
                seen_paths.add(chunk["file"])

        print(
            f"   Selected top {len(top_chunks)} chunks "
            f"(Targeted: {len(targeted_chunks)}, Search: {len(top_chunks) - len(targeted_chunks)})"
        )

        # Build call graph context for top symbols
        graph_parts = []
        seen_symbols: set = set()
        for chunk in top_chunks:
            sym = chunk.get("symbol", "")
            if sym and sym not in seen_symbols:
                seen_symbols.add(sym)
                ctx = cg.get_context_for_function(sym, depth=2)
                if ctx and "No call graph data" not in ctx:
                    graph_parts.append(ctx)
        call_graph_context = "\n\n---\n\n".join(graph_parts)

    # Step 8: Agentic Action Loop
    print("[Step 8/8] Agentic Action Loop (Synthesizing/Editing)...")
    initial_context = "\n\n---\n\n".join(c["content"] for c in top_chunks)
    extra_prefix = f"{skeleton_context}\n\n{call_graph_context}".strip()

    answer = execute_action_loop(
        question=question,
        initial_context=initial_context,
        llm=llm,
        file_editor=file_editor,
        available_tools=available_tools,
        project_structure=project_structure,
        extra_context_prefix=extra_prefix,
        history=history,
    )
    return answer, initial_context


# ─────────────────────────────────────────────
#  Pipeline: General (Local File System)
# ─────────────────────────────────────────────
def run_local_pipeline(
    question: str,
    search_path: str,
    llm,
    project_context: str,
    available_tools: str,
    file_editor: FileEditorTool,
    history=None,
    rebuild_index: bool = False,
) -> tuple:
    """
    4-step general pipeline for local file searches.
    Returns (answer: str, full_context: str).
    """
    print("\n[General Search Pipeline]")
    project_structure = "[Local search tree not injected by default]"

    refined_data = llm.refine_user_query(
        question, project_context=project_context, file_structure=project_structure
    )
    is_action_request = refined_data.get("is_action_request", False)

    top_chunks: List[Dict] = []

    if is_action_request:
        print("   [Fast-Path] Action command detected. Skipping search pipeline (Steps 1-2).")
    else:
        print("[Step 1/4] Triple-Hybrid Search (Keyword + Semantic + Statistical)...")
        emb_client = EmbeddingClient()
        vector_tool = VectorSearchTool(
            embedding_client=emb_client, cache_dir=os.path.join(".cache", "vector_index")
        )
        vector_tool.build_index(search_path, force_rebuild=rebuild_index)
        vector_results = vector_tool.search(question, top_k=20)

        bm25_tool = BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
        bm25_tool.build_index(vector_tool.metadata, force_rebuild=rebuild_index)
        bm25_results = bm25_tool.search(question, top_k=20)

        searcher = SearchTool()
        queries = llm.generate_search_queries(
            question, tool="ripgrep", project_context=project_context
        )
        keyword_chunks: List[Dict] = []
        for q in queries[:3]:
            keyword_chunks.extend(searcher.search_and_chunk(q, search_path))

        print("   Applying Reciprocal Rank Fusion (RRF)...")
        candidates = reciprocal_rank_fusion([vector_results, bm25_results, keyword_chunks])
        print(f"   Collected {len(candidates)} unique candidate chunks.")

        print("[Step 2/4] Reranking Chunks (Local BERT Cross-Encoder)...")
        reranker = CrossEncoderReranker()
        top_chunks = reranker.rerank(question, candidates, top_k=5)
        print(f"   Selected top {len(top_chunks)} chunks.")

    print("[Step 3/4] Agentic Action Loop (Synthesizing/Editing)...")
    initial_context = "\n\n---\n\n".join(c["content"] for c in top_chunks)

    answer = execute_action_loop(
        question=question,
        initial_context=initial_context,
        llm=llm,
        file_editor=file_editor,
        available_tools=available_tools,
        project_structure=project_structure,
        history=history,
    )
    return answer, initial_context
