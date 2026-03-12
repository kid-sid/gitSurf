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
#  PipelineContext: Caches tool instances
# ─────────────────────────────────────────────
class PipelineContext:
    """Holds initialized tool instances to avoid re-instantiation across queries.
    Create once in main.py and pass into pipeline calls."""

    def __init__(self, search_path: str, rebuild_index: bool = False):
        self.search_path = os.path.abspath(search_path)
        self.rebuild_index = rebuild_index

        # Lazily initialized tools (populated on first search query)
        self._emb_client = None
        self._vector_tool = None
        self._bm25_tool = None
        self._sym_extractor = None
        self._call_graph = None
        self._reranker = None
        self._searcher = None

        # Track if indexes have been built at least once
        self._indexes_built = False

    @property
    def emb_client(self):
        if self._emb_client is None:
            self._emb_client = EmbeddingClient()
        return self._emb_client

    @property
    def vector_tool(self):
        if self._vector_tool is None:
            self._vector_tool = VectorSearchTool(
                embedding_client=self.emb_client,
                cache_dir=os.path.join(".cache", "vector_index"),
            )
        return self._vector_tool

    @property
    def bm25_tool(self):
        if self._bm25_tool is None:
            self._bm25_tool = BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
        return self._bm25_tool

    @property
    def sym_extractor(self):
        if self._sym_extractor is None:
            self._sym_extractor = SymbolExtractor(cache_dir=os.path.join(".cache", "symbols"))
        return self._sym_extractor

    @property
    def call_graph(self):
        if self._call_graph is None:
            self._call_graph = CallGraph(cache_dir=os.path.join(".cache", "call_graph"))
        return self._call_graph

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def searcher(self):
        if self._searcher is None:
            self._searcher = SearchTool()
        return self._searcher


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
    ctx: Optional[PipelineContext] = None,
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
        sym_extractor = ctx.sym_extractor if ctx else SymbolExtractor(cache_dir=os.path.join(".cache", "symbols"))
        symbol_index = sym_extractor.extract_from_directory(
            search_path, force_rebuild=rebuild_index
        )
        cg = ctx.call_graph if ctx else CallGraph(cache_dir=os.path.join(".cache", "call_graph"))
        cg.build_from_symbols(symbol_index, force_rebuild=rebuild_index)

        # Step 6: Triple-Hybrid Search
        print("[Step 6/8] Triple-Hybrid Search (Skeleton-Guided)...")
        vector_tool = ctx.vector_tool if ctx else VectorSearchTool(
            embedding_client=EmbeddingClient(), cache_dir=os.path.join(".cache", "vector_index")
        )
        vector_tool.build_index_with_symbols(
            search_path, symbol_index, force_rebuild=rebuild_index
        )
        vector_results = vector_tool.search(query_to_use, top_k=20)

        bm25_tool = ctx.bm25_tool if ctx else BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
        bm25_tool.build_index(vector_tool.metadata, force_rebuild=rebuild_index)
        bm25_results = bm25_tool.search(query_to_use, top_k=20)

        searcher = ctx.searcher if ctx else SearchTool()
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
        reranker = ctx.reranker if ctx else CrossEncoderReranker()
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
                graph_ctx = cg.get_context_for_function(sym, depth=2)
                if graph_ctx and "No call graph data" not in graph_ctx:
                    graph_parts.append(graph_ctx)
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
#  Helpers: Local File Tree & Direct Retrieval
# ─────────────────────────────────────────────
SKIP_DIRS_LOCAL = {
    'node_modules', '.git', '.cache', '__pycache__', 'venv', '.venv',
    'dist', 'build', 'target', 'bin', 'obj', 'vendor', '.idea', '.vscode',
    '.next', '.nuxt', 'coverage', '.tox', 'eggs', '.eggs',
}

INDEXABLE_EXTENSIONS_LOCAL = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.c', '.cpp', '.h',
    '.cs', '.rb', '.php', '.rs', '.swift', '.kt', '.m', '.mm', '.sh', '.bat',
    '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.env', '.ini',
    '.cfg', '.conf', '.xml', '.rst', '.sql', '.html', '.css',
}


def build_local_file_tree(root_path: str, max_files: int = 500) -> str:
    """Build a lightweight file tree string for skeleton analysis."""
    tree_lines = []
    count = 0
    root_path = os.path.abspath(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS_LOCAL]

        rel_dir = os.path.relpath(dirpath, root_path)
        if rel_dir == ".":
            rel_dir = ""

        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in INDEXABLE_EXTENSIONS_LOCAL:
                continue

            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename
            rel_path = rel_path.replace("\\", "/")
            tree_lines.append(rel_path)
            count += 1
            if count >= max_files:
                tree_lines.append(f"... ({count}+ files, truncated)")
                return "\n".join(tree_lines)

    return "\n".join(tree_lines)


def retrieve_local_files(
    root_path: str,
    file_paths: List[str],
    max_chars_per_file: int = 100000,
) -> List[Dict]:
    """Read files directly from the local file system (no cache needed)."""
    chunks = []
    root_path = os.path.abspath(root_path)

    for fpath in file_paths:
        # Resolve the file path against root
        abs_path = os.path.join(root_path, fpath)
        if not os.path.isfile(abs_path):
            # Try as-is (might already be absolute or differently rooted)
            if not os.path.isfile(fpath):
                print(f"   [LocalRetriever] File not found: {fpath}")
                continue
            abs_path = fpath

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"   [LocalRetriever] Error reading {fpath}: {e}")
            continue

        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n... [truncated]"

        lines = content.split("\n")
        rel_path = os.path.relpath(abs_path, root_path).replace("\\", "/")

        chunks.append({
            "file": rel_path,
            "content": f"File: {rel_path}\n\n{content}",
            "start_line": 1,
            "end_line": len(lines),
            "source": "targeted",
            "symbol": "",
        })
        print(f"   [LocalRetriever] Retrieved: {rel_path} ({len(lines)} lines)")

    return chunks


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
    ctx: Optional[PipelineContext] = None,
) -> tuple:
    """
    Skeleton-first pipeline for local file searches.
    Uses LLM to identify relevant files FIRST, reads only those,
    and falls back to full search only when needed.
    Returns (answer: str, full_context: str).
    """
    print("\n[Smart Local Pipeline]")

    # Step 1: Build local file tree
    print("[Step 1/6] Building local file tree...")
    project_structure = build_local_file_tree(search_path)
    file_count = len([l for l in project_structure.splitlines() if not l.startswith("...")])
    print(f"   Found {file_count} indexable files")

    # Step 2: Query Refinement
    print("[Step 2/6] Refining Query...")
    refined_data = llm.refine_user_query(
        question, project_context=project_context, file_structure=project_structure
    )
    query_to_use = refined_data.get("refined_question", question)
    expansion_keywords = refined_data.get("keywords", [])
    is_action_request = refined_data.get("is_action_request", False)

    top_chunks: List[Dict] = []

    if is_action_request:
        print("   [Fast-Path] Action command detected. Skipping search pipeline.")
    else:
        # Step 3: Skeleton Analysis (LLM identifies key files)
        print("[Step 3/6] Skeleton Analysis (identifying relevant files)...")
        targeted_files = llm.identify_relevant_files(
            query_to_use, project_structure, symbol_minimap=None
        )

        # Step 4: Targeted Retrieval (read only those files)
        print("[Step 4/6] Targeted File Retrieval...")
        targeted_chunks: List[Dict] = []
        if targeted_files:
            targeted_chunks = retrieve_local_files(search_path, targeted_files)
            print(f"   Retrieved {len(targeted_chunks)} targeted file(s)")
        else:
            print("   No targeted files identified")

        # Step 5: Ripgrep for keyword matches
        print("[Step 5/6] Keyword Search (Ripgrep)...")
        searcher = ctx.searcher if ctx else SearchTool()
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

        # Merge targeted + keyword results
        top_chunks = list(targeted_chunks)
        seen_files = {c["file"] for c in targeted_chunks}
        for chunk in keyword_chunks:
            if chunk["file"] not in seen_files:
                top_chunks.append(chunk)
                seen_files.add(chunk["file"])

        # If we have enough context, skip heavy FAISS/BM25
        if len(top_chunks) >= 3:
            print(f"   [Smart-Route] Sufficient context ({len(top_chunks)} chunks). Skipping full embedding index.")
            # Light reranking on the smaller result set
            if len(top_chunks) > 5:
                reranker = ctx.reranker if ctx else CrossEncoderReranker()
                top_chunks = (
                    list(targeted_chunks)
                    + reranker.rerank(query_to_use, keyword_chunks, top_k=max(5 - len(targeted_chunks), 2))
                )
        else:
            # Fallback: not enough from skeleton — run full FAISS + BM25
            print(f"   [Fallback] Only {len(top_chunks)} chunks found. Running full search index...")
            vector_tool = ctx.vector_tool if ctx else VectorSearchTool(
                embedding_client=EmbeddingClient(), cache_dir=os.path.join(".cache", "vector_index")
            )
            vector_tool.build_index(search_path, force_rebuild=rebuild_index)
            vector_results = vector_tool.search(query_to_use, top_k=20)

            bm25_tool = ctx.bm25_tool if ctx else BM25SearchTool(cache_dir=os.path.join(".cache", "bm25_index"))
            bm25_tool.build_index(vector_tool.metadata, force_rebuild=rebuild_index)
            bm25_results = bm25_tool.search(query_to_use, top_k=20)

            candidates = reciprocal_rank_fusion(
                [vector_results, bm25_results, keyword_chunks]
            )
            reranker = ctx.reranker if ctx else CrossEncoderReranker()
            search_chunks = reranker.rerank(query_to_use, candidates, top_k=5)

            # Merge with any targeted chunks
            for chunk in search_chunks:
                if chunk["file"] not in seen_files:
                    top_chunks.append(chunk)
                    seen_files.add(chunk["file"])

        print(f"   Final context: {len(top_chunks)} chunks")

    # Step 6: Agentic Action Loop
    print("[Step 6/6] Agentic Action Loop (Synthesizing/Editing)...")
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

