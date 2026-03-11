# Architecture: gitSurf (Agentic Search Tool)

This document outlines the high-level architecture and processing pipeline of the Agentic Search Tool. The system is designed to perform context-aware, semantic code searches across large repositories by combining statistical models with LLM intelligence.

## System Architecture

The tool follows a **Modular Pipeline Architecture**, separating context acquisition, search execution, and answer synthesis.

  ```mermaid
graph TD
    User([User Request]) --> Main[main.py: CLI]
    Main --> Orch[src/orchestrator.py]
    
    Orch -- "Fast-Path (Action)" --> ActionLoop
    Orch -- "Search Request" --> Phase1
    
    subgraph "Phase 1: Context & Intelligence"
        Phase1[Project Skeleton] --> QM[LLM: Query Refinement]
        QM --> Skeleton[LLM: Identify Key Files]
    end
    
    subgraph "Phase 2: Hybrid Retrieval"
        Skeleton --> TR[Targeted Retriever: Full Files]
        QM --> VS[Vector Search: FAISS HNSW]
        QM --> BM25[BM25 Search: Statistical]
        QM --> RG[Grep Search: Keywords]
    end
    
    subgraph "Phase 3: Processing & Action"
        TR & VS & BM25 & RG --> Merge[Deduplication & Reranking]
        Merge --> RE[Cross-Encoder Reranker]
        RE --> ActionLoop[Agentic Action Loop: ReAct]
    end
    
    ActionLoop -- "FileEditorTool" --> FileSys[(Local Files)]
    ActionLoop --> Output([Final Answer / Result])
```

---

## Core Components

### 1. The Entrypoint (`main.py`)
A thin CLI wrapper that parses arguments and initializes the agents.

### 2. The Pipeline (`src/orchestrator.py`)
Handles the routing and heavy lifting. 
-   **Fast-Path Routing**: If the user's intent is identified as a direct command (e.g., "Create a file"), the orchestrator skips the entire search retrieval phase and passes execution straight to the Action Agent.

### 3. Search Workflow (For informational queries)
1.  **Project Skeleton & MiniMap Loading**: Load file tree + `symbol_minimap.json` (signatures, docstrings, keywords).
2.  **Query Expansion**: LLM refines user question into "Technical Intent", checks if it's an action command, and outputs keywords.
3.  **Skeleton Analysis**: LLM identifies 3-8 key files using the MiniMap and file tree.
4.  **Targeted Retrieval**: Full content of identified files is read immediately.
5.  **Symbol & Call Graph Analysis**: Extract symbols and relationships from targeted files.
6.  **Triple-Hybrid Search**: Parallel Vector + BM25 + Ripgrep (regex) search for broader context.
7.  **Merge & Rerank**: Combine targeted files + search results, rerank using Cross-Encoder.
8.  **Agentic Action Loop**: The ReAct agent takes the curated context and decides whether to write a final answer or execute tools to modify files.

### 4. Key Components

-   **`src/orchestrator.py`**: Coordinates the search pipelines and the shared execution loop.
-   **`LLMClient`**: Handles all LLM interactions and intention parsing (`src/prompts.py`).
-   **`FileEditorTool`**: Gives the autonomous agent read/write/edit access to the filesystem.
-   **`MarkdownRepoManager`**: Syncs GitHub repos to `.cache`, builds `symbol_minimap.json`.
-   **`TargetedRetriever`**: Surgically reads files identified by Skeleton Analysis.
-   **`SymbolExtractor` / `CallGraph`**: Static analysis for Python/JS dependencies.
-   **`VectorSearchTool` / `BM25SearchTool`**: Semantic & Keyword search.
-   **`SearchTool` (Ripgrep)**: Regex pattern matching with technical keywords.

### 3. The Search Engine
The tool uses a **Triple-Hybrid Search** strategy to maximize recall:
- **Vector Search (`src/tools/vector_search_tool.py`)**: Uses `text-embedding-3-small` and FAISS (HNSW) for semantic understanding (e.g., finding "auth logic" when searching for "security").
- **BM25 Search (`src/tools/bm25_search_tool.py`)**: A statistical model that finds the most relevant code chunks based on term frequency (TF-IDF).
- **Regex Search (`src/tools/search_tool.py`)**: A high-speed `ripgrep` wrapper that finds literal matches and complex patterns.

### 4. Data & Retrieval Tools
- **Targeted Retriever (`src/tools/targeted_retriever.py`)**: Specifically designed to bypass search limitations by loading the entire content of small-to-medium files (up to 100k characters).
- **Markdown Repo Manager (`src/tools/markdown_repo_manager.py`)**: Efficiently fetches remote GitHub repositories using GraphQL batching and caches them locally as flattened Markdown.

---

## Data Flow
1. **Discovery**: The tool reads the `README.md` and file structure to understand the "big picture."
2. **Focus**: Instead of searching everything, it "targets" likely files (e.g., `config.py` for settings).
3. **Retrieval**: It gathers broad semantic matches (Vector) and exact matches (BM25/Grep).
4. **Filtering**: A **Cross-Encoder Reranker** verifies the relevance of each snippet against the actual question.
5. **Grounding**: The final LLM call is "clamped" to the provided context to prevent hallucinations.
