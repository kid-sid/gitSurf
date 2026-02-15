# gitSurf

An AI-powered CLI tool that performs context-aware, semantic code search over any GitHub repository. It treats code search as a reasoning problem, not just a keyword matching problem.

## Why This Exists?
Standard RAG (Retrieval Augmented Generation) often fails on code because it misses abstract relationships and specific variable names. This tool uses a **Triple-Hybrid Search** (Semantic + Keyword + Regex) combined with an **8-Step Reasoning Pipeline** to find the exact lines of code you need, even in massive or unfamiliar codebases.

## Features (v1.1)

-   **8-Step Reasoning Pipeline**: From Skeleton Analysis to MiniMap-guided retrieval to final answer synthesis.
-   **Symbol MiniMap**: Automatically extracts and tracks function signatures, classes, and exported symbols to catch abstract names.
-   **Triple-Hybrid Search**: Runs 3 search engines in parallel:
    -   **Vector (FAISS)**: For conceptual understanding ("auth logic").
    -   **BM25 (Statistical)**: For keyword relevance.
    -   **Ripgrep (Regex)**: For exact string/pattern matching.
-   **Query Expansion**: Translates vague questions ("how is data saved?") into technical intent ("persistence layer implementation").
-   **Smart Reranking**: content-aware cross-encoder validates every chunk before the LLM sees it.
-   **Markdown Caching**: Flattens repo structure into optimized Markdown for fast, token-efficient analysis.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/gitSurf.git
    cd gitSurf
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Ripgrep** (Required for Regex Search):
    -   **Windows**: `winget install BurntSushi.ripgrep.MSVC`
    -   **macOS**: `brew install ripgrep`
    -   **Linux**: `apt-get install ripgrep`

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your-openai-api-key
    GITHUB_TOKEN=your-github-token (Required for GitHub Search)
    Check example.env for more details.
    ```

## Usage

### 1. GitHub Context-Aware Search (Primary Mode)
Analyzes a remote GitHub repo without cloning the full history.
```bash
python main.py "How is the JWT validation implemented?" --github-repo owner/repo
```

### 2. Local Search
Run against a local directory.
```bash
python main.py "Where is the main entry point?" --path .
```

### 3. Generate Questions
Auto-generate technical questions to help you explore a new codebase.
```bash
python main.py --github-repo owner/repo --suggest
```

### 4. Advanced Options
```bash
# Force rebuild of vector index
python main.py "query" --github-repo owner/repo --rebuild-index

# Skip the verification step where the AI critiques its own answer (faster)
python main.py "query" --github-repo owner/repo --skip-verify

# Clear all cache
python main.py --clear-cache

# Reset conversation history
python main.py --reset

# Use legacy git clone (instead of markdown cache)
python main.py "query" --github-repo owner/repo --clone
```

## Architecture

The system follows an **8-Step Pipeline**:
1.  **Load Skeleton & MiniMap**: Context loading.
2.  **Query Expansion**: Intent classification.
3.  **Skeleton Analysis**: Identifying key files.
4.  **Targeted Retrieval**: Fetching full file content.
5.  **Symbol Extraction**: Building call graphs.
6.  **Hybrid Search**: FAISS + BM25 + Ripgrep.
7.  **Merge & Rerank**: Cross-Encoder validation.
8.  **Synthesis**: Final answer generation.

See [**ARCHITECTURE.md**](ARCHITECTURE.md) for a deep dive.

## Project Structure
-   `main.py`: CLI Orchestrator.
-   `src/llm_client.py`: Intelligence layer (uses `src/prompts.py`).
-   `src/tools/`:
    -   `markdown_repo_manager.py`: GitHub sync & MiniMap builder.
    -   `vector_search_tool.py`: FAISS implementation.
    -   `symbol_extractor.py`: Static analysis.

## Contributing

We welcome contributions! Please follow this workflow:

1. **Fork the repository** on GitHub and clone your fork.
2. **Create a new branch** from `dev`:
    ```bash
    git checkout dev
    git checkout -b feature/your-feature-name
    ```
3. **Make your changes** and commit them.
4. **Commit Messages**: Please follow this format:
    - `feat: ...`
    - `bugfix: ...`
    - `docs: ...`
    - `refactor: ...`
5. **Open a Pull Request** targeting the `dev` branch.