# gitSurf

An AI-powered Agentic CLI tool that performs context-aware, semantic code search and **autonomous file editing** over any local or remote GitHub repository. It treats code interaction as a reasoning problem, navigating your codebase to find answers or execute commands.

## Why This Exists?
Standard RAG (Retrieval Augmented Generation) often fails on code because it misses abstract relationships and specific variable names. This tool uses a **Triple-Hybrid Search** (Semantic + Keyword + Regex) combined with an **Agentic Action Loop** to find the exact lines of code you need, and can safely modify files on your behalf.

## Features (v2.0)

-   **Agentic Action Loop (ReAct)**: The AI reasons about context and can autonomously use tools to read, edit, or create files to fulfill your request.
-   **Fast-Path Action Routing**: Directly executes simple commands (e.g., "Create a file") without running the heavyweight search pipeline.
-   **Triple-Hybrid Search**: Runs 3 search engines in parallel:
    -   **Vector (FAISS)**: For conceptual understanding ("auth logic").
    -   **BM25 (Statistical)**: For keyword relevance.
    -   **Ripgrep (Regex)**: For exact string/pattern matching.
-   **Symbol MiniMap**: Automatically extracts and tracks function signatures, classes, and exported symbols to catch abstract names.
-   **Query Expansion**: Translates vague questions ("how is data saved?") into technical intent ("persistence layer implementation").
-   **Smart Reranking**: A context-aware cross-encoder validates every chunk before the LLM sees it.

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

# reset conversation history (clearing agent memory)
python main.py --reset
```

## Architecture

The system follows a modular architecture managed by `src/orchestrator.py`:
1.  **Fast-Path Evaluation**: Checks if the user is asking a direct command (skips to Step 8) or a search question.
2.  **Load Skeleton & MiniMap**: Context loading.
3.  **Query Expansion**: Intent classification.
4.  **Skeleton Analysis**: Identifying key files.
5.  **Targeted Retrieval**: Fetching full file content.
6.  **Symbol Extraction**: Building call graphs.
7.  **Hybrid Search**: FAISS + BM25 + Ripgrep.
8.  **Merge & Rerank**: Cross-Encoder validation.
9.  **Agentic Action Loop (ReAct)**: LLM decides to answer directly or use tools (like `FileEditorTool`) to modify the codebase.

See [**ARCHITECTURE.md**](ARCHITECTURE.md) for a deep dive.

## Project Structure
-   `main.py`: Thin CLI Entrypoint.
-   `src/orchestrator.py`: Pipeline coordinator and Agentic Action loop.
-   `src/llm_client.py`: Intelligence layer (uses `src/prompts.py`).
-   `src/tools/`:
    -   `file_editor_tool.py`: Allows the agent to read/write files.
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