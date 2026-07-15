# miniRAG

A simple RAG (Retrieval-Augmented Generation) system to help you understand the core concepts and implementation principles.

## Architecture

```mermaid
flowchart TB
    subgraph interface["User / Interface Layer"]
        direction LR
        cli["CLI query tools<br/>query_docs_online/offline.py"]
        clients["API clients<br/>SSE stream consumers"]
        indexers["Indexing CLIs<br/>scripts/index_docs_*.py"]
    end

    subgraph application["Application Layer"]
        direction LR
        api["FastAPI server<br/>GET /query (SSE)"]
        wiring["Dependency wiring<br/>api/deps.py"]
        settings["Settings<br/>pydantic-settings + .env"]
    end

    subgraph logic["AI / Logic Layer"]
        direction TB

        subgraph rag["RAG pipeline (production)"]
            direction LR
            transform["Query transform<br/>Identity / HyDE"]
            retrieve["Retrieve + rerank<br/>Chroma top-k / Vector / CrossEncoder"]
            generate["Generate<br/>Chat history + event stream"]
            transform --> retrieve --> generate
        end

        subgraph agent["agent101 (experimental orchestrator)"]
            direction LR
            classifier[Classifier]
            retriever[Retriever]
            answer[Answer]
            verifier[Verifier]
            classifier --> retriever --> answer --> verifier
        end
    end

    subgraph data["Data Layer"]
        direction LR
        chroma["Chroma vector store<br/>Persistent: data/chroma/"]
        corpus["Document corpus<br/>Markdown: data/raw/"]
        cache["Embedding cache<br/>temp/ + MiniLM-L6-v2"]
    end

    subgraph external["External Services"]
        direction LR
        openrouter["OpenRouter API<br/>LLM: glm-5.2<br/>Embeddings: gemini-embedding-2"]
        sentence_transformers["sentence-transformers<br/>Local fallback embeddings"]
    end

    cli --> api
    clients --> api
    api --> wiring
    settings --> wiring
    wiring --> transform

    indexers --> corpus
    indexers --> chroma
    retrieve --> chroma
    retriever --> chroma
    transform --> openrouter
    generate --> openrouter
    answer --> openrouter
    verifier --> openrouter
    retrieve --> sentence_transformers
    indexers --> cache
```
