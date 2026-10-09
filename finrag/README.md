# FinRAG — Financial Research Assistant Powered by RAG

FinRAG is a Streamlit portfolio project that answers questions about company financial filings and revenue trends using **retrieval-augmented generation (RAG)**. It combines local document retrieval with web-search fallback and uses a Groq-hosted language model to synthesize evidence-grounded answers with source citations.

The project is designed to demonstrate practical RAG engineering—not just a chat interface around an LLM—including query decomposition, hybrid retrieval, reranking, confidence gating, context management, source tracing, and financial comparison workflows.

## Features

- **Financial Q&A:** Ask questions about the companies and filing excerpts covered by the local corpus.
- **Multi-company question decomposition:** Splits comparison questions into company-scoped searches to reduce cross-company evidence contamination.
- **Hybrid retrieval:** Combines TF-IDF ranking with a custom BM25 implementation and Reciprocal Rank Fusion (RRF).
- **Financial-aware reranking:** Reorders candidate evidence using signals such as number/year overlap and sentence-level relevance.
- **Confidence gating:** Checks retrieval evidence before generation, helping the app avoid answering from weak local matches alone.
- **Web-search fallback:** Uses Tavily to find additional evidence when local filing evidence is insufficient.
- **LLM answer synthesis:** Uses the Groq API to generate explanations, comparison tables, growth calculations, rankings, and follow-up responses from the supplied context.
- **Source citations:** Keeps source details visible so users can inspect the evidence behind an answer.
- **Conversation context:** Supports follow-up questions while keeping the history sent to the model bounded.
- **Revenue Explorer:** Provides interactive Plotly charts from the revenue series defined in the application, covering 12 selected companies.
- **Graceful handling of large requests:** Bounds the context sent to the model and includes a smaller-request retry path for certain Groq request-size/rate-limit errors.

## Architecture

```text
User question
     |
     v
Query decomposition (decompose.py)
     |
     v
Hybrid retrieval (retrieval.py)
  ├── TF-IDF
  ├── BM25
  └── Reciprocal Rank Fusion (RRF)
     |
     v
Financial-aware reranking (rerank.py)
     |
     v
Confidence gate (gate.py)
     |
     ├── Local evidence sufficient ─────────┐
     |                                      |
     └── Local evidence insufficient        |
                |                           |
                v                           |
          Tavily web search                 |
                |                           |
                └───────────┬───────────────┘
                            v
                  Context-bounded generation
                     through Groq API
                            |
                            v
                 Answer with source citations
```

### Main components

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, application flow, Groq answer generation, chat history, Revenue Explorer, and About tab |
| `decompose.py` | Rule-based company detection and company-scoped query decomposition |
| `retrieval.py` | TF-IDF and BM25 retrieval combined using Reciprocal Rank Fusion |
| `rerank.py` | Financial-aware heuristic reranking of retrieved evidence |
| `gate.py` | Retrieval confidence checks and citation formatting |
| `web_search.py` | Tavily client setup, web search, and web-source formatting |
| `data/processed/chunks.json` | Local processed chunks used by the retrieval pipeline |
| `.env` | Local API-key configuration; **do not commit this file** |
| `requirements.txt` | Python dependencies |

The application uses the rule-based decomposition and heuristic reranking paths shown in the code. The retrieval implementation combines TF-IDF and BM25; it does not currently use a vector database or a dense embedding model.

## Technology stack

- Python
- Streamlit
- Groq API and a configured Groq chat model
- TF-IDF and scikit-learn
- BM25 implemented in Python
- Reciprocal Rank Fusion (RRF)
- Tavily Search API
- Pandas
- Plotly
- python-dotenv

## Requirements

- Python 3.10 or newer is recommended.
- A Groq API key for LLM-generated answers.
- A Tavily API key for web-search fallback.
- The project files and processed corpus, including `data/processed/chunks.json`.

API access and rate limits depend on your provider account and selected model. FinRAG bounds the context it sends to the LLM, but very large questions or provider limits can still affect the response.

## Setup and run locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd finrag
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API keys

Create a file named `.env` in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get API keys from:

- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com/

Keep the real values private. **Never commit `.env`, API keys, or other secrets to GitHub.** If the app is run without one of the keys, the corresponding live functionality may be unavailable; the app displays configuration guidance.

### 5. Start FinRAG

```bash
streamlit run app.py
```

Streamlit will print a local URL—usually `http://localhost:8501`—to open in your browser.

## Example questions

Try questions such as:

- What was Apple's total net sales in fiscal year 2024? Show the amount in billions and cite the source.
- Compare Apple's and Microsoft's revenue in fiscal years 2023 and 2024. Include absolute and percentage growth.
- Compare Apple, Microsoft, NVIDIA, Amazon, and Alphabet over fiscal years 2022–2024. Show revenue, year-over-year growth, cumulative growth, rankings, and cited business drivers.
- Which covered company had the largest revenue increase over a specified period?
- Explain the business drivers behind a company's revenue trend using the available filing evidence.

Results depend on which relevant evidence is present in the local corpus and what the web-search provider returns. Fiscal-year end dates can differ between companies, so comparisons should make those differences explicit where relevant. Treat calculated values as calculations from cited reported figures, not as separately reported metrics.

## Revenue Explorer

The **Revenue Explorer** tab displays interactive charts from revenue values stored in `app.py`. These chart values are a predefined series, separate from the live RAG retrieval flow. Check the underlying source and reporting period before using a chart value as a financial reference.

Banks may report net revenue on a basis that is not directly comparable with the gross revenue measures commonly used by non-financial companies. FinRAG flags this distinction when relevant companies are plotted together.

## Security and responsible use

- Store credentials in `.env` locally or in the hosting platform's secrets manager.
- Add `.env` and `.venv/` to `.gitignore`.
- Do not hard-code API keys into source files.
- Financial answers are generated from retrieved evidence and can still contain omissions or arithmetic mistakes. Verify important figures against the cited filings.
- Web-search results depend on Tavily availability, query coverage, and source quality.
- API requests are subject to model-provider context and rate limits.

FinRAG is an educational and portfolio demonstration, not investment advice or a substitute for reviewing official filings.

## Known limitations

- Retrieval quality depends on the contents and coverage of `data/processed/chunks.json`.
- TF-IDF and BM25 are lexical retrieval methods. They can miss relevant passages when the wording differs substantially from the source.
- Heuristic reranking and a retrieval confidence gate cannot fully determine whether a source supports every nuance of a claim.
- Web fallback can add relevant evidence, but it does not guarantee that every requested figure or explanation will be found.
- LLM output is probabilistic. Long multi-company questions may need a smaller scope if the provider's request or rate limits are reached.
- The Revenue Explorer uses predefined values in `app.py`; it is not automatically refreshed from SEC filings.

## Future improvements

Potential extensions include a labeled retrieval evaluation set, automated faithfulness checks, dense embeddings, richer metadata filters, more robust numeric validation, and automated updates to the source corpus. These are possible future directions, not claims about features already implemented.

## License

No license is specified here. Add a `LICENSE` file if you want to explicitly permit reuse of the project.
