# Agentic RFP Evaluation and Supplier Ranking

A Streamlit + SQLite mini-project implementing the uploaded classroom brief. The LLM evaluates each supplier independently and returns evidence-grounded JSON. Python alone performs validation, weighted arithmetic, peer benchmarks, PPI, deterministic tie-breaks, and ranking.

## Project structure

```text
agentic_rfp_solution/
├── app.py
├── requirements.txt
├── seed.sql
├── README.md
├── .streamlit/
│   └── secrets.toml.example
├── data/
│   └── .gitkeep
├── sample_pdfs/
│   └── (put the four or more synthetic supplier PDFs here)
└── tests/
    └── test_scoring.py
```

## Requirements

- Python 3.10+
- An OpenAI-compatible API key for live LLM evaluation
- Internet access to install packages and call the configured LLM

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The application creates `data/rfp_evaluation.db` and seeds criteria on first start. Set your key in the shell before running:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py

# macOS/Linux
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```

For Streamlit Community Cloud, add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in **App → Settings → Secrets**. Never commit secrets to GitHub.

## Criteria and formulas

Default active criteria from the brief total 100%:
- Technical Capability — 30%
- Implementation Plan — 20%
- Commercial Value — 20%
- Security & Compliance — 20%
- Support & Experience — 10%

Absolute weighted score (0–100):

`sum((criterion score / criterion max score) * criterion weight)`

Criterion benchmark: maximum valid score for that criterion among evaluated suppliers.

Criterion gap: `supplier score - benchmark`.

Relative performance: `(supplier score / benchmark) * 100`. If benchmark is zero, this implementation assigns 100% to every supplier for that criterion because no supplier has a positive score to compare against; this avoids division by zero and avoids unfairly giving everyone zero relative performance.

PPI: weighted average of criterion relative-performance percentages, divided by 100 because weights are percentage values.

Tie-break order (exactly as the brief specifies):
1. Higher PPI
2. Earlier submission date
3. Higher historical experience rating
4. Supplier name ascending (case-insensitive)

Sequential ranks are assigned after sorting. Python owns all formulas and ordering.

## Validation behavior

- Requires valid JSON object from the LLM.
- Missing active criterion: inserted with score 0 and a warning.
- Invalid/non-finite score: set to 0 and a warning.
- Score outside range: clipped to 0–max score and a warning.
- Unknown/inactive criterion IDs: ignored with a warning.
- Invalid/missing evidence or justification: retained as blank strings and warnings where type is invalid.
- Active weights must sum to exactly 100% before a run.
- Scanned PDFs with no extractable text fail clearly; OCR is not included in this MVP.

## Agentic workflow

1. Streamlit loads active criteria from SQLite.
2. User uploads multiple PDFs and enters supplier name, date, and experience rating.
3. Orchestrator creates a unique `RFP_RUN_ID`.
4. PyMuPDF extracts text per PDF.
5. Prompt builder supplies current criteria and one supplier document to the LLM.
6. Validation normalizes the JSON result.
7. Deterministic Python calculates scores, peer metrics, PPI, tie-breaks, and ranks.
8. SQLite persists the full JSON result and summary columns.
9. Streamlit displays leaderboard, scorecard, warnings, run history, and JSON download.

## Deployment to Streamlit Community Cloud

1. Create a GitHub repository and push this project.
2. Make sure `app.py`, `requirements.txt`, and `seed.sql` are in the repository root.
3. In Streamlit Community Cloud, deploy the repository with `app.py` as the entry point.
4. Add the API key under app Secrets.
5. Test with at least four synthetic PDFs.
6. Download one complete result JSON and capture screenshots for the submission.

Note: Streamlit Community Cloud's local filesystem may not be durable across restarts/redeploys. SQLite is suitable for the classroom MVP, but persistent production deployment should use durable storage.

## Testing

Run:

```bash
pytest -q
```

The tests cover weighted scoring, peer benchmarking, tie-break ordering, and malformed/missing LLM results.

## Limitations and responsible use

- This is a classroom MVP, not a production procurement decision system.
- LLM scores can vary by model/version; temperature is set to zero but outputs are not guaranteed deterministic.
- Scores should be reviewed by procurement staff, especially where evidence is missing or ambiguous.
- Supplier documents are untrusted input. The prompt instructs the model not to follow embedded instructions.
- Accuracy and fairness require evaluation on representative documents and human review.
