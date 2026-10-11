# MP3 · Grounded Q&A — A Cited Document Assistant

My MP2 Mini-RAG, made trustworthy. Same 5 Sherlock Holmes summaries, same
chunker, same stack (OpenAI `text-embedding-3-small` + `gpt-4o-mini`, Qdrant) —
plus three things on top:

- **Citations** — every claim in the answer ends with `[n]`, and `[n]` maps to the
  n-th retrieved chunk. `validate_citations` checks each marker against what was
  actually retrieved and flags any that don't exist (the "model invents `[7]`" case).
- **Abstention** — if the best chunk scores below `RETRIEVAL_THRESHOLD`, the LLM
  isn't called at all and the answer is *"I don't know based on the provided
  documents."* The prompt also tells the model to say exactly that when the
  excerpts don't contain the answer, so there are two layers.
- **Evaluation** — `eval` runs a question set and prints retrieval hit rate,
  grounded response rate, abstention accuracy and hallucination rate.

No LangChain / LlamaIndex, no new dependencies over MP2.

## Results

| Metric | 9 questions (k=5, threshold 0.35) |
|---|---|
| Retrieval hit rate | 1.00 |
| Grounded response rate | 1.00 |
| Abstention accuracy | 1.00 |
| Hallucination rate | 0.00 |

Full table in [`mp3_eval.txt`](mp3_eval.txt), threshold sweep in
[`mp3_threshold_sweep.txt`](mp3_threshold_sweep.txt), and what those numbers
do and don't show in [`mp3_reflection.md`](mp3_reflection.md).

## Demo

📹 [2–3 min demo video](VIDEO_LINK_HERE) — two cited answers and one correct "I don't know".

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill it in. I'm on Windows, so the script loads
`.env` with `python-dotenv` instead of `source .env`.

## Running it

```bash
python mp3_grounded_qa.py selftest   # offline, no API key — checks the 4 functions
python mp3_grounded_qa.py ingest     # build the mp3_sherlock collection (once)
python mp3_grounded_qa.py ask        # interactive, shows [n] sources + scores
python mp3_grounded_qa.py eval       # the graded table, also written to mp3_eval.txt
python mp3_grounded_qa.py scores     # just the retrieval scores per question (cheap)
python mp3_grounded_qa.py sweep      # stretch: metrics across thresholds -> mp3_threshold_sweep.txt
```

The threshold can be overridden without editing code:
`RETRIEVAL_THRESHOLD=0.4 python mp3_grounded_qa.py eval` (or set it in `.env`).

## The four functions

| Function | What it does |
|---|---|
| `format_context` | Numbers the retrieved chunks `[1]..[n]` with title + section, returns a parallel `sources` list so `[i]` ↔ `sources[i-1]` |
| `validate_citations` | Regex-parses `[1]`, `[1, 3]`, `[1][2]`; splits into valid / invalid / unused; `ok` only if there's ≥1 valid and 0 invalid |
| `should_abstain` | `True` when there are no chunks or `max(score) < threshold` |
| `evaluate` | Runs each question, scores it, returns the four rates + per-question rows |

## How the metrics are defined

| Metric | Denominator | Counts as a success when |
|---|---|---|
| Retrieval hit rate | answerable questions | the expected story was in the top-k **and** the gate let it through |
| Grounded response rate | questions it answered | ≥1 valid citation and no invalid ones |
| Abstention accuracy | should-abstain questions | it said "I don't know…" |
| Hallucination rate | questions it answered | (failure) it answered a should-abstain question, or answered with no / fabricated citations |

"Answered" = didn't abstain. If a denominator is empty the metric prints `n/a`.

## Questions

`data/predefined_questions.jsonl` — 4 answerable + 2 that must abstain (Moriarty,
Reichenbach Falls — both real Holmes things, neither in these 5 summaries).

`data/learner_questions.jsonl` — mine:

- **l1** (multi-hop, Red-Headed League) — total paid to Wilson *and* what was in the
  vault. The two facts are in different sections.
- **l2** (Scandal in Bohemia) — where the photo was hidden and the smoke-rocket trick.
- **l3** (must abstain) — the first name of Mrs Oakshott's husband. A near-miss on
  purpose: the Blue Carbuncle summary mentions her husband bred geese but never
  names him, so retrieval scores high and only the model-level abstention can catch it.

## Changes from MP2

- `TOP_K` 3 → 5. In MP2 the multi-hop Becher/counterfeit question kept losing one
  of its two chunks at k=3.
- Clients are created lazily, so `selftest` runs with no key or network.
- Separate collection name (`mp3_sherlock`) so it doesn't clobber the MP2 one.

## Files

| File | What it is |
|---|---|
| `mp3_grounded_qa.py` | Pipeline + the 4 functions + CLI |
| `corpus/*.txt` | 5 story summaries (unchanged from MP2) |
| `data/predefined_questions.jsonl` | 4 answerable + 2 abstain |
| `data/learner_questions.jsonl` | My 3 questions |
| `mp3_eval.txt` | Saved output of `eval` |
| `mp3_threshold_sweep.txt` | Saved output of `sweep` (stretch) |
| `mp3_reflection.md` | Write-up |
