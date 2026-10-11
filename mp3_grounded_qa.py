"""MP3 · Grounded Q&A — A Cited Document Assistant
================================================

Builds on my MP2 Mini-RAG (same Sherlock Holmes corpus, same chunker, same
OpenAI + Qdrant stack). MP3 adds the three things that make it trustworthy:

    1. Citations   -- every claim tagged [1], [2]... and each marker checked
                      against the chunks that were actually retrieved
    2. Abstention  -- "I don't know based on the provided documents" when
                      retrieval finds nothing relevant
    3. Evaluation  -- hit rate / grounded rate / abstention accuracy /
                      hallucination rate over a question set

Run sequence:
    pip install -r requirements.txt
    python mp3_grounded_qa.py selftest    # offline check of the 4 functions, no API key needed
    python mp3_grounded_qa.py ingest      # build the collection (once)
    python mp3_grounded_qa.py ask         # interactive Q&A with [n] citations
    python mp3_grounded_qa.py eval        # the graded table -> also saved to mp3_eval.txt
    python mp3_grounded_qa.py scores      # top retrieval scores per question (for picking the threshold)
    python mp3_grounded_qa.py sweep       # stretch: abstention vs hit rate across thresholds

Still on Windows, so .env is loaded with python-dotenv rather than `source .env`.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──────────────────────────────────────────────────────

BASE_DIR          = Path(__file__).parent
CORPUS_DIR        = BASE_DIR / "corpus"
DATA_DIR          = BASE_DIR / "data"
EVAL_OUT          = BASE_DIR / "mp3_eval.txt"
SWEEP_OUT         = BASE_DIR / "mp3_threshold_sweep.txt"

COLLECTION_NAME   = "mp3_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP     = 80

# MP2 lesson: k=3 kept losing the second half of multi-hop questions (Becher vs.
# the counterfeit press), so MP3 retrieves 5.
TOP_K = 5

# Cosine similarity of the *best* chunk must clear this, otherwise we don't even
# call the LLM. Picked by running `scores` / `sweep` -- see mp3_reflection.md.
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.35"))

ABSTAIN_MSG = "I don't know based on the provided documents."


# Clients are created lazily so `selftest` runs with no key and no network.
_openai = None
_qdrant = None


def get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI()
    return _openai


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
    return _qdrant


# ─── Pipeline (carried over from MP2, unchanged logic) ──────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


def _looks_like_header(block: str) -> bool:
    """Short standalone line with no sentence punctuation = section heading."""
    block = block.strip()
    if not block or "\n" in block or len(block) > 60:
        return False
    return block[-1] not in ".!?\"'"


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Paragraph-aware chunking: ~500 chars, 80-char overlap, section tracked."""
    paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
    if paragraphs and paragraphs[0] == doc["title"]:
        paragraphs = paragraphs[1:]

    chunks: list[dict[str, Any]] = []
    section = doc["title"]
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append({"source": doc["source"], "title": doc["title"],
                           "section": section, "text": buf.strip()})
        buf = ""

    for para in paragraphs:
        if _looks_like_header(para):
            flush()
            section = para
            continue
        if buf and len(buf) + len(para) + 1 > TARGET_CHUNK_SIZE:
            prev = buf
            flush()
            overlap = prev[-CHUNK_OVERLAP:].strip()
            buf = f"{overlap} {para}" if overlap else para
        else:
            buf = f"{buf}\n{para}" if buf else para

    flush()
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = get_openai().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def setup_collection() -> None:
    from qdrant_client.models import Distance, VectorParams
    q = get_qdrant()
    if q.collection_exists(COLLECTION_NAME):
        q.delete_collection(COLLECTION_NAME)
    q.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    from qdrant_client.models import PointStruct
    vectors = embed_texts([c["text"] for c in chunks])
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
              for c, v in zip(chunks, vectors)]
    get_qdrant().upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query: str, k: int = TOP_K) -> list[dict[str, Any]]:
    query_vec = embed_texts([query])[0]
    results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME, query=query_vec, limit=k, with_payload=True,
    ).points
    out = []
    for r in results:
        chunk = dict(r.payload)
        chunk["score"] = r.score
        out.append(chunk)
    return out


# ─── TODO 1: format_context ─────────────────────────────────────────────

def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks [1..n] for the prompt.

    Returns (context_string, sources). sources[i] is what marker [i+1] points
    at, so the two stay in lock-step -- validate_citations relies on that.
    """
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title", c.get("source", "?"))
        section = c.get("section", "")
        label = f"{title} — {section}" if section and section != title else title
        blocks.append(f"[{i}] ({label})\n{c['text'].strip()}")
        sources.append({
            "n": i,
            "source": c.get("source"),
            "title": title,
            "section": section,
            "score": round(float(c.get("score", 0.0)), 4),
            "text": c["text"],
        })
    return "\n\n".join(blocks), sources


# ─── TODO 2: validate_citations ─────────────────────────────────────────

# Handles [1], [1, 3] and [1][2]. Deliberately ignores things like [a] or [12a].
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(answer_text: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull every [n] out of the answer and check it maps to a real source.

    Returns:
        cited    -- sorted unique marker numbers found in the answer
        valid    -- the ones that map to a retrieved chunk (1..len(sources))
        invalid  -- the ones that don't (the fabricated [7] case)
        unused   -- retrieved sources the answer never cited (informational)
        ok       -- True if there's at least one citation and none are invalid
    """
    cited: set[int] = set()
    for group in _CITE_RE.findall(answer_text or ""):
        for num in group.split(","):
            cited.add(int(num.strip()))

    available = {s["n"] for s in sources}
    valid = sorted(n for n in cited if n in available)
    invalid = sorted(n for n in cited if n not in available)
    unused = sorted(available - cited)

    return {
        "cited": sorted(cited),
        "valid": valid,
        "invalid": invalid,
        "unused": unused,
        "ok": bool(valid) and not invalid,
    }


# ─── TODO 3: should_abstain ─────────────────────────────────────────────

def should_abstain(chunks: list[dict[str, Any]], threshold: float | None = None) -> bool:
    """True if there's nothing relevant enough to answer from.

    Only the top score matters -- if the best chunk is below threshold the rest
    are too. Not assuming the list is sorted though, so take max().
    """
    if threshold is None:
        threshold = RETRIEVAL_THRESHOLD
    if not chunks:
        return True
    top = max(float(c.get("score", 0.0)) for c in chunks)
    return top < threshold


def is_abstention(answer_text: str) -> bool:
    """The model can also abstain on its own (gate passed but excerpts don't
    actually answer it) -- treat any answer starting with the phrase as one."""
    norm = (answer_text or "").strip().lower().replace("’", "'")
    return norm.startswith("i don't know based on the provided documents")


# ─── Answer generation ──────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You answer questions about a small set of Sherlock Holmes story
summaries, using ONLY the numbered excerpts you are given.

Rules:
- Every sentence that states a fact must end with the citation marker(s) of the
  excerpt(s) that support it, e.g. "... a swamp adder [2]." or "[1][3]".
- Only use markers that appear in the excerpts. Never invent a number.
- Do not use outside knowledge of the Holmes stories, even if you know it.
- If the excerpts do not contain the answer, reply with exactly:
  {ABSTAIN_MSG}
  and nothing else. Don't guess and don't give a partial answer from memory.
- Keep it short: 2-5 sentences."""


def generate(question: str, chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    context, sources = format_context(chunks)
    user_msg = f"Question: {question}\n\nExcerpts:\n{context}"
    resp = get_openai().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip(), sources


def answer(question: str, k: int = TOP_K, threshold: float | None = None) -> dict[str, Any]:
    """retrieve -> gate -> generate -> validate citations."""
    if threshold is None:
        threshold = RETRIEVAL_THRESHOLD
    start = time.time()
    chunks = retrieve(question, k=k)
    top_score = max((c["score"] for c in chunks), default=0.0)

    if should_abstain(chunks, threshold):
        return {
            "question": question, "answer": ABSTAIN_MSG, "abstained": True,
            "abstain_reason": f"retrieval gate (top score {top_score:.3f} < {threshold})",
            "sources": [], "retrieved": chunks, "top_score": top_score,
            "citations": validate_citations("", []),
            "latency_ms": int((time.time() - start) * 1000),
        }

    text, sources = generate(question, chunks)
    abstained = is_abstention(text)
    return {
        "question": question, "answer": text, "abstained": abstained,
        "abstain_reason": "model said it doesn't know" if abstained else None,
        "sources": sources, "retrieved": chunks, "top_score": top_score,
        "citations": validate_citations(text, sources),
        "latency_ms": int((time.time() - start) * 1000),
    }


# ─── TODO 4: evaluate ───────────────────────────────────────────────────

def evaluate(questions: list[dict[str, Any]],
             ask_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Run every question through ask_fn and score it.

    Each question has `should_abstain` (bool). Answerable ones also have
    `expected_source` and optionally `expected_facts`.

    Metrics (denominators in brackets):
      retrieval_hit_rate     [answerable]  expected_source was retrieved AND we
                                           didn't gate it out
      grounded_response_rate [answered]    answer has >=1 valid citation, no
                                           invalid ones
      abstention_accuracy    [should-abstain] we actually abstained
      hallucination_rate     [answered]    answered something it should have
                                           abstained on, OR answered with no /
                                           fabricated citations
    "answered" = everything the assistant didn't abstain on.
    A metric with an empty denominator comes back as None (printed as n/a).
    """
    rows: list[dict[str, Any]] = []
    for q in questions:
        res = ask_fn(q["question"])
        expect_abstain = bool(q.get("should_abstain", False))
        retrieved_sources = {c.get("source") for c in res.get("retrieved", [])}
        gated = res.get("abstained") and not res.get("sources")
        cit = res.get("citations") or validate_citations(res.get("answer", ""), res.get("sources", []))

        hit = None
        if not expect_abstain:
            hit = (not gated) and q.get("expected_source") in retrieved_sources

        answered = not res.get("abstained", False)
        grounded = answered and cit["ok"]
        hallucinated = answered and (expect_abstain or not cit["ok"])

        facts = q.get("expected_facts", [])
        ans_low = res.get("answer", "").lower()
        facts_hit = sum(1 for f in facts if f.lower() in ans_low)

        rows.append({
            "id": q.get("id", q["question"][:30]),
            "question": q["question"],
            "expect_abstain": expect_abstain,
            "abstained": res.get("abstained", False),
            "abstain_reason": res.get("abstain_reason"),
            "top_score": res.get("top_score", 0.0),
            "hit": hit,
            "grounded": grounded,
            "hallucinated": hallucinated,
            "cited": cit["cited"],
            "invalid": cit["invalid"],
            "facts": f"{facts_hit}/{len(facts)}" if facts else "-",
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "latency_ms": res.get("latency_ms"),
        })

    return summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The four rates from already-scored rows (lets eval merge two question sets)."""
    def rate(num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    answerable = [r for r in rows if not r["expect_abstain"]]
    must_abstain = [r for r in rows if r["expect_abstain"]]
    answered = [r for r in rows if not r["abstained"]]

    metrics = {
        "retrieval_hit_rate": rate(sum(bool(r["hit"]) for r in answerable), len(answerable)),
        "grounded_response_rate": rate(sum(r["grounded"] for r in answered), len(answered)),
        "abstention_accuracy": rate(sum(r["abstained"] for r in must_abstain), len(must_abstain)),
        "hallucination_rate": rate(sum(r["hallucinated"] for r in answered), len(answered)),
    }
    counts = {"questions": len(rows), "answerable": len(answerable),
              "should_abstain": len(must_abstain), "answered": len(answered)}
    return {"metrics": metrics, "counts": counts, "rows": rows}


# ─── Printing ───────────────────────────────────────────────────────────

def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _yn(v: bool | None) -> str:
    return "-" if v is None else ("Y" if v else "N")


def render_report(report: dict[str, Any], title: str, threshold: float) -> str:
    lines = [f"=== {title}  (k={TOP_K}, threshold={threshold}) ===", ""]
    hdr = f"{'id':<34} {'type':<8} {'top':>5}  {'abst':>4} {'hit':>3} {'grnd':>4} {'hall':>4}  {'cites':<10} {'facts':>5}"
    lines += [hdr, "-" * len(hdr)]
    for r in report["rows"]:
        typ = "abstain" if r["expect_abstain"] else "answer"
        cites = ",".join(map(str, r["cited"])) or "-"
        if r["invalid"]:
            cites += f" !{','.join(map(str, r['invalid']))}"
        lines.append(
            f"{r['id'][:34]:<34} {typ:<8} {r['top_score']:>5.3f}  {_yn(r['abstained']):>4} "
            f"{_yn(r['hit']):>3} {_yn(r['grounded'] if not r['abstained'] else None):>4} "
            f"{_yn(r['hallucinated'] if not r['abstained'] else None):>4}  {cites:<10} {r['facts']:>5}"
        )
    m, c = report["metrics"], report["counts"]
    lines += [
        "",
        f"  Retrieval hit rate      {_fmt(m['retrieval_hit_rate'])}   (of {c['answerable']} answerable)",
        f"  Grounded response rate  {_fmt(m['grounded_response_rate'])}   (of {c['answered']} answered)",
        f"  Abstention accuracy     {_fmt(m['abstention_accuracy'])}   (of {c['should_abstain']} should-abstain)",
        f"  Hallucination rate      {_fmt(m['hallucination_rate'])}   (of {c['answered']} answered)",
        "",
    ]
    return "\n".join(lines)


def render_answers(report: dict[str, Any]) -> str:
    out = ["--- Answers ---", ""]
    for r in report["rows"]:
        out.append(f"[{r['id']}] {r['question']}")
        out.append(f"  -> {r['answer']}")
        if r["abstain_reason"]:
            out.append(f"     (abstained: {r['abstain_reason']})")
        for s in r["sources"]:
            if s["n"] in r["cited"]:
                out.append(f"     [{s['n']}] {s['title']} — {s['section']}  ({s['score']:.3f})")
        if r["invalid"]:
            out.append(f"     !! citation(s) {r['invalid']} don't match any retrieved chunk")
        out.append("")
    return "\n".join(out)


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    qs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [q for q in qs if not q["question"].startswith("Replace this")]


# ─── Selftest (offline, synthetic data) ─────────────────────────────────

def selftest() -> None:
    def check(cond: bool, msg: str) -> None:
        if not cond:
            print(f"✗ selftest failed: {msg}")
            sys.exit(1)

    fake = [
        {"source": "a.txt", "title": "Story A", "section": "Opening", "text": "Alpha text.", "score": 0.61},
        {"source": "b.txt", "title": "Story B", "section": "Middle", "text": "Bravo text.", "score": 0.48},
        {"source": "a.txt", "title": "Story A", "section": "End", "text": "Charlie text.", "score": 0.30},
    ]

    # format_context
    ctx, sources = format_context(fake)
    check(len(sources) == 3, "format_context should return one source per chunk")
    check([s["n"] for s in sources] == [1, 2, 3], "sources must be numbered 1..n")
    check("[1]" in ctx and "[3]" in ctx and "Bravo text." in ctx, "context must contain numbered chunk text")
    check(ctx.index("[1]") < ctx.index("[2]") < ctx.index("[3]"), "context order must match sources order")
    check(format_context([]) == ("", []), "empty chunks -> empty context")
    print("  ✓ format_context")

    # validate_citations
    v = validate_citations("The snake did it [1]. It used the rope [1, 2][3].", sources)
    check(v["cited"] == [1, 2, 3] and v["invalid"] == [] and v["ok"], f"normal citations: {v}")
    v = validate_citations("Made up claim [7]. Real one [2].", sources)
    check(v["invalid"] == [7] and v["valid"] == [2] and not v["ok"], f"fabricated [7] not caught: {v}")
    v = validate_citations("No citations here at all.", sources)
    check(v["cited"] == [] and not v["ok"], "uncited answer should not be ok")
    v = validate_citations("Zero is not a source [0].", sources)
    check(v["invalid"] == [0], "[0] should be invalid")
    check(validate_citations("x [1]", [])["invalid"] == [1], "no sources -> every marker invalid")
    print("  ✓ validate_citations")

    # should_abstain
    check(should_abstain([], 0.35) is True, "no chunks -> abstain")
    check(should_abstain(fake, 0.35) is False, "top 0.61 >= 0.35 -> answer")
    check(should_abstain(fake, 0.70) is True, "top 0.61 < 0.70 -> abstain")
    check(should_abstain(list(reversed(fake)), 0.5) is False, "must use max score, not first")
    check(should_abstain([{"text": "x", "score": 0.35}], 0.35) is False, "score == threshold should pass")
    check(is_abstention(ABSTAIN_MSG) and is_abstention("I don’t know based on the provided documents."),
          "abstain phrase detection")
    print("  ✓ should_abstain")

    # evaluate -- canned responses, no API
    canned = {
        "good": {"answer": "It was a snake [1].", "abstained": False, "sources": sources,
                 "retrieved": fake, "top_score": 0.61},
        "fabricated": {"answer": "Thing [9].", "abstained": False, "sources": sources,
                       "retrieved": fake, "top_score": 0.5},
        "gated": {"answer": ABSTAIN_MSG, "abstained": True, "sources": [], "retrieved": fake[2:],
                  "top_score": 0.30},
        "leaked": {"answer": "Moriarty is a professor [2].", "abstained": False, "sources": sources,
                   "retrieved": fake, "top_score": 0.40},
    }
    for v in canned.values():
        v["citations"] = validate_citations(v["answer"], v["sources"])
    qs = [
        {"id": "1", "question": "good", "expected_source": "a.txt", "should_abstain": False},
        {"id": "2", "question": "fabricated", "expected_source": "b.txt", "should_abstain": False},
        {"id": "3", "question": "gated", "expected_source": "a.txt", "should_abstain": False},
        {"id": "4", "question": "gated", "should_abstain": True},
        {"id": "5", "question": "leaked", "should_abstain": True},
    ]
    rep = evaluate(qs, lambda q: canned[q])
    m = rep["metrics"]
    check(m["retrieval_hit_rate"] == round(2 / 3, 3), f"hit rate expected 0.667, got {m['retrieval_hit_rate']}")
    check(m["grounded_response_rate"] == round(2 / 3, 3), f"grounded expected 0.667, got {m['grounded_response_rate']}")
    check(m["abstention_accuracy"] == 0.5, f"abstention expected 0.5, got {m['abstention_accuracy']}")
    check(m["hallucination_rate"] == round(2 / 3, 3), f"hallucination expected 0.667, got {m['hallucination_rate']}")
    empty = evaluate([], lambda q: {})["metrics"]
    check(all(v is None for v in empty.values()), "empty question set -> all None")
    print("  ✓ evaluate")

    print("\n✓ selftest passed")


# ─── CLI ────────────────────────────────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")
    print(f"→ Total chunks: {len(all_chunks)}")
    setup_collection()
    ingest_chunks(all_chunks)
    print("✓ Done. Try: python mp3_grounded_qa.py ask")


def cmd_ask() -> None:
    print(f"Grounded Q&A over the Sherlock Holmes corpus (threshold {RETRIEVAL_THRESHOLD}).")
    print("Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        r = answer(q)
        print(f"\n{r['answer']}\n")
        if r["abstained"]:
            print(f"  (abstained — {r['abstain_reason']})")
        cit = r["citations"]
        for s in r["sources"]:
            mark = "*" if s["n"] in cit["cited"] else " "
            print(f"  {mark}[{s['n']}] {s['title']} — {s['section']}  ({s['score']:.3f})")
        if cit["invalid"]:
            print(f"  !! invalid citation(s): {cit['invalid']}")
        print(f"  top score {r['top_score']:.3f} · {r['latency_ms']}ms\n")


def cmd_eval() -> None:
    pre = load_questions(DATA_DIR / "predefined_questions.jsonl")
    mine = load_questions(DATA_DIR / "learner_questions.jsonl")

    rep_pre = evaluate(pre, answer)
    out = [render_report(rep_pre, "Predefined questions", RETRIEVAL_THRESHOLD)]
    final = rep_pre
    if mine:
        rep_mine = evaluate(mine, answer)
        final = summarize(rep_pre["rows"] + rep_mine["rows"])
        out.append(render_report(final, "Predefined + my questions", RETRIEVAL_THRESHOLD))
    out.append(render_answers(final))

    text = "\n".join(out)
    print(text)
    EVAL_OUT.write_text(text, encoding="utf-8")
    print(f"(saved to {EVAL_OUT.name})")


def cmd_scores() -> None:
    """Retrieval only -- cheap. Shows where the threshold needs to sit."""
    qs = load_questions(DATA_DIR / "predefined_questions.jsonl") + \
         load_questions(DATA_DIR / "learner_questions.jsonl")
    print(f"{'id':<36} {'type':<8} {'top':>6} {'2nd':>6}  top source")
    for q in qs:
        chunks = retrieve(q["question"])
        sc = sorted((c["score"] for c in chunks), reverse=True)
        typ = "abstain" if q.get("should_abstain") else "answer"
        best = max(chunks, key=lambda c: c["score"])
        print(f"{q.get('id','?')[:36]:<36} {typ:<8} {sc[0]:>6.3f} {sc[1] if len(sc) > 1 else 0:>6.3f}  "
              f"{best['source']} / {best['section']}")


def cmd_sweep() -> None:
    """Stretch: one retrieval + one ungated generation per question, then
    re-apply the gate at each threshold offline. ~9 LLM calls total."""
    qs = load_questions(DATA_DIR / "predefined_questions.jsonl") + \
         load_questions(DATA_DIR / "learner_questions.jsonl")
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    cache: dict[str, dict[str, Any]] = {}
    for q in qs:
        chunks = retrieve(q["question"])
        text, sources = generate(q["question"], chunks)
        cache[q["question"]] = {"chunks": chunks, "text": text, "sources": sources,
                                "top": max(c["score"] for c in chunks)}

    def ask_at(th: float) -> Callable[[str], dict[str, Any]]:
        def _ask(question: str) -> dict[str, Any]:
            c = cache[question]
            if should_abstain(c["chunks"], th):
                return {"answer": ABSTAIN_MSG, "abstained": True, "abstain_reason": "gate",
                        "sources": [], "retrieved": c["chunks"], "top_score": c["top"],
                        "citations": validate_citations("", [])}
            return {"answer": c["text"], "abstained": is_abstention(c["text"]),
                    "abstain_reason": "model" if is_abstention(c["text"]) else None,
                    "sources": c["sources"], "retrieved": c["chunks"], "top_score": c["top"],
                    "citations": validate_citations(c["text"], c["sources"])}
        return _ask

    lines = [f"Threshold sweep over {len(qs)} questions (k={TOP_K})", "",
             f"{'threshold':>9}  {'hit':>5} {'grounded':>8} {'abstain':>7} {'halluc':>6}",
             "-" * 42]
    for th in thresholds:
        m = evaluate(qs, ask_at(th))["metrics"]
        lines.append(f"{th:>9.2f}  {_fmt(m['retrieval_hit_rate']):>5} {_fmt(m['grounded_response_rate']):>8} "
                     f"{_fmt(m['abstention_accuracy']):>7} {_fmt(m['hallucination_rate']):>6}")
    lines += ["", "Top retrieval score per question:"]
    for q in qs:
        typ = "abstain" if q.get("should_abstain") else "answer"
        lines.append(f"  {cache[q['question']]['top']:.3f}  {typ:<8} {q.get('id','?')}")
    text = "\n".join(lines)
    print(text)
    SWEEP_OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\n(saved to {SWEEP_OUT.name})")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmds = {"selftest": selftest, "ingest": cmd_ingest, "ask": cmd_ask,
            "eval": cmd_eval, "scores": cmd_scores, "sweep": cmd_sweep}
    fn = cmds.get(sys.argv[1])
    if fn is None:
        print(f"Unknown command: {sys.argv[1]}\n\n{__doc__}")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
