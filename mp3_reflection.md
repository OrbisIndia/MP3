# MP3 Reflection

Final config: `k=5`, `RETRIEVAL_THRESHOLD=0.35`, gpt-4o-mini at temperature 0.
Full output is in `mp3_eval.txt` and `mp3_threshold_sweep.txt`.

## Grounding & abstention

All three should-abstain questions abstained (Moriarty, Reichenbach, and my
Oakshott one), and none of the six answerable ones got refused. So the
numbers are clean.

How it got there wasn't what I expected, though. **The retrieval gate never
fired once.** Every abstention says `(abstained: model said it doesn't know)`:
the question passed the score check, went to the LLM, and the prompt rule
("if the excerpts don't contain the answer, reply with exactly …") is what
stopped it.

The top scores show why the gate can't do this job on this corpus:

| question | should | top score |
|---|---|---|
| p5 Moriarty | abstain | 0.473 |
| p1 Julia Stoner | answer | **0.514** |
| p6 Reichenbach | abstain | **0.526** |
| l3 Oakshott's husband | abstain | 0.593 |
| p3 Becher / press | answer | 0.622 |
| … other answerable | answer | 0.623 – 0.711 |

The abstain questions and the answerable ones overlap. Reichenbach (0.526)
scores *higher* than "What killed Julia Stoner?" (0.514), and my Oakshott
question (0.593) beats three of the answerable ones. No single threshold can
separate them. Anything above 0.514 to catch Reichenbach also throws away
the Julia question.

So I kept the threshold at 0.35. It sits well under the lowest answerable
score (0.514), so it never costs a real answer, and its only job is to stop
the LLM call for questions that have nothing to do with the corpus. My
question set doesn't actually test that case: all nine questions are
Holmes-flavoured, so none of them got near 0.35. The real abstention
behaviour comes from the prompt.

The Oakshott question was meant to be a near-miss: the Blue Carbuncle summary
says Mrs Oakshott's "husband bred geese" but never names him. It retrieved
the right story with a high score and the model still said it didn't know,
rather than inventing a name. That's the case I was most worried about, and
the one I'd put in the demo.

## Citations

`validate_citations` caught nothing: 0 invalid markers across all 9 answers.
Every `[n]` mapped to a retrieved chunk. I tested the fabricated-marker path
in `selftest` (`[7]` and `[0]` against 3 sources both get flagged), so the
check works. gpt-4o-mini at temperature 0 just didn't invent any numbers
with the "only use markers that appear in the excerpts" rule.

The problem it *can't* catch is a citation that exists but doesn't really
support the claim. In p2 the answer says Spaulding's "real identity is that
of John Clay, a criminal … [4]", and [4] is the **Capture** chunk. That chunk
shows Clay coming up through the bank floor, but the Spaulding = Clay
reveal is in "The Identity of the Assistant" section. The citation is valid
by my check but only half-supports the sentence. `validate_citations` checks
that the marker points at a real chunk, not that the chunk says the thing.

Also some over-citing: p4 tags one short sentence with `[1][2][3]`. Not
wrong, but it's the model covering itself rather than being precise.

## Reading your eval table

| metric | value |
|---|---|
| Retrieval hit rate | 1.00 (6/6) |
| Grounded response rate | 1.00 (6/6) |
| Abstention accuracy | 1.00 (3/3) |
| Hallucination rate | 0.00 (0/6) |

Four perfect numbers on 9 questions mostly tells me the metrics are too
lenient, not that the system is perfect. The weak spot is the facts column,
which isn't one of the four:

- **p1 Julia Stoner: 0/3.** The answer says "a snake trained by her stepfather"
  and never names the swamp adder, the ventilator, or the bell-rope. It cited
  only the two "The Explanation" chunks (0.42 each), not "The Vigil" where
  "swamp adder" actually appears. Grounded, yes. Complete, no. And grounded
  rate still scores it 1.
- **l1: 1/3**, but that one's my substring check. The answer says "thirty
  thousand pounds in French gold" and I'd listed "napoleons" / "Bank of France".
  Same phrasing issue I hit in MP2.

What I'd change: grounded response rate should mean "the cited chunk
supports the claim", not just "the marker exists". A cheap version would be
one extra gpt-4o-mini call per answer asking whether chunk [n] supports
sentence X (yes/no). That would have flagged the p2 Capture citation. I'd also
add many more near-miss abstain questions like Oakshott, since that's where
the real risk is, and 3 of them isn't enough to trust a 1.00.

## One surprise

How high the "not in the corpus" questions scored. The brief's example has
Moriarty at around 0.28, so I expected the gate to do most of the work. Mine
came in at 0.473, and Reichenbach at 0.526. In hindsight it makes sense: on a
corpus that's all Sherlock Holmes, any Holmes question is "on topic" to an
embedding model. Cosine similarity measures whether the question is about
the same stuff, not whether the answer is in there. Once I saw that, the
threshold felt less like the main lever and more like a backstop.

A smaller, nicer one: the k=3 → 5 change from my MP2 reflection paid off.
The Becher / counterfeit question that only got 1/3 facts in MP2 now gets 3/3,
citing two different chunks from "The Coining Operation".

## Stretch: threshold sweep

`sweep` over 0.20–0.50 is completely flat: 1.00 / 1.00 / 1.00 / 0.00 at every
value. Nothing answerable scores below 0.514, and the model handles the
abstains regardless, so moving the gate in that range changes nothing.

The tradeoff only shows up above 0.50 (I've since extended the sweep range
to 0.70). Working it out from the top scores above, since generation is
cached and only the gate changes:

| threshold | hit rate | abstention acc. | what got gated |
|---|---|---|---|
| ≤ 0.50 | 1.00 | 1.00 | nothing that matters (Moriarty at 0.50) |
| 0.55 | 0.83 | 1.00 | + Julia (0.514), Reichenbach |
| 0.60 | 0.83 | 1.00 | + Oakshott |
| 0.65 | 0.50 | 1.00 | + Becher (0.622), Adler marriage (0.623) |
| 0.70 | 0.17 | 1.00 | everything but l2 (0.711) |

Abstention accuracy can't go up because it's already 1.00, so raising the
threshold only costs hit rate. The sweet spot is anywhere from 0.35 up to
just under 0.51. Below that is fine on this set but gives no protection
against truly off-corpus questions. Above it, you start refusing real questions
to "catch" abstains that the prompt was already catching. On a mixed corpus,
where off-topic questions really do score low, I'd expect the curve to
look very different.
