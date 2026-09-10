# Phase 1 experiment

## Question

Can a small neural Chinese causal language model improve the ranking of **the same candidates already produced by Rime** compared with Rime + Octagram?

The first phase intentionally does not change pinyin segmentation, dictionary lookup, candidate generation, learning from user choices, or the UI.

## Unit of evaluation

Each example contains:

```text
context:    text committed immediately before the current composition
pinyin:     current keyboard input
candidates: candidates produced by Rime
expected:   candidate the user intended
```

The neural scorer sees `context` and each candidate. In the first baseline, pinyin is metadata because Rime has already conditioned the candidate set on pinyin.

## Scores

### Conditional LM score

For candidate `c` and context `C`:

```text
S_cond(c, C) = mean log P(tokens(c) | C)
```

Mean token log probability is used so that longer candidates are not automatically punished simply for containing more model tokens.

### Context gain

```text
S_gain(c, C) = S_cond(c, C) - mean log P(tokens(c))
```

This asks a slightly different question: *how much does this particular context increase support for this candidate?*

It is useful because raw LM probability can otherwise over-reward globally common candidates.

## Baselines

We ultimately want three columns for every real Rime case:

```text
Rime without Octagram
Rime + Octagram
Rime candidates + neural LM reranking
```

For the neural column, record both `conditional` and `context_gain` rankings.

## Metrics

Start simple:

- Top-1 accuracy
- Mean reciprocal rank (later)
- rank change relative to Octagram
- scorer latency for one candidate set

The most important latency is not cold model startup. It is warm inference latency while typing.

## Data collection order

1. Use the small hand-written cases in `benchmark/cases.jsonl` only as a smoke test.
2. Capture real Rime candidate sets where the first choice is wrong or ambiguous.
3. Add the user's intended choice as `expected`.
4. Preserve the original Rime/Octagram order so the comparison is reproducible.
5. Build a larger unbiased sample after the pipeline works.

## Integration boundary

Do **not** call a Python HTTP service synchronously for every keystroke and declare that the final architecture. The API in `scorer/server.py` is for experimentation.

If the neural scorer is useful, the likely production paths are:

- native librime plugin with an embedded inference runtime;
- ONNX Runtime / another compact local runtime;
- or an asynchronous/local cached bridge if latency measurements prove it acceptable.

The model should initially be a scorer, not a text generator.

## Relation to ACL 2022

The ACL 2022 paper *Investigating Chinese Pinyin Input Method in the Age of Neural Language Models* demonstrates that GPT-style Chinese language models are useful for Pinyin IME decoding. Our first experiment is narrower: Rime supplies legal candidates and the neural LM only estimates contextual preference among them.
