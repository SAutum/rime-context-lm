# Rime integration notes

Rime stays responsible for:

```text
keyboard input -> pinyin parsing -> dictionary/candidate generation -> candidate objects
```

This project experiments only with the contextual scoring/ranking stage.

## What we need from Rime

For each composition event we eventually need:

```json
{
  "context": "刚刚通过输入法提交的前文",
  "pinyin": "dangqianpinyin",
  "candidates": ["候选一", "候选二", "候选三"],
  "rime_order": ["候选一", "候选二", "候选三"]
}
```

For an Octagram comparison we should additionally preserve the order produced with `contextual_suggestions` enabled.

## Why there is no synchronous HTTP Lua filter yet

`scorer/server.py` exposes HTTP intentionally as an experiment/debug interface. Spawning `curl` or synchronously crossing a Python HTTP boundary on every keyboard update would distort the latency result and is not a good final Rime architecture.

The order of work is therefore:

1. prove ranking quality offline;
2. capture real Rime candidate sets and Octagram rankings;
3. measure warm neural inference latency;
4. choose the live integration mechanism only after the model is worth integrating.

## Intended live pipeline

```text
Rime translator
     |
     v
candidate set
     |
     v
context LM scorer
     |
     v
adjust candidate quality / order
     |
     v
Rime UI
```

A live implementation should ideally keep the model loaded in-process or in a very low-overhead local runtime. A native librime plugin / ONNX-style runtime is the likely direction if the Phase 1 benchmark is positive.

## Context scope

Start with context already known to Rime (recent committed text / commit history). Do not make access to arbitrary text in Word, Chrome, etc. a Phase 1 dependency. That can be investigated separately at the Windows TSF/front-end layer later.
