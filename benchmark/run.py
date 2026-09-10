from __future__ import annotations

import json
from pathlib import Path

from scorer import CausalLMScorer


CASES = Path(__file__).with_name("cases.jsonl")


def load_cases() -> list[dict]:
    with CASES.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(scorer: CausalLMScorer, mode: str, cases: list[dict]) -> None:
    correct = 0
    print(f"\n=== {mode} ===")

    for index, case in enumerate(cases, start=1):
        ranked = scorer.rank(case["context"], case["candidates"], mode=mode)
        predicted = ranked[0].candidate
        ok = predicted == case["expected"]
        correct += int(ok)

        ranking = " > ".join(
            f"{item.candidate}({item.context_gain:.3f})"
            if mode == "context_gain"
            else f"{item.candidate}({item.conditional_avg_logprob:.3f})"
            for item in ranked
        )
        print(
            f"{index:02d} {'OK' if ok else 'XX'}  "
            f"[{case['context']}] + {case['pinyin']}  =>  {ranking}"
        )

    accuracy = correct / len(cases) if cases else 0.0
    print(f"Top-1: {correct}/{len(cases)} = {accuracy:.1%}")


def main() -> None:
    cases = load_cases()
    scorer = CausalLMScorer()
    print(f"Model: {scorer.model_name}")
    print(f"Cases: {len(cases)}")
    evaluate(scorer, "conditional", cases)
    evaluate(scorer, "context_gain", cases)


if __name__ == "__main__":
    main()
