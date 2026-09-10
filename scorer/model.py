from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "uer/gpt2-chinese-cluecorpussmall"


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    conditional_logprob: float
    conditional_avg_logprob: float
    prior_logprob: float
    prior_avg_logprob: float
    context_gain: float

    def to_dict(self) -> dict:
        return asdict(self)


class CausalLMScorer:
    """Score existing Rime candidates with a causal Chinese LM.

    Important: this class never generates candidate text. Rime remains the source
    of truth for pinyin parsing and candidate generation.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.getenv("RIME_CONTEXT_LM_MODEL", DEFAULT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise RuntimeError("Tokenizer has neither pad_token nor eos_token.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.prefix_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if self.prefix_token_id is None:
            raise RuntimeError("A BOS or EOS token is required to score the first token.")

        self.max_positions = int(
            getattr(self.model.config, "n_positions", 0)
            or getattr(self.model.config, "max_position_embeddings", 0)
            or 1024
        )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _batch_scores(self, context: str, candidates: list[str]) -> list[tuple[float, float]]:
        """Return (sum_logprob, mean_logprob) for each candidate given context."""
        if not candidates:
            return []

        context_ids = self._encode(context)
        sequences: list[list[int]] = []
        candidate_masks: list[list[bool]] = []

        for candidate in candidates:
            candidate_ids = self._encode(candidate)
            if not candidate_ids:
                sequences.append([self.prefix_token_id])
                candidate_masks.append([False])
                continue

            # Keep the candidate intact and discard old context from the left first.
            room_for_context = max(0, self.max_positions - 1 - len(candidate_ids))
            truncated_context = context_ids[-room_for_context:] if room_for_context else []

            seq = [self.prefix_token_id] + truncated_context + candidate_ids
            mask = [False] * (1 + len(truncated_context)) + [True] * len(candidate_ids)
            sequences.append(seq)
            candidate_masks.append(mask)

        max_len = max(len(seq) for seq in sequences)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full(
            (len(sequences), max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(sequences), max_len), dtype=torch.long, device=self.device
        )
        score_mask = torch.zeros(
            (len(sequences), max_len), dtype=torch.bool, device=self.device
        )

        for row, (seq, mask) in enumerate(zip(sequences, candidate_masks)):
            length = len(seq)
            input_ids[row, :length] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_mask[row, :length] = 1
            score_mask[row, :length] = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = score_mask[:, 1:] & attention_mask[:, 1:].bool()

            sums = (target_log_probs * valid).sum(dim=1)
            counts = valid.sum(dim=1).clamp_min(1)
            means = sums / counts

        return [(float(s), float(m)) for s, m in zip(sums.cpu(), means.cpu())]

    def score_candidates(self, context: str, candidates: Iterable[str]) -> list[CandidateScore]:
        candidates = list(candidates)
        conditional = self._batch_scores(context, candidates)
        prior = self._batch_scores("", candidates)

        result: list[CandidateScore] = []
        for candidate, (cond_sum, cond_avg), (prior_sum, prior_avg) in zip(
            candidates, conditional, prior
        ):
            result.append(
                CandidateScore(
                    candidate=candidate,
                    conditional_logprob=cond_sum,
                    conditional_avg_logprob=cond_avg,
                    prior_logprob=prior_sum,
                    prior_avg_logprob=prior_avg,
                    context_gain=cond_avg - prior_avg,
                )
            )
        return result

    def rank(self, context: str, candidates: Iterable[str], mode: str = "context_gain") -> list[CandidateScore]:
        scores = self.score_candidates(context, candidates)
        if mode == "conditional":
            key = lambda item: item.conditional_avg_logprob
        elif mode == "context_gain":
            key = lambda item: item.context_gain
        else:
            raise ValueError("mode must be 'conditional' or 'context_gain'")
        return sorted(scores, key=key, reverse=True)
