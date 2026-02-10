from aenum import extend_enum
import numpy as np
import torch
import json
import re

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    SamplingMethod,
    SampleLevelComputation,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import Doc
from lighteval.metrics.imports.bert_scorer import BERTScorer


class NERSpanComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold_entities = [
            (t.lower().strip(), ty.lower().strip())
            for (t, ty) in doc.specific.get("gold_entities", [])
        ]

        if (
            hasattr(model_response, "text_post_processed")
            and model_response.text_post_processed
        ):
            pred_text = model_response.text_post_processed[0]
        elif hasattr(model_response, "text") and model_response.text:
            pred_text = model_response.text[0]
        else:
            pred_text = ""

        pred_entities = self.parse_pred(pred_text)

        pred_by_tag = {}
        for p_text, p_tag in pred_entities:
            pred_by_tag.setdefault(p_tag, []).append(p_text)

        correct = 0
        for g_text, g_tag in gold_entities:
            if g_tag in pred_by_tag:
                for p_text in pred_by_tag[g_tag]:
                    if self.char_overlap_ratio(g_text, p_text) >= 0.5:
                        correct += 1
                        break

        return {
            "correct": correct,
            "gold": len(gold_entities),
            "pred": len(pred_entities),
            "accuracy": correct / len(gold_entities) if gold_entities else 0.0,
        }

    def parse_pred(self, pred: str):
        pred = pred.strip()
        if pred.startswith("```"):
            pred = pred.strip("`").lstrip("json").strip()

        entities = []

        bracket_count = 0
        start_idx = -1
        json_candidates = []

        for i, char in enumerate(pred):
            if char == "[":
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    json_candidates.append(pred[start_idx : i + 1])
                    start_idx = -1

        for json_str in json_candidates:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for e in parsed:
                        if isinstance(e, dict) and "entity" in e and "tag" in e:
                            entity_text = e["entity"].strip()
                            entity_tag = e["tag"].strip()
                            if entity_text and entity_tag:
                                entities.append(
                                    (entity_text.lower(), entity_tag.lower())
                                )
                    if entities:
                        break
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if not entities:
            try:
                parsed = json.loads(pred)
                if isinstance(parsed, list):
                    for e in parsed:
                        if "entity" in e and "tag" in e:
                            entity_text = e["entity"].strip()
                            entity_tag = e["tag"].strip()
                            if entity_text and entity_tag:
                                entities.append(
                                    (entity_text.lower(), entity_tag.lower())
                                )
            except Exception:
                parts = [p for p in pred.replace(";", ",").split(",") if ":" in p]
                for part in parts:
                    try:
                        t, tag = part.split(":", 1)
                        t = t.strip()
                        tag = tag.strip()
                        if t and tag:
                            entities.append((t.lower(), tag.lower()))
                    except ValueError:
                        continue

        return entities

    def char_overlap_ratio(self, a: str, b: str) -> float:
        gold_words = a.split()
        pred_words = b.split()

        matches = 0
        total = sum(len(w) for w in gold_words)

        for g, p in zip(gold_words, pred_words):
            for gc, pc in zip(g, p):
                if gc == pc:
                    matches += 1

        if total == 0:
            return 0.0
        return matches / total


class UDPosComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold_entities = [
            (str(t).lower().strip(), str(ty).lower().strip())
            for (t, ty) in doc.specific.get("gold_entities", [])
            if t is not None and ty is not None
        ]

        if (
            hasattr(model_response, "text_post_processed")
            and model_response.text_post_processed
        ):
            pred_text = model_response.text_post_processed[0]
        elif hasattr(model_response, "text") and model_response.text:
            pred_text = model_response.text[0]
        else:
            pred_text = ""

        pred_entities = []

        m_hy = re.search(r"խոսքի մասն է\s+([^\s.,;]+)", pred_text, flags=re.IGNORECASE)
        m_en = re.search(
            r"part of speech of this word is\s+([^\s.,;]+)",
            pred_text,
            flags=re.IGNORECASE,
        )

        if m_hy:
            tag = m_hy.group(1).lower().strip()
        elif m_en:
            tag = m_en.group(1).lower().strip()
        else:
            tag = None

        if tag:
            if gold_entities:
                word = gold_entities[0][0]
            else:
                word = ""
            pred_entities.append((word.lower(), tag))

        correct = 0
        for g_text, g_tag in gold_entities:
            for _, p_tag in pred_entities:
                if g_tag == p_tag:
                    correct += 1
                    break

        return {
            "correct": correct,
            "gold": len(gold_entities),
            "pred": len(pred_entities),
        }


def pos_agg(items):
    total_correct = sum(i["correct"] for i in items)
    total_gold = sum(i["gold"] for i in items)
    return total_correct / total_gold if total_gold > 0 else 0.0


class BertScoreArm(SampleLevelComputation):
    def __init__(self):
        self.scorer = BERTScorer(
            model_type="Metric-AI/armenian-text-embeddings-1",
            num_layers=9,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        pred = (
            model_response.text_post_processed[0]
            if hasattr(model_response, "text_post_processed")
            and model_response.text_post_processed
            else (
                model_response.text[0]
                if hasattr(model_response, "text") and model_response.text
                else ""
            )
        ).strip()

        golds = doc.choices or []
        if not golds:
            return 0.0

        P, R, F = self.scorer.score([pred], [golds])
        return F[0].item()


ner_span_metric = SampleLevelMetric(
    metric_name="ner_accuracy",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=NERSpanComputation(),
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
pos_metric = SampleLevelMetric(
    metric_name="ud_pos_regex_acc",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=UDPosComputation(),
    corpus_level_fn=pos_agg,
    higher_is_better=True,
)
bert_score_arm = SampleLevelMetric(
    metric_name="bert_score_arm",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=BertScoreArm(),
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


extend_enum(Metrics, "ner_accuracy", ner_span_metric)
extend_enum(Metrics, "ud_pos_regex_acc", pos_metric)
extend_enum(Metrics, "bert_score_arm", bert_score_arm)
