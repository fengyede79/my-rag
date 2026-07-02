from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_builder import build_testset, load_recipe_catalog, save_testset
from evaluation.scoring import score_rule_metrics
from main import RecipeRAGSystem


def _project_root() -> Path:
    return PROJECT_ROOT


def _default_data_root() -> Path:
    return _project_root().parents[1] / "data" / "C8"


def _extract_numbered_dishes(answer: str) -> List[str]:
    matches = re.findall(r"^\s*\d+\.\s*(.+?)\s*$", answer, flags=re.MULTILINE)
    return [match.strip() for match in matches]


def _resolve_dynamic_expectation(expectation: Dict, previous_answer: Optional[str]) -> Dict:
    resolved = dict(expectation)
    reference_index = resolved.pop("expected_dish_from_previous_index", None)
    if reference_index and previous_answer:
        dishes = _extract_numbered_dishes(previous_answer)
        if 1 <= reference_index <= len(dishes):
            resolved["expected_dish"] = dishes[reference_index - 1]
    return resolved


def _judge_with_llm(rag_system: RecipeRAGSystem, query: str, answer: str, expectation: Dict) -> Dict:
    prompt = f"""
你是食谱问答评测裁判。请只输出 JSON。

用户问题: {query}
系统回答: {answer}
期望: {json.dumps(expectation, ensure_ascii=False)}

请从以下维度各打 0 到 5 分：
- accuracy
- helpfulness
- clarity
- fallback_quality
- hallucination_control

额外给出:
- overall_score: 0 到 5
- verdict: pass 或 fail
- reason: 一句话中文说明
"""
    response = rag_system.generation_module.llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    # 尝试从 markdown 代码块中提取 JSON
    json_match = re.search(r'\{[\s\S]*\}', content)
    raw = json_match.group() if json_match else content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "overall_score": 0,
            "verdict": "fail",
            "reason": f"LLM judge JSON 解析失败: {content[:200]}",
        }


def _safe_average(values: List[Optional[float]]) -> Optional[float]:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 4)


def summarize_diagnostic_trends(results: List[Dict]) -> Dict:
    retrieval_purities: List[Optional[float]] = []
    purity_gains: List[Optional[float]] = []
    mismatch_ratios: List[Optional[float]] = []
    filter_overkill_count = 0
    failures_by_layer: Dict[str, int] = {}
    scenario_failure_counts: Dict[str, int] = {}

    for case_result in results:
        scenario_type = case_result.get("scenario_type", "unknown")
        case_failed = False
        for turn in case_result.get("turns", []):
            diagnostics = turn.get("diagnostics", {})
            retrieval_metrics = diagnostics.get("retrieval", {}).get("metrics", {})
            retrieval_purities.append(retrieval_metrics.get("final_target_purity"))
            purity_gains.append(retrieval_metrics.get("purity_gain"))
            mismatch_ratios.append(retrieval_metrics.get("content_type_mismatch_ratio"))
            if retrieval_metrics.get("filter_overkill_risk"):
                filter_overkill_count += 1

            layer = diagnostics.get("summary", {}).get("primary_failure_layer", "none")
            failures_by_layer[layer] = failures_by_layer.get(layer, 0) + 1
            if layer != "none":
                case_failed = True

        if case_failed:
            scenario_failure_counts[scenario_type] = scenario_failure_counts.get(scenario_type, 0) + 1

    return {
        "retrieval": {
            "avg_final_target_purity": _safe_average(retrieval_purities),
            "avg_purity_gain": _safe_average(purity_gains),
            "avg_content_type_mismatch_ratio": _safe_average(mismatch_ratios),
            "filter_overkill_count": filter_overkill_count,
        },
        "failures_by_layer": failures_by_layer,
        "scenario_failure_counts": scenario_failure_counts,
    }


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    use_llm_judge: bool = False,
    limit: Optional[int] = None,
) -> Dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"][:limit] if limit else dataset["cases"]

    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    results = []
    for case in cases:
        session_id = case["case_id"]
        previous_answer = None
        turn_results = []
        for turn in case["turns"]:
            expectation = _resolve_dynamic_expectation(turn["expectation"], previous_answer)
            response = rag_system.ask_question(
                turn["user"],
                stream=False,
                session_id=session_id,
                return_diagnostics=True,
                expectation=expectation,
            )
            answer = response["answer"]
            rule_result = score_rule_metrics(answer, expectation)
            turn_result = {
                "user": turn["user"],
                "answer": answer,
                "rule_result": rule_result,
                "diagnostics": response["diagnostics"],
            }
            if use_llm_judge:
                turn_result["llm_judge"] = _judge_with_llm(rag_system, turn["user"], answer, expectation)
            turn_results.append(turn_result)
            previous_answer = answer
        results.append({"case_id": case["case_id"], "scenario_type": case["scenario_type"], "turns": turn_results})

    flat_rule_scores = [
        turn["rule_result"]["score"]
        for case_result in results
        for turn in case_result["turns"]
    ]
    summary = {
        "cases": len(results),
        "turns": sum(len(case_result["turns"]) for case_result in results),
        "avg_rule_score": round(sum(flat_rule_scores) / len(flat_rule_scores), 4) if flat_rule_scores else 0.0,
        "pass_rate": round(
            sum(turn["rule_result"]["passed"] for case_result in results for turn in case_result["turns"])
            / max(1, sum(len(case_result["turns"]) for case_result in results)),
            4,
        ),
    }

    payload = {
        "summary": summary,
        "diagnostic_trends": summarize_diagnostic_trends(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument("--dataset", type=Path, default=_project_root() / "evaluation" / "testset.json")
    parser.add_argument("--output", type=Path, default=_project_root() / "evaluation" / "latest_report.json")
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.build_dataset:
        catalog = load_recipe_catalog(_default_data_root())
        testset = build_testset(catalog)
        save_testset(args.dataset, testset)

    report = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        use_llm_judge=args.use_llm_judge,
        limit=args.limit,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
