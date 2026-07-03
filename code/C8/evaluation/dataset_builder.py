from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


CATEGORY_MAPPING = {
    "meat_dish": "荤菜",
    "vegetable_dish": "素菜",
    "soup": "汤品",
    "dessert": "甜品",
    "breakfast": "早餐",
    "staple": "主食",
    "aquatic": "水产",
    "condiment": "调料",
    "drink": "饮品",
    "semi-finished": "半成品",
}

DIFFICULTY_MARKERS = [
    ("★★★★★", "非常困难"),
    ("★★★★", "困难"),
    ("★★★", "中等"),
    ("★★", "简单"),
    ("★", "非常简单"),
]

INGREDIENT_HEADINGS = {"必备原料和工具", "食材", "材料", "原料"}
STEP_HEADINGS = {"操作", "步骤", "做法"}
TIP_HEADINGS = {"附加内容", "小贴士", "技巧"}

AMBIGUOUS_QUERIES = [
    "今天吃什么",
    "我想吃点清淡的",
    "想做个快手菜",
    "晚上来点下饭的吧",
    "给我推荐一个新手友好的菜",
    "想整点甜的",
    "有没有适合早餐的",
    "来个适合夏天吃的菜",
    "我冰箱里东西不多，做点啥",
    "想吃口味重点的菜",
]

TEMPORAL_UNKNOWN_QUERIES = [
    "我昨天吃了什么",
    "我上周做过哪道菜",
    "你记得我前天晚饭吃了啥吗",
    "我明天中午会吃什么",
    "我上次煮的汤是哪一种",
    "你知道我昨晚喝了什么吗",
]

IRRELEVANT_QUERIES = [
    "不锈钢玻璃怎么清洗",
    "路由器总断网怎么办",
    "手机壳发黄怎么处理",
    "羽绒服怎么洗不结团",
    "电脑风扇噪音大怎么办",
    "书桌划痕怎么修复",
    "怎么给绿植换盆",
    "窗帘发霉怎么处理",
]

BOUNDARY_QUERIES = [
    "麻婆豆腐和红烧鲤鱼是一个菜吗",
    "鸡蛋三明治需要牛排吗",
    "给我红烧鲤鱼的奶茶做法",
    "昨天吃什么，顺便来个鱼香肉丝",
    "红烧鲤鱼的做法和长岛冰茶一起说",
    "推荐个早餐，再告诉我怎么修玻璃",
]


def _extract_sections(content: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"__intro__": []}
    current = "__intro__"
    for line in content.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("# "):
            continue
        sections.setdefault(current, []).append(line.rstrip())
    return sections


def _clean_lines(lines: Iterable[str]) -> List[str]:
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!["):
            continue
        cleaned.append(stripped)
    return cleaned


def _detect_category(path: Path) -> str:
    for part in path.parts:
        if part in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[part]
    return "其他"


def _detect_difficulty(content: str) -> str:
    for marker, label in DIFFICULTY_MARKERS:
        if marker in content:
            return label
    return "未知"


def load_recipe_catalog(data_root: Path) -> List[Dict]:
    catalog = []
    for md_file in sorted(data_root.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        sections = _extract_sections(content)
        catalog.append(
            {
                "dish_name": md_file.stem,
                "source": str(md_file),
                "category": _detect_category(md_file),
                "difficulty": _detect_difficulty(content),
                "sections": {key: _clean_lines(value) for key, value in sections.items()},
            }
        )
    return catalog


def _has_useful_section(item: Dict, headings: Iterable[str]) -> bool:
    return any(item["sections"].get(name) for name in headings)


def _pick_representative_recipes(catalog: List[Dict], seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    by_category: Dict[str, List[Dict]] = {}
    for item in catalog:
        if _has_useful_section(item, INGREDIENT_HEADINGS) and _has_useful_section(item, STEP_HEADINGS):
            by_category.setdefault(item["category"], []).append(item)

    selected = []
    for category, items in sorted(by_category.items()):
        rng.shuffle(items)
        selected.extend(items[:3])

    unique = {}
    for item in selected:
        unique[item["dish_name"]] = item

    if len(unique) < 24:
        extras = [
            item
            for item in catalog
            if item["dish_name"] not in unique
            and _has_useful_section(item, INGREDIENT_HEADINGS)
            and _has_useful_section(item, STEP_HEADINGS)
        ]
        rng.shuffle(extras)
        for item in extras:
            unique[item["dish_name"]] = item
            if len(unique) >= 24:
                break

    return list(unique.values())[:24]


def _flatten_section_terms(item: Dict, headings: Iterable[str], limit: int = 3) -> List[str]:
    terms: List[str] = []
    for heading in headings:
        for line in item["sections"].get(heading, []):
            cleaned = re.sub(r"^[\-\d\.\s、]+", "", line).strip()
            if cleaned:
                terms.append(cleaned[:20])
            if len(terms) >= limit:
                return terms
    return terms


def _make_single_turn_cases(selected: List[Dict]) -> List[Dict]:
    cases: List[Dict] = []
    for item in selected[:18]:
        ingredient_terms = _flatten_section_terms(item, INGREDIENT_HEADINGS)
        step_terms = _flatten_section_terms(item, STEP_HEADINGS)
        tip_terms = _flatten_section_terms(item, TIP_HEADINGS, limit=2)

        cases.append(
            {
                "case_id": f"single-ingredients-{item['dish_name']}",
                "scenario_type": "single_turn_detail",
                "evaluation_mode": "strict",
                "source_dishes": [item["dish_name"]],
                "turns": [
                    {
                        "user": f"{item['dish_name']}需要什么食材",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "required_terms": ingredient_terms[:3],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
        cases.append(
            {
                "case_id": f"single-steps-{item['dish_name']}",
                "scenario_type": "single_turn_detail",
                "evaluation_mode": "strict",
                "source_dishes": [item["dish_name"]],
                "turns": [
                    {
                        "user": f"{item['dish_name']}怎么做",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "required_terms_any": step_terms[:3],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
        cases.append(
            {
                "case_id": f"single-tips-{item['dish_name']}",
                "scenario_type": "single_turn_detail",
                "evaluation_mode": "strict",
                "source_dishes": [item["dish_name"]],
                "turns": [
                    {
                        "user": f"{item['dish_name']}有什么制作技巧",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "required_terms_any": tip_terms or step_terms[:2],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
    return cases


def _make_category_cases(catalog: List[Dict]) -> List[Dict]:
    cases = []
    counts = Counter(item["category"] for item in catalog)
    for category in sorted(category for category, count in counts.items() if count >= 5)[:8]:
        cases.append(
            {
                "case_id": f"category-{category}",
                "scenario_type": "category_recommendation",
                "evaluation_mode": "strict",
                "source_dishes": [],
                "turns": [
                    {
                        "user": f"推荐几个{category}",
                        "expectation": {
                            "required_terms_any": [category, "1.", "2."],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
    return cases


def _make_difficulty_cases(catalog: List[Dict]) -> List[Dict]:
    cases = []
    difficulties = Counter(item["difficulty"] for item in catalog)
    for difficulty in sorted(diff for diff, count in difficulties.items() if count >= 5 and diff != "未知")[:4]:
        cases.append(
            {
                "case_id": f"difficulty-{difficulty}",
                "scenario_type": "difficulty_recommendation",
                "evaluation_mode": "strict",
                "source_dishes": [],
                "turns": [
                    {
                        "user": f"推荐几道{difficulty}的菜",
                        "expectation": {
                            "required_terms_any": ["1.", "2.", "推荐"],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
    return cases


def _make_multi_turn_cases(selected: List[Dict]) -> List[Dict]:
    cases = []
    for item in selected[:12]:
        ingredient_terms = _flatten_section_terms(item, INGREDIENT_HEADINGS)
        step_terms = _flatten_section_terms(item, STEP_HEADINGS)
        cases.append(
            {
                "case_id": f"multi-explicit-{item['dish_name']}",
                "scenario_type": "multi_turn_followup",
                "evaluation_mode": "strict",
                "source_dishes": [item["dish_name"]],
                "turns": [
                    {
                        "user": f"我们聊聊{item['dish_name']}",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "response_policy": "grounded_answer",
                        },
                    },
                    {
                        "user": "它需要什么食材",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "required_terms_any": ingredient_terms[:3],
                            "response_policy": "grounded_answer",
                        },
                    },
                    {
                        "user": "再说一下怎么做",
                        "expectation": {
                            "expected_dish": item["dish_name"],
                            "required_terms_any": step_terms[:3],
                            "response_policy": "grounded_answer",
                        },
                    },
                ],
            }
        )
    return cases


def _make_ambiguous_cases() -> List[Dict]:
    cases = []
    for index, query in enumerate(AMBIGUOUS_QUERIES, start=1):
        cases.append(
            {
                "case_id": f"ambiguous-{index}",
                "scenario_type": "ambiguous_related",
                "evaluation_mode": "fallback",
                "source_dishes": [],
                "turns": [
                    {
                        "user": query,
                        "expectation": {
                            "required_terms_any": ["推荐", "可以", "菜", "吃"],
                            "response_policy": "grounded_answer",
                        },
                    }
                ],
            }
        )
    return cases


def _make_temporal_unknown_cases() -> List[Dict]:
    cases = []
    for index, query in enumerate(TEMPORAL_UNKNOWN_QUERIES, start=1):
        cases.append(
            {
                "case_id": f"temporal-{index}",
                "scenario_type": "temporal_unknown",
                "evaluation_mode": "fallback",
                "source_dishes": [],
                "turns": [
                    {
                        "user": query,
                        "expectation": {
                            "required_terms_any": ["不知道", "不清楚", "可以", "推荐"],
                            "forbidden_terms": ["鸡蛋三明治", "红烧鲤鱼", "长岛冰茶"],
                            "response_policy": "polite_fallback",
                        },
                    }
                ],
            }
        )
    return cases


def _make_irrelevant_cases() -> List[Dict]:
    cases = []
    for index, query in enumerate(IRRELEVANT_QUERIES, start=1):
        cases.append(
            {
                "case_id": f"irrelevant-{index}",
                "scenario_type": "irrelevant_question",
                "evaluation_mode": "fallback",
                "source_dishes": [],
                "turns": [
                    {
                        "user": query,
                        "expectation": {
                            "required_terms_any": ["不是", "不清楚", "可以", "如果你愿意"],
                            "response_policy": "polite_fallback",
                        },
                    }
                ],
            }
        )
    return cases


def _make_boundary_cases() -> List[Dict]:
    cases = []
    for index, query in enumerate(BOUNDARY_QUERIES, start=1):
        cases.append(
            {
                "case_id": f"boundary-{index}",
                "scenario_type": "boundary_mixed_intent",
                "evaluation_mode": "boundary",
                "source_dishes": [],
                "turns": [
                    {
                        "user": query,
                        "expectation": {
                            "required_terms_any": ["不能", "需要确认", "可以", "推荐", "不知道"],
                            "response_policy": "polite_fallback",
                        },
                    }
                ],
            }
        )
    return cases


def build_testset(catalog: List[Dict]) -> Dict:
    selected = _pick_representative_recipes(catalog)
    cases = []
    cases.extend(_make_single_turn_cases(selected))
    cases.extend(_make_category_cases(catalog))
    cases.extend(_make_difficulty_cases(catalog))
    cases.extend(_make_multi_turn_cases(selected))
    cases.extend(_make_ambiguous_cases())
    cases.extend(_make_temporal_unknown_cases())
    cases.extend(_make_irrelevant_cases())
    cases.extend(_make_boundary_cases())

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases[:120],
    }


def summarize_testset_coverage(testset: Dict) -> Dict:
    cases = testset["cases"]
    multi_turn_cases = [case for case in cases if len(case["turns"]) > 1]
    categories = set()
    for case in cases:
        for dish in case.get("source_dishes", []):
            categories.add(dish)

    return {
        "single_turn_cases": len(cases) - len(multi_turn_cases),
        "multi_turn_cases": len(multi_turn_cases),
        "irrelevant_cases": sum(case["scenario_type"] == "irrelevant_question" for case in cases),
        "temporal_unknown_cases": sum(case["scenario_type"] == "temporal_unknown" for case in cases),
        "ambiguous_cases": sum(case["scenario_type"] == "ambiguous_related" for case in cases),
        "categories_covered": len({case["scenario_type"] for case in cases if case["scenario_type"] == "category_recommendation"})
        + len({dish for dish in categories}),
    }


def save_testset(output_path: Path, testset: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
