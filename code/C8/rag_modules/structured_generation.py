"""
结构化回答生成模块。
从 Markdown 文档结构直接组装答案，避免不必要的 LLM 调用。
"""

import re
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _extract_markdown_sections(doc: Document) -> Dict[str, List[str]]:
    """将菜谱文档按 Markdown 标题结构切分。"""
    lines = doc.page_content.splitlines()
    sections: Dict[str, List[str]] = {"__intro__": []}
    current_section = "__intro__"

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections.setdefault(current_section, [])
            continue
        if line.startswith("# "):
            continue
        sections.setdefault(current_section, []).append(line)

    return sections


def _clean_section_lines(lines: List[str]) -> List[str]:
    """清理分段内容，去掉空行、图片和模板占位内容。"""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!["):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("TODO") or stripped.startswith("TBD"):
            continue
        cleaned.append(stripped)
    return cleaned


def _format_structured_section_answer(
    doc: Document,
    lines: List[str],
    content_type: str,
) -> Optional[str]:
    """根据文档结构直接组装答案。"""
    cleaned_lines = _clean_section_lines(lines)
    if not cleaned_lines:
        return None

    dish_name = doc.metadata.get("dish_name", "该食谱")
    source_label = f"【食谱：{dish_name}】"

    if content_type == "ingredients":
        body = "\n".join(cleaned_lines)
        return f"## 所需食材\n根据{source_label}整理：\n{body}"

    if content_type == "steps":
        numbered_lines = []
        step_index = 1
        for line in cleaned_lines:
            if re.match(r"^\d+[\.、\s]", line):
                numbered_lines.append(line)
            else:
                normalized = line.lstrip("- ").strip()
                numbered_lines.append(f"{step_index}. {normalized}")
            step_index += 1
        body = "\n".join(numbered_lines)
        return f"## 制作步骤\n参考{source_label}，可以按以下顺序操作：\n{body}"

    if content_type == "tips":
        body = "\n".join(f"- {line.lstrip('- ').strip()}" for line in cleaned_lines)
        return f"## 制作技巧\n以下内容基于{source_label}中的步骤与补充说明整理：\n{body}"

    if content_type == "calculation":
        body = "\n".join(cleaned_lines)
        return f"## 用量计算\n根据{source_label}整理：\n{body}"

    if content_type == "introduction":
        body = "\n".join(cleaned_lines)
        return f"## 菜品介绍\n来自{source_label}的相关介绍：\n{body}"

    return None


def _build_tips_fallback_lines(sections: Dict[str, List[str]]) -> List[str]:
    """在没有独立技巧段时，从步骤与补充内容中提取可复用提示。"""
    fallback_headings = ["附加内容", "操作", "做法", "步骤"]
    candidate_lines: List[str] = []
    for heading in fallback_headings:
        candidate_lines.extend(_clean_section_lines(sections.get(heading, [])))

    tips: List[str] = []
    for line in candidate_lines:
        normalized = re.sub(r"^\d+[\.、\s]*", "", line).strip()
        if not normalized:
            continue
        tips.append(normalized)
        if len(tips) >= 4:
            break
    return tips


def try_build_structured_answer(
    query: str,
    context_docs: List[Document],
    content_type: str = None,
) -> Optional[str]:
    """文档结构明确时，优先直接回答，不交给 LLM 自由生成。

    Args:
        query: 用户查询
        context_docs: 上下文文档列表
        content_type: 内容类型（ingredients/steps/tips/calculation/introduction）

    Returns:
        结构化回答文本，如果无法构建则返回 None
    """
    if not context_docs or not content_type:
        return None

    section_aliases = {
        "ingredients": ["必备原料和工具", "食材", "材料", "原料"],
        "steps": ["操作", "做法", "步骤"],
        "tips": ["附加内容", "小贴士", "技巧"],
        "calculation": ["计算", "用量计算"],
        "introduction": ["__intro__"],
    }
    candidate_headings = section_aliases.get(content_type)
    if not candidate_headings:
        return None

    for doc in context_docs:
        sections = _extract_markdown_sections(doc)
        for heading in candidate_headings:
            if heading in sections:
                structured_answer = _format_structured_section_answer(
                    doc,
                    sections[heading],
                    content_type,
                )
                if structured_answer:
                    logger.info(
                        f"[StructuredAnswer] query='{query}' content_type='{content_type}' dish='{doc.metadata.get('dish_name', '')}'"
                    )
                    return structured_answer

        if content_type == "tips":
            fallback_lines = _build_tips_fallback_lines(sections)
            structured_answer = _format_structured_section_answer(
                doc,
                fallback_lines,
                content_type,
            )
            if structured_answer:
                logger.info(
                    f"[StructuredAnswerFallback] query='{query}' content_type='{content_type}' dish='{doc.metadata.get('dish_name', '')}'"
                )
                return structured_answer

    return None
