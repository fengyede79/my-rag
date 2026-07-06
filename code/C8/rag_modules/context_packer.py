"""Context packing and answer-mode finalization for recipe generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.documents import Document


SECTION_TYPE_ALIASES = {
    "ingredients": ("必备原料和工具", "原料", "食材", "材料", "配料"),
    "steps": ("操作", "步骤", "做法", "制作"),
    "tips": ("附加内容", "注意", "小贴士", "技巧", "提示"),
    "calculation": ("计算", "用量", "份量", "分量"),
}


def _section_type_for_heading(title: str) -> str:
    for section_type, aliases in SECTION_TYPE_ALIASES.items():
        if any(alias in title for alias in aliases):
            return section_type
    return "introduction"


def extract_markdown_sections(parent_doc: Document) -> List[Dict[str, Any]]:
    """Extract coarse recipe sections from a Markdown parent document."""
    text = parent_doc.page_content or ""
    dish_name = (parent_doc.metadata or {}).get("dish_name")
    parent_id = (parent_doc.metadata or {}).get("parent_id")
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        title = current_title or (dish_name or "简介")
        sections.append(
            {
                "title": title,
                "section_type": _section_type_for_heading(title) if current_level >= 2 else "introduction",
                "text": section_text,
                "level": current_level,
                "dish_name": dish_name,
                "parent_id": parent_id,
                "source_metadata": dict(parent_doc.metadata or {}),
            }
        )

    for line in text.splitlines():
        match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections


def finalize_answer_mode(
    *,
    turn_info: dict,
    execution_plan: dict,
    query_plan: dict,
    retrieval_quality: dict | None,
) -> str:
    """Finalize answer mode without letting document shape change the user task."""
    action = turn_info.get("action") or execution_plan.get("action")
    hint = turn_info.get("answer_mode_hint") or execution_plan.get("answer_mode")
    route_type = query_plan.get("route_type")

    if action == "smalltalk" or hint == "safe_direct":
        return "safe_direct"
    if action == "history_answer" or hint == "history_based":
        return "history_based"
    if action == "substitution" or hint == "substitution":
        return "substitution"
    if action == "compare" or hint == "comparison":
        return "comparison"
    if hint == "troubleshooting":
        return "troubleshooting"
    if action == "retrieve_list" or route_type == "list":
        return "recommendation"
    return "recipe_detail"


SECTION_PREFERENCES = {
    "ingredients": (["ingredients"], ["calculation", "tips"]),
    "steps": (["steps"], ["ingredients", "tips"]),
    "tips": (["tips"], ["steps"]),
    "introduction": (["introduction"], ["tips"]),
    "calculation": (["calculation"], ["ingredients"]),
    "substitution": (["ingredients", "tips"], ["steps"]),
    "troubleshooting": (["tips", "steps"], ["ingredients"]),
}


class ContextPacker:
    """Build generation-facing packed context from retrieved chunks and parent docs."""

    def __init__(
        self,
        *,
        max_chars_total: int = 2400,
        max_chars_per_doc: int = 1200,
        max_docs: int = 5,
    ):
        self.max_chars_total = max_chars_total
        self.max_chars_per_doc = max_chars_per_doc
        self.max_docs = max_docs

    def build_context_pack(
        self,
        *,
        query: str,
        retrieval_result: dict,
        query_plan: dict,
        execution_plan: dict,
        turn_info: dict,
        parent_docs: list[Document],
    ) -> dict:
        answer_mode = finalize_answer_mode(
            turn_info=turn_info,
            execution_plan=execution_plan,
            query_plan=query_plan,
            retrieval_quality=(retrieval_result or {}).get("quality"),
        )
        content_type = self._content_type(query_plan, answer_mode)

        if answer_mode == "recommendation":
            context_docs, selected_sections, trimmed = self._build_recommendation_context(parent_docs)
        else:
            context_docs, selected_sections, trimmed = self._build_section_context(
                parent_docs=parent_docs,
                content_type=content_type,
                answer_mode=answer_mode,
            )

        fallback_to_chunks = False
        if not context_docs:
            context_docs = self._chunk_fallback((retrieval_result or {}).get("chunks") or [])
            selected_sections = []
            fallback_to_chunks = bool(context_docs)
            trimmed = False

        answer_mode_initial = turn_info.get("answer_mode_hint") or execution_plan.get("answer_mode")
        return {
            "answer_mode": answer_mode,
            "context_docs": context_docs,
            "parent_docs": list(parent_docs or []),
            "selected_sections": selected_sections,
            "content_type": content_type,
            "trace": {
                "input_chunk_count": len((retrieval_result or {}).get("chunks") or []),
                "parent_doc_count": len(parent_docs or []),
                "selected_section_count": len(selected_sections),
                "answer_mode_initial": answer_mode_initial,
                "answer_mode_final": answer_mode,
                "answer_mode_source": "turn_info" if turn_info.get("answer_mode_hint") else "execution_plan",
                "content_type": content_type,
                "trimmed": trimmed,
                "fallback_to_chunks": fallback_to_chunks,
            },
        }

    def _content_type(self, query_plan: dict, answer_mode: str) -> str:
        filters = query_plan.get("filters") or {}
        if answer_mode in {"substitution", "troubleshooting"}:
            return answer_mode
        return filters.get("content_type") or ("steps" if answer_mode == "recipe_detail" else answer_mode)

    def _build_section_context(
        self,
        *,
        parent_docs: list[Document],
        content_type: str,
        answer_mode: str,
    ) -> tuple[list[Document], list[dict], bool]:
        preferred, secondary = SECTION_PREFERENCES.get(
            content_type,
            SECTION_PREFERENCES.get(answer_mode, (["steps"], ["ingredients", "tips"])),
        )
        sections = []
        for parent_doc in parent_docs or []:
            sections.extend(extract_markdown_sections(parent_doc))

        chosen = [section for section in sections if section["section_type"] in preferred]
        if not chosen:
            chosen = [section for section in sections if section["section_type"] in secondary]
        elif answer_mode in {"substitution", "troubleshooting"}:
            chosen.extend(section for section in sections if section["section_type"] in secondary)

        return self._sections_to_docs(chosen)

    def _sections_to_docs(self, sections: list[dict]) -> tuple[list[Document], list[dict], bool]:
        context_docs: list[Document] = []
        selected: list[dict] = []
        total = 0
        trimmed_any = False
        for section in sections:
            if len(context_docs) >= self.max_docs or total >= self.max_chars_total:
                break
            text, trimmed = self._trim_text(section["text"], self.max_chars_per_doc)
            remaining = self.max_chars_total - total
            if len(text) > remaining:
                text, trimmed = self._trim_text(text, remaining)
            if not text:
                continue
            trimmed_any = trimmed_any or trimmed
            total += len(text)
            context_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **(section.get("source_metadata") or {}),
                        "dish_name": section.get("dish_name"),
                        "parent_id": section.get("parent_id"),
                        "section_title": section.get("title"),
                        "section_type": section.get("section_type"),
                        "context_pack_mode": "section",
                    },
                )
            )
            selected.append(
                {
                    "dish_name": section.get("dish_name"),
                    "section_title": section.get("title"),
                    "section_type": section.get("section_type"),
                    "source_parent_id": section.get("parent_id"),
                    "token_estimate": max(1, len(text) // 4),
                }
            )
        return context_docs, selected, trimmed_any

    def _build_recommendation_context(self, parent_docs: list[Document]) -> tuple[list[Document], list[dict], bool]:
        docs: list[Document] = []
        selected: list[dict] = []
        total = 0
        trimmed_any = False
        for parent_doc in (parent_docs or [])[: self.max_docs]:
            dish_name = (parent_doc.metadata or {}).get("dish_name")
            summary = self._summary_text(parent_doc)
            summary, trimmed = self._trim_text(summary, min(self.max_chars_per_doc, self.max_chars_total - total))
            if not summary:
                continue
            total += len(summary)
            trimmed_any = trimmed_any or trimmed
            docs.append(
                Document(
                    page_content=summary,
                    metadata={**(parent_doc.metadata or {}), "context_pack_mode": "summary"},
                )
            )
            selected.append(
                {
                    "dish_name": dish_name,
                    "section_title": "summary",
                    "section_type": "summary",
                    "source_parent_id": (parent_doc.metadata or {}).get("parent_id"),
                    "token_estimate": max(1, len(summary) // 4),
                }
            )
            if total >= self.max_chars_total:
                break
        return docs, selected, trimmed_any

    def _summary_text(self, parent_doc: Document) -> str:
        dish_name = (parent_doc.metadata or {}).get("dish_name", "未知菜品")
        sections = extract_markdown_sections(parent_doc)
        intro = next((section["text"] for section in sections if section["section_type"] == "introduction"), "")
        ingredients = next((section["text"] for section in sections if section["section_type"] == "ingredients"), "")
        return f"# {dish_name}\n{intro}\n{ingredients}".strip()

    def _chunk_fallback(self, chunks: list[Document]) -> list[Document]:
        docs = []
        for chunk in chunks[: self.max_docs]:
            text, _ = self._trim_text(chunk.page_content or "", self.max_chars_per_doc)
            docs.append(Document(page_content=text, metadata={**(chunk.metadata or {}), "context_pack_mode": "chunk_fallback"}))
        return docs

    def _trim_text(self, text: str, limit: int) -> tuple[str, bool]:
        if limit <= 0:
            return "", True
        if len(text) <= limit:
            return text, False
        cut = text[:limit].rstrip()
        last_newline = cut.rfind("\n")
        if last_newline > limit * 0.6:
            cut = cut[:last_newline].rstrip()
        return cut, True
