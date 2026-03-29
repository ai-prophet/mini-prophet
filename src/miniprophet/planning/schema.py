"""XML plan schema: parsing, validation, and dataclasses."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class SubQuery:
    id: str
    query: str
    trigger_condition: str | None = None


@dataclass
class SubProblem:
    id: str
    title: str
    context: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    trigger_condition: str | None = None


@dataclass
class ForecastPlan:
    main_problem_id: str
    main_problem_title: str
    main_problem_context: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    sub_problems: list[SubProblem] = field(default_factory=list)
    factors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _text(element: ET.Element | None, default: str = "") -> str:
    """Get stripped text content of an element."""
    if element is None:
        return default
    return (element.text or "").strip()


def _parse_sub_queries(parent: ET.Element, errors: list[str], context: str) -> list[SubQuery]:
    """Parse <sub_queries> children from *parent*."""
    sq_container = parent.find("sub_queries")
    if sq_container is None:
        errors.append(f"Missing <sub_queries> inside <{context}>")
        return []

    queries: list[SubQuery] = []
    for sq in sq_container.findall("sub_query"):
        sq_id = _text(sq.find("id"))
        query = _text(sq.find("query"))
        trigger = _text(sq.find("trigger_condition")) or None

        if not sq_id:
            errors.append(f"Empty or missing <id> in a <sub_query> inside <{context}>")
        if not query:
            errors.append(f"Empty or missing <query> in sub_query '{sq_id}' inside <{context}>")

        queries.append(SubQuery(id=sq_id, query=query, trigger_condition=trigger))
    return queries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_and_validate(xml_string: str) -> tuple[ForecastPlan | None, list[str]]:
    """Parse an XML plan string and validate its structure.

    Returns ``(plan, errors)``.  If *errors* is non-empty the plan may be
    ``None`` or partially populated.  Error strings are human-readable and
    suitable for returning to the LLM so it can fix the issues.
    """
    errors: list[str] = []

    # --- XML well-formedness ---
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        return None, [f"XML parse error: {exc}"]

    if root.tag != "plan":
        errors.append(f"Root element must be <plan>, got <{root.tag}>")
        return None, errors

    # --- main_problem ---
    mp = root.find("main_problem")
    if mp is None:
        errors.append("Missing required element <main_problem>")
        return None, errors

    mp_id = _text(mp.find("id"))
    mp_title = _text(mp.find("title"))
    mp_context = _text(mp.find("context"))

    if not mp_id:
        errors.append("Missing or empty <id> inside <main_problem>")
    if not mp_title:
        errors.append("Missing or empty <title> inside <main_problem>")
    if not mp_context:
        errors.append("Missing or empty <context> inside <main_problem>")

    main_queries = _parse_sub_queries(mp, errors, "main_problem")

    # --- sub_problems ---
    sub_problems: list[SubProblem] = []
    sp_container = root.find("sub_problems")
    if sp_container is not None:
        for sp in sp_container.findall("sub_problem"):
            sp_id = _text(sp.find("id"))
            sp_title = _text(sp.find("title"))
            sp_context = _text(sp.find("context"))
            sp_trigger = _text(sp.find("trigger_condition")) or None

            if not sp_id:
                errors.append("Empty or missing <id> in a <sub_problem>")
            if not sp_title:
                errors.append(f"Empty or missing <title> in sub_problem '{sp_id}'")
            if not sp_context:
                errors.append(f"Empty or missing <context> in sub_problem '{sp_id}'")

            sp_queries = _parse_sub_queries(sp, errors, f"sub_problem '{sp_id}'")
            if not sp_queries:
                errors.append(f"sub_problem '{sp_id}' must have at least one <sub_query>")

            sub_problems.append(
                SubProblem(
                    id=sp_id,
                    title=sp_title,
                    context=sp_context,
                    sub_queries=sp_queries,
                    trigger_condition=sp_trigger,
                )
            )

    # --- factors_to_consider ---
    factors: list[str] = []
    ftc = root.find("factors_to_consider")
    if ftc is not None:
        for f in ftc.findall("factor"):
            text = (f.text or "").strip()
            if not text:
                errors.append("Empty <factor> inside <factors_to_consider>")
            else:
                factors.append(text)

    # --- ID uniqueness ---
    all_ids: list[str] = []
    if mp_id:
        all_ids.append(mp_id)
    for sq in main_queries:
        if sq.id:
            all_ids.append(sq.id)
    for sp in sub_problems:
        if sp.id:
            all_ids.append(sp.id)
        for sq in sp.sub_queries:
            if sq.id:
                all_ids.append(sq.id)

    seen: set[str] = set()
    for id_val in all_ids:
        if " " in id_val or "\t" in id_val or "\n" in id_val:
            errors.append(f"ID '{id_val}' must not contain whitespace")
        if id_val in seen:
            errors.append(f"Duplicate ID '{id_val}'")
        seen.add(id_val)

    plan = ForecastPlan(
        main_problem_id=mp_id,
        main_problem_title=mp_title,
        main_problem_context=mp_context,
        sub_queries=main_queries,
        sub_problems=sub_problems,
        factors=factors,
    )

    return plan, errors
