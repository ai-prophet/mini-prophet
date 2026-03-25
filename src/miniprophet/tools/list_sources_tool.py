"""ListSourcesTool: list all discovered source metadata."""

from __future__ import annotations

from miniprophet.environment.source_registry import SourceRegistry

LIST_SOURCES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_sources",
        "description": (
            "List all discovered sources with their metadata (ID, title, date, gist). "
            "Optionally filter by problem_id to see sources from a specific subproblem context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "problem_id": {
                    "type": "string",
                    "description": (
                        "Optional. Filter sources by problem context "
                        "(e.g. 'main' or a subproblem ID)."
                    ),
                },
            },
        },
    },
}


class ListSourcesTool:
    """List all discovered sources with metadata and gists."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "list_sources"

    def get_schema(self) -> dict:
        return LIST_SOURCES_SCHEMA

    async def execute(self, args: dict) -> dict:
        problem_id = args.get("problem_id") or None
        sources = await self._registry.list_sources(problem_id=problem_id)

        if not sources:
            suffix = f" (problem_id={problem_id})" if problem_id else ""
            return {"output": f"No sources found.{suffix}"}

        lines: list[str] = [f'<sources count="{len(sources)}">']
        for s in sources:
            lines.append(
                f'<source id="{s["source_id"]}" title="{s["title"]}" '
                f'date="{s["date"] or "N/A"}" problem_id="{s["problem_id"]}">\n'
                f"{s['gist']}\n"
                f"</source>"
            )
        lines.append("</sources>")
        return {"output": "\n".join(lines)}

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
