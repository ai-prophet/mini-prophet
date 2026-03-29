"""ReadSourceTool: read full content of a source by ID."""

from __future__ import annotations

from miniprophet.environment.source_registry import SourceRegistry

READ_SOURCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_source",
        "description": (
            "Read the full content of a previously discovered source by its ID. "
            "Use this to get detailed information from a source."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The source ID (e.g. 'S3') from search results.",
                },
            },
            "required": ["source_id"],
        },
    },
}


class ReadSourceTool:
    """Read the full content of a source by its ID."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "read_source"

    def get_schema(self) -> dict:
        return READ_SOURCE_SCHEMA

    async def execute(self, args: dict) -> dict:
        source_id = args.get("source_id", "")
        if isinstance(source_id, int):
            source_id = f"S{source_id}"
        source_id = str(source_id).strip().upper()

        if not source_id:
            return {"output": "Error: 'source_id' is required (e.g. 'S3').", "error": True}

        try:
            source = await self._registry.get_full_content(source_id)
        except KeyError:
            return {"output": f"Error: unknown source_id '{source_id}'.", "error": True}

        date_line = f"Date: {source.date or 'No date info'}\n" if source.date else ""
        return {
            "output": (
                f'<source id="{source_id}" title="{source.title}" url="{source.url}">\n'
                f"{date_line}"
                f"{source.snippet}\n"
                f"</source>"
            ),
        }

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
