"""Source board tools: add_source and edit_note."""

from __future__ import annotations

from miniprophet.environment.source_board import VALID_SENTIMENTS, Source, SourceBoard

ADD_SOURCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_source",
        "description": (
            "Add a source from search results to your source board. "
            "Include an analytical note and optionally a reaction per outcome."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The global source ID (e.g. 'S3') from search results.",
                },
                "note": {
                    "type": "string",
                    "description": "Your analytical note about this source.",
                },
                "reaction": {
                    "type": "object",
                    "description": (
                        "Optional. Map outcome names to sentiment: "
                        "'very_positive', 'positive', 'neutral', 'negative', 'very_negative'. "
                        "Only include outcomes this source is relevant to."
                    ),
                },
            },
            "required": ["source_id", "note"],
        },
    },
}

EDIT_NOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_note",
        "description": (
            "Edit the note of a previously added source on the board. "
            "Use this to update your analysis as new information becomes available "
            "(e.g. mark a source as unreliable based on contradicting evidence)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "integer",
                    "description": "The board entry ID (shown as #N on the board).",
                },
                "new_note": {
                    "type": "string",
                    "description": "The updated analytical note.",
                },
                "reaction": {
                    "type": "object",
                    "description": (
                        "Optional. Updated reaction map (outcome -> sentiment). "
                        "Replaces the previous reaction entirely if provided."
                    ),
                },
            },
            "required": ["board_id", "new_note"],
        },
    },
}


def _validate_reaction(
    reaction: dict | None, outcomes: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Validate a reaction dict. Returns (cleaned, errors)."""
    if not reaction:
        return {}, []
    errors: list[str] = []
    cleaned: dict[str, str] = {}
    for key, val in reaction.items():
        if key not in outcomes:
            errors.append(f"Unknown outcome in reaction: '{key}'.")
        elif val not in VALID_SENTIMENTS:
            errors.append(
                f"Invalid sentiment for '{key}': '{val}'. "
                f"Must be one of: {', '.join(sorted(VALID_SENTIMENTS))}."
            )
        else:
            cleaned[key] = val
    return cleaned, errors


def _display_board_output(output: dict, board: SourceBoard) -> None:
    """Shared display logic for both add_source and edit_note."""
    from miniprophet.cli.components.observation import print_observation
    from miniprophet.cli.components.source_board import print_board_state

    if output.get("error"):
        print_observation(output)
    else:
        print_observation(output)
        print_board_state(board)


class AddSourceTool:
    """Add a source from search results to the source board."""

    def __init__(
        self,
        source_registry: dict[str, Source],
        board: SourceBoard,
        outcomes: list[str],
    ) -> None:
        self._source_registry = source_registry
        self._board = board
        self._outcomes = outcomes

    @property
    def name(self) -> str:
        return "add_source"

    def get_schema(self) -> dict:
        return ADD_SOURCE_SCHEMA

    async def execute(self, args: dict) -> dict:
        return self._execute_impl(args)

    def _execute_impl(self, args: dict) -> dict:
        source_id = args.get("source_id", "")
        if isinstance(source_id, int):
            source_id = f"S{source_id}"
        source_id = str(source_id).strip().upper()
        note = args.get("note", "").strip()
        raw_reaction = args.get("reaction")

        if not source_id:
            return {"output": "Error: 'source_id' is required (e.g. 'S3').", "error": True}
        if not note:
            return {"output": "Error: 'note' is required.", "error": True}
        if source_id not in self._source_registry:
            valid_ids = ", ".join(sorted(self._source_registry.keys(), key=lambda s: int(s[1:])))
            return {
                "output": f"Error: unknown source_id '{source_id}'. Valid IDs: {valid_ids or '(none)'}.",
                "error": True,
            }

        reaction, reaction_errors = _validate_reaction(raw_reaction, self._outcomes)
        if reaction_errors:
            return {
                "output": "Reaction validation errors:\n" + "\n".join(reaction_errors),
                "error": True,
            }

        source = self._source_registry[source_id]
        entry = self._board.add(source, note, reaction=reaction, source_id=source_id)
        return {"output": f"Source {source_id} added to board as #{entry.id}."}

    def display(self, output: dict) -> None:
        _display_board_output(output, self._board)


class EditNoteTool:
    """Edit the note/reaction of a previously added source on the board."""

    def __init__(
        self,
        board: SourceBoard,
        outcomes: list[str],
    ) -> None:
        self._board = board
        self._outcomes = outcomes

    @property
    def name(self) -> str:
        return "edit_note"

    def get_schema(self) -> dict:
        return EDIT_NOTE_SCHEMA

    async def execute(self, args: dict) -> dict:
        return self._execute_impl(args)

    def _execute_impl(self, args: dict) -> dict:
        board_id = args.get("board_id")
        new_note = args.get("new_note", "").strip()
        raw_reaction = args.get("reaction")

        if board_id is None:
            return {"output": "Error: 'board_id' is required.", "error": True}
        if not new_note:
            return {"output": "Error: 'new_note' is required.", "error": True}

        reaction, reaction_errors = _validate_reaction(raw_reaction, self._outcomes)
        if reaction_errors:
            return {
                "output": "Reaction validation errors:\n" + "\n".join(reaction_errors),
                "error": True,
            }

        try:
            entry = self._board.edit_note(
                board_id, new_note, reaction=reaction if raw_reaction is not None else None
            )
        except KeyError:
            return {"output": f"Error: no board entry with id #{board_id}.", "error": True}

        return {"output": f"Note for #{entry.id} updated."}

    def display(self, output: dict) -> None:
        _display_board_output(output, self._board)
