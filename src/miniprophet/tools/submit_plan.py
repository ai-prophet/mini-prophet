"""SubmitPlan tool: validate and submit an XML research plan."""

from __future__ import annotations

from miniprophet.exceptions import PlanSubmitted
from miniprophet.planning.schema import parse_and_validate

SUBMIT_PLAN_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "Submit your research plan as an XML string following the required schema. "
            "The plan will be validated. If validation fails, you will receive error "
            "messages and should correct the plan and resubmit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_xml": {
                    "type": "string",
                    "description": "The complete plan in XML format.",
                },
            },
            "required": ["plan_xml"],
        },
    },
}


class SubmitPlanTool:
    """Validates an XML plan and raises :class:`PlanSubmitted` on success."""

    @property
    def name(self) -> str:
        return "submit_plan"

    def get_schema(self) -> dict:
        return SUBMIT_PLAN_SCHEMA

    async def execute(self, args: dict) -> dict:
        plan_xml = args.get("plan_xml", "")
        if not plan_xml.strip():
            return {"output": "Error: 'plan_xml' must not be empty.", "error": True}

        plan, errors = parse_and_validate(plan_xml)

        if errors:
            error_text = "Plan validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            return {"output": error_text, "error": True}

        raise PlanSubmitted(
            plan_xml,
            {
                "role": "exit",
                "content": "Plan submitted and validated successfully.",
                "extra": {
                    "exit_status": "plan_submitted",
                    "plan_xml": plan_xml,
                    "plan": plan,
                },
            },
        )

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
