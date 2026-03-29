"""Rich tree rendering of a ForecastPlan."""

from __future__ import annotations

from rich.tree import Tree

from miniprophet.cli.utils import get_console
from miniprophet.planning.schema import ForecastPlan, SubQuery


def _trigger_label(trigger: str | None) -> str:
    if not trigger:
        return ""
    short = trigger[:80] + ("..." if len(trigger) > 80 else "")
    return f"  [dim italic](trigger: {short})[/dim italic]"


def _add_sub_queries(branch: Tree, queries: list[SubQuery]) -> None:
    if not queries:
        return
    sq_branch = branch.add("[bold]Sub-queries[/bold]")
    for sq in queries:
        label = f"[magenta]\\[{sq.id}][/magenta] {sq.query}{_trigger_label(sq.trigger_condition)}"
        sq_branch.add(label)


def build_plan_tree(plan: ForecastPlan) -> Tree:
    """Build a Rich :class:`Tree` from a :class:`ForecastPlan`."""
    root_label = (
        f"[bold cyan]Plan:[/bold cyan] {plan.main_problem_title}"
        f"  [dim](id: {plan.main_problem_id})[/dim]"
    )
    tree = Tree(root_label)

    # Main problem sub-queries
    _add_sub_queries(tree, plan.sub_queries)

    # Sub-problems
    if plan.sub_problems:
        sp_branch = tree.add("[bold]Sub-problems[/bold]")
        for sp in plan.sub_problems:
            sp_label = (
                f"[magenta]\\[{sp.id}][/magenta] {sp.title}{_trigger_label(sp.trigger_condition)}"
            )
            sp_node = sp_branch.add(sp_label)
            _add_sub_queries(sp_node, sp.sub_queries)

    # Factors
    if plan.factors:
        f_branch = tree.add("[bold]Factors to consider[/bold]")
        for factor in plan.factors:
            f_branch.add(f"[dim]{factor}[/dim]")

    return tree


def print_plan(plan: ForecastPlan) -> None:
    """Print the plan tree to the shared console."""
    console = get_console()
    console.print()
    console.print(build_plan_tree(plan))
    console.print()
