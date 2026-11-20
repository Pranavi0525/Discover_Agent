import argparse
import json

from .graph import WorkflowError, run_discovery_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Gemini planner -> LangGraph -> Shopify MCP executor workflow (M1.4)."
    )
    parser.add_argument("query", help="Natural-language shopping request")
    args = parser.parse_args()
    try:
        state = run_discovery_workflow(args.query)
    except WorkflowError as exc:
        parser.error(str(exc))
        return
    decision = state.get("decision")
    payload = {
        "graph_iterations": state.get("graph_iterations"),
        "refinement_count": state.get("refinement_count"),
        "decision": decision.model_dump(mode="json") if decision else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()