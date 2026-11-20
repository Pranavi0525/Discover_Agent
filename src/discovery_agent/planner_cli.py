import argparse
import json

from .planner import create_planner


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a discovery plan with Gemini.")
    parser.add_argument("query", help="Natural-language shopping request")
    args = parser.parse_args()
    plan = create_planner().plan(args.query)
    print(json.dumps(plan.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()