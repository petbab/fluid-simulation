#!/usr/bin/env python3
"""Extract the best configuration from a KTT JSON result file."""

import json
import sys


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <result.json>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    # KTT result files have a "Results" array; each element has "Configuration"
    results = data.get("Results", [])
    if not results:
        print("No 'Results' array found in JSON", file=sys.stderr)
        sys.exit(1)

    # Find the result with the best (lowest) total duration
    best = min(results, key=lambda r: r.get("TotalDuration", float("inf")))
    config = best.get("Configuration", [])

    # Print as a JSON array that load_config_json accepts
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
