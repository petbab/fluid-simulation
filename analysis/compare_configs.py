#!/usr/bin/env python3
"""Compare two JSON configuration files (UnsignedInt and Double parameters only)."""

import json
import sys
from pathlib import Path


def load_config(path: Path) -> dict:
    """Load a config file and return a dict mapping Name -> Value for numeric types only."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        entry["Name"]: entry["Value"]
        for entry in data
        if entry.get("ValueType") in ("UnsignedInt", "Double")
    }


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <config1.json> <config2.json>", file=sys.stderr)
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    for p in (path1, path2):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    cfg1 = load_config(path1)
    cfg2 = load_config(path2)

    name1 = path1.stem
    name2 = path2.stem

    all_params = sorted(set(cfg1.keys()) | set(cfg2.keys()))

    # Determine column widths
    param_width = max(len(p) for p in all_params)
    val_width1 = max(len(str(v)) for v in cfg1.values()) if cfg1 else 0
    val_width2 = max(len(str(v)) for v in cfg2.values()) if cfg2 else 0
    name_width = max(len(name1), len(name2), val_width1, val_width2, 10)

    # Header
    header = f"{'Parameter':<{param_width}} | {name1:^{name_width}} | {name2:^{name_width}}"
    print(header)
    print("-" * len(header))

    differing = []
    for param in all_params:
        v1 = cfg1.get(param, "<missing>")
        v2 = cfg2.get(param, "<missing>")
        marker = " ***" if v1 != v2 else ""
        row = f"{param:<{param_width}} | {str(v1):>{name_width}} | {str(v2):>{name_width}}{marker}"
        print(row)
        if v1 != v2:
            differing.append((param, v1, v2))

    print()
    print(f"Total parameters: {len(all_params)}")
    print(f"Differences: {len(differing)}")

    if differing:
        print()
        print("Differing parameters:")
        for param, v1, v2 in differing:
            print(f"  {param}:")
            print(f"    {name1}: {v1}")
            print(f"    {name2}: {v2}")


if __name__ == "__main__":
    main()
