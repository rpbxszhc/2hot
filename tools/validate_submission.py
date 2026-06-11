#!/usr/bin/env python
"""Validate the challenge submission zip structure and JSON schema."""

import argparse
import json
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        names = zf.namelist()
        if names != ["result.json"]:
            raise SystemExit(f"Expected zip to contain only result.json, got {names}")
        data = json.loads(zf.read("result.json"))

    expected_keys = [str(i) for i in range(2468)]
    if sorted(data.keys(), key=int) != expected_keys:
        raise SystemExit("Keys must be contiguous strings from '0' to '2467'")
    bad_values = [v for v in data.values() if not isinstance(v, int) or v < 0 or v >= 40]
    if bad_values:
        raise SystemExit("Values must be integers in [0, 39]")
    print(f"OK: {args.zip_path} contains {len(data)} predictions")


if __name__ == "__main__":
    main()
