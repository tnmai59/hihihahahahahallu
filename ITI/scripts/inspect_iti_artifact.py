#!/usr/bin/env python
from __future__ import annotations

import argparse

from iti_paper.config import ITIDirections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ITI directions artifact.")
    parser.add_argument("path")
    parser.add_argument("--num-heads", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = ITIDirections.load(args.path)
    print(f"path={args.path}")
    print(f"directions_shape={tuple(artifact.directions.shape)}")
    print(f"selected_heads={artifact.selected_heads[:args.num_heads]}")
    print(f"best_probe_accuracy={artifact.probe_accuracies.max().item():.4f}")
    print("metadata:")
    for key, value in sorted(artifact.metadata.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
