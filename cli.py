"""Unified CLI wrapper for training and visualization.

Usage examples:
- python cli.py train [trainer-args]
- python cli.py viz [visualizer-args]

This consolidates root-level wrappers (trainer.py, visualizer.py)
into one entry-point while delegating to package modules:
- soil_segment.trainer:main
- soil_segment.visualizer:main
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List


def _dispatch(argv: List[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print("Soil Segment CLI")
        print("\nCommands:")
        print("  train        Run training (soil_segment.trainer:main)")
        print("  viz          Plot training history (soil_segment.visualizer:main)")
        print("\nExamples:")
        print("  python cli.py train --help")
        print("  python cli.py viz checkpoints --save checkpoints/curves.png")
        return 0

    cmd = argv[1].lower()

    # Lazy imports so that help text works even if deps are missing
    def _train_main():
        try:
            from soil_segment.trainer import main as _main  # type: ignore
        except ModuleNotFoundError:
            # Allow running from repo root without installing the package
            import os, sys as _sys
            here = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(here, "src")
            if os.path.isdir(src_path) and src_path not in _sys.path:
                _sys.path.insert(0, src_path)
            from soil_segment.trainer import main as _main  # type: ignore
        return _main()

    def _viz_main():
        try:
            from soil_segment.visualizer import main as _main  # type: ignore
        except ModuleNotFoundError:
            import os, sys as _sys
            here = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(here, "src")
            if os.path.isdir(src_path) and src_path not in _sys.path:
                _sys.path.insert(0, src_path)
            from soil_segment.visualizer import main as _main  # type: ignore
        return _main()

    routes: Dict[str, Callable[[], int | None]] = {
        "train": _train_main,
        "trainer": _train_main,
        "viz": _viz_main,
        "visualizer": _viz_main,
        "plot": _viz_main,
    }

    if cmd not in routes:
        print(f"Unknown command: {cmd}")
        print("Use --help to see available commands.")
        return 2

    # Forward remaining args to the selected subcommand's parser
    sub_argv = [f"{argv[0]} {cmd}"] + argv[2:]
    sys.argv = sub_argv
    result = routes[cmd]()
    return int(result) if isinstance(result, int) else 0


def main() -> None:
    raise SystemExit(_dispatch(sys.argv))


if __name__ == "__main__":
    main()
