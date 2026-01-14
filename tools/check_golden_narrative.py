from __future__ import annotations

"""
Golden narrative regression checker (H.2/H.3).

NOTE: UI narrative generation has been removed. This tool is now disabled.
"""

import argparse
import sys

def _disabled() -> None:
    print("Legacy module removed. Use pack-only NarrativeDraftPack/NarrativeFinalPack.")
    sys.exit(2)


def main() -> None:
    _ = argparse.ArgumentParser()
    _disabled()


if __name__ == "__main__":
    main()
