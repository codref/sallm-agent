"""``python -m sallm.tools.memory`` entry (avoids importing app via package __init__)."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
