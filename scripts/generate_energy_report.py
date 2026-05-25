from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.cli import main as cli_main


def main() -> int:
    return cli_main(["report", "energy"])


if __name__ == "__main__":
    raise SystemExit(main())
