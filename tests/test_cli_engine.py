from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics import cli


def test_ask_parser_accepts_custom_engine_after_question():
    parser = cli.build_parser()

    args = parser.parse_args(["ask", "Berapa nilainya?", "--engine", "custom"])

    assert args.command == "ask"
    assert args.question == ["Berapa nilainya?"]
    assert args.engine == "custom"


def test_ask_parser_accepts_langgraph_engine_after_question():
    parser = cli.build_parser()

    args = parser.parse_args(["ask", "Berapa nilainya?", "--engine", "langgraph"])

    assert args.command == "ask"
    assert args.question == ["Berapa nilainya?"]
    assert args.engine == "langgraph"


def test_report_parser_defaults_to_custom_engine():
    parser = cli.build_parser()

    args = parser.parse_args(["report", "energy"])

    assert args.command == "report"
    assert args.report_type == "energy"
    assert args.engine == "custom"


def test_report_parser_accepts_custom_engine():
    parser = cli.build_parser()

    args = parser.parse_args(["report", "energy", "--engine", "custom"])

    assert args.command == "report"
    assert args.report_type == "energy"
    assert args.engine == "custom"


def test_report_parser_accepts_langgraph_engine():
    parser = cli.build_parser()

    args = parser.parse_args(["report", "energy", "--engine", "langgraph"])

    assert args.command == "report"
    assert args.report_type == "energy"
    assert args.engine == "langgraph"
