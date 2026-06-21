from datetime import date
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_finance_news import embed_news, load_news_documents


FAKE_CSV = """,headline,url,publisher,date,stock
0,NVDA hits new high,http://x/1,Benzinga,2019-06-03 10:30:54-04:00,NVDA
1,Old NVDA note,http://x/2,Benzinga,2018-12-31 09:00:00-04:00,NVDA
2,Some other stock,http://x/3,Reuters,2019-06-03 09:00:00-04:00,A
3,TSLA upgrade,http://x/4,Reuters,2020-06-09 11:00:00-04:00,TSLA
4,Too late TSLA,http://x/5,Reuters,2020-07-01 11:00:00-04:00,TSLA
"""


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "raw_analyst_ratings.csv"
    csv_path.write_text(FAKE_CSV, encoding="utf-8")
    return csv_path


def test_load_news_documents_filters_tickers_and_date_range(tmp_path):
    csv_path = _write_csv(tmp_path)

    loaded = load_news_documents(
        csv_path,
        tickers=("NVDA", "NFLX", "TSLA", "GOOGL"),
        start=date(2019, 1, 1),
        end=date(2020, 6, 10),
    )

    documents = loaded["documents"]
    metadatas = loaded["metadatas"]

    # Only the in-range NVDA and TSLA rows survive (other ticker, pre-range,
    # and post-range rows are dropped).
    assert documents == ["NVDA hits new high", "TSLA upgrade"]
    assert loaded["stats"]["total_rows"] == 5
    assert loaded["stats"]["matched_rows"] == 2

    first = metadatas[0]
    assert first["ticker"] == "NVDA"
    assert first["date"] == "2019-06-03"
    assert first["publisher"] == "Benzinga"
    assert first["url"] == "http://x/1"


class _FakeChromaTool:
    def __init__(self):
        self.batches = []

    def add_documents(self, documents, metadatas):
        self.batches.append(list(documents))
        return {"added": len(documents), "skipped": 0, "ids": []}


def test_embed_news_batches_documents():
    tool = _FakeChromaTool()
    documents = [f"headline {i}" for i in range(5)]
    metadatas = [{"ticker": "NVDA"} for _ in range(5)]

    summary = embed_news(tool, documents, metadatas, batch_size=2)

    assert summary == {"added": 5, "skipped": 0}
    assert [len(batch) for batch in tool.batches] == [2, 2, 1]
