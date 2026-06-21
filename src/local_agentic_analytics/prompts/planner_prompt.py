"""Prompt template for retrieval-route planning with a small local SLM.

gemma2:2b is weak, so the prompt is extremely explicit: it enumerates exactly
three labels, gives two worked examples per route, and demands a single bare
label as output. Parsing on the agent side is forgiving regardless.
"""

from __future__ import annotations


PLANNER_PROMPT_TEMPLATE = """Anda adalah router pertanyaan analitik keuangan.

Tugas:
Pilih SATU rute pengambilan data yang paling tepat untuk menjawab pertanyaan user.

Rute yang tersedia (jawab PERSIS salah satu label ini, huruf besar semua):
- STRUCTURED_SQL : pertanyaan tentang angka harga saham (rata-rata, tertinggi,
  terendah, volume, jumlah hari) yang dijawab dari tabel harga.
- RAG_NEWS : pertanyaan tentang berita, headline, sentimen, analis, atau
  publisher yang dijawab dari kumpulan berita.
- HYBRID : pertanyaan yang meminta pergerakan/tren harga SEKALIGUS dikaitkan
  dengan beritanya.

Aturan keluaran:
- Jawab HANYA satu label: STRUCTURED_SQL, RAG_NEWS, atau HYBRID.
- Tanpa tanda baca, tanpa penjelasan, tanpa kalimat tambahan.

Contoh:
Q: Berapa rata-rata harga penutupan NVDA antara 2 Januari 2019 dan 31 Januari 2019?
A: STRUCTURED_SQL
Q: Berapa harga penutupan tertinggi TSLA selama periode dataset?
A: STRUCTURED_SQL
Q: Bagaimana sentimen berita terbaru tentang TSLA?
A: RAG_NEWS
Q: Berita analis apa saja yang muncul terkait NVDA?
A: RAG_NEWS
Q: Ringkas pergerakan harga NVDA pada Juni 2019 dan kaitkan dengan beritanya.
A: HYBRID
Q: Bagaimana performa harga TSLA pada awal 2020 dan apa berita pendukungnya?
A: HYBRID

Q: {question}
A:"""


def build_planner_prompt(question: str) -> str:
    """Build the route-planning prompt for one user question."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    return PLANNER_PROMPT_TEMPLATE.format(question=question.strip())
