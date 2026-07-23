"""Teacher prompt assembly for synthetic insight generation.

The template is the user-supplied, self-contained anti-hallucination prompt: it
embeds its own concept bank and hard rules, and exposes a single
``{STATISTIK_INPUT}`` slot. The statistics block is recovered by an offline fake
teacher via the ``=== STATISTIK INPUT ===`` section header, so no extra markers
pollute the prompt the real teacher sees.
"""

from __future__ import annotations


STATS_INPUT_PLACEHOLDER = "{STATISTIK_INPUT}"
STATS_INPUT_HEADER = "=== STATISTIK INPUT ==="


# Instruction stored on each JSONL example (what the student learns to do).
DEFAULT_INSTRUCTION = (
    "Ubah statistik deterministik sebuah grafik konsumsi listrik rumah tangga "
    "menjadi narasi insight 3-5 kalimat dalam Bahasa Indonesia formal-teknis. "
    "Setiap angka harus berasal dari statistik input (atau turunannya seperti "
    "load factor dan koefisien variasi) dan disertai satuan baku; gunakan "
    "minimal dua istilah domain yang relevan tanpa mengarang angka."
)


TEACHER_PROMPT_TEMPLATE = """Anda adalah analis data energi senior yang menyusun narasi insight untuk laporan
analitik konsumsi listrik rumah tangga. Tugas Anda: mengubah statistik deterministik
sebuah grafik menjadi narasi 3-5 kalimat dalam Bahasa Indonesia formal-teknis.

=== STATISTIK INPUT ===
{STATISTIK_INPUT}

=== KOSAKATA YANG BOLEH DIPAKAI (concept bank) ===
Statistika: rata-rata, median, simpangan baku, koefisien variasi (CV=SD/mean),
modus bin, distribusi, outlier, korelasi Pearson (hanya jika r disediakan), tren,
rentang, beban puncak, base load.
Kelistrikan: daya aktif (kW), daya reaktif (kVAR), energi (kWh), tegangan (V),
intensitas arus (A), load factor (mean/puncak), kurva beban, profil beban
residensial, disparitas puncak-lembah, stabilitas tegangan, sub-metering,
demand response (kualitatif), load shifting (kualitatif), peak shaving (kualitatif).

=== ATURAN KERAS (WAJIB) ===
1. Setiap ANGKA dalam narasi HARUS berasal dari STATISTIK INPUT, atau dihitung
   langsung darinya (mis. load factor = rata-rata/puncak; CV = SD/mean). Tuliskan
   angka persis seperti di input (jaga koma desimal, mis. "1,091 kW").
2. Setiap nilai WAJIB disertai SATUAN baku: kW, kWh, V, A, Wh, kVAR.
3. DILARANG menyebut: standar/benchmark eksternal berangka (IEEE, SPLN, PLN "X%"),
   hasil uji statistik yang tidak disediakan (p-value, ANOVA, Jarque-Bera),
   prediksi kuantitatif masa depan ("turun 15%", "likelihood 19%"), atau
   perbandingan industri/kompetitor berangka.
4. Interpretasi KUALITATIF diperbolehkan dan didorong ("mengindikasikan",
   "konsisten dengan profil residensial", "berpotensi mendukung load shifting"),
   selama tidak menambah angka baru.
5. Pakai minimal 2 istilah dari concept bank yang RELEVAN dengan jenis grafik ini.
   Jangan paksakan istilah yang tidak didukung data.
6. Variasikan struktur kalimat - jangan memulai dengan pola yang sama berulang.

=== STRUKTUR NARASI (longgar, bukan template kaku) ===
- Buka dengan temuan utama + angka kunci & satuan.
- Sebut 1-2 konsep domain yang menjelaskan pola tersebut.
- Tutup dengan interpretasi kualitatif yang bermakna (mengapa / apa artinya).

=== OUTPUT ===
Keluarkan HANYA narasi final. Tanpa preambule, tanpa penjelasan, tanpa markdown.
"""


def build_teacher_prompt(stats_block: str, **_ignored: object) -> str:
    """Inject a statistics block into the teacher prompt.

    Extra keyword arguments are accepted and ignored so the call site stays
    stable even though this template is self-contained (concept bank embedded).
    """
    return TEACHER_PROMPT_TEMPLATE.replace(STATS_INPUT_PLACEHOLDER, stats_block.strip())


def extract_stats_block(prompt: str) -> str:
    """Recover the statistics block from a built prompt (used by the fake teacher)."""
    start = prompt.find(STATS_INPUT_HEADER)
    if start == -1:
        raise ValueError("Prompt does not contain a '=== STATISTIK INPUT ===' section")
    rest = prompt[start + len(STATS_INPUT_HEADER) :]
    end = rest.find("\n===")
    block = rest if end == -1 else rest[:end]
    return block.strip()
