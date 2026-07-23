# Prompt Judge — Jalankan di SESI CLAUDE CODE BARU

Salin **seluruh blok di dalam pagar `====` di bawah** dan tempel sebagai pesan
pertama di sesi Claude Code yang **baru** (bukan sesi ini — sesi ini sudah
melihat kedua report dan tidak netral). Sesi baru = konteks bersih = judge
tak terkontaminasi. Autentikasinya memakai langganan Pro Anda.

Alternatif headless (sama saja): `claude -p "$(cat docs/llm_judge_prompt.md)"`
setelah menghapus baris-baris panduan ini — atau cukup tempel blok di bawah.

Prasyarat: `reports/experiments/judge/narration_pairs.jsonl` sudah ada
(dihasilkan `scripts/export_report_narrations_for_judging.py`).

============================================================================
Anda adalah penilai (judge) yang netral dan ketat untuk kualitas narasi insight
laporan energi berbahasa Indonesia. Tugas Anda menilai pasangan narasi secara
BUTA — Anda tidak tahu model mana yang menghasilkan "A" atau "B", dan Anda
DILARANG menebak atau mencoba mencari tahu.

ATURAN INTEGRITAS (WAJIB):
- Baca HANYA file `reports/experiments/judge/narration_pairs.jsonl`.
- JANGAN membuka file lain, JANGAN menjelajah repositori, dan secara khusus
  JANGAN membuka `blinding_key.json` atau file apa pun yang memetakan A/B ke
  nama model. Membukanya membatalkan validitas penilaian.
- Nilai setiap item berdasarkan dirinya sendiri. JANGAN membandingkan antar
  item atau membiarkan item sebelumnya memengaruhi item berikutnya.

DATA: setiap baris JSONL berisi `item_id`, `chart_id`, `stat_block` (statistik
input deterministik — ini SATU-SATUNYA sumber kebenaran angka), `narration_A`,
`narration_B`.

Untuk SETIAP item, nilai narration_A dan narration_B secara TERPISAH pada 5
dimensi, skala 1–5 (bilangan bulat):

1. faithfulness — setiap angka/klaim berasal dari stat_block (atau turunan sah
   seperti load factor); tidak ada angka/benchmark/prediksi karangan.
   5 = semua klaim didukung; 1 = berhalusinasi/kontradiktif dengan stat_block.
2. unit_correctness — satuan baku benar & konsisten (kW/kWh/Wh/V/A) menyertai
   angka. 5 = semua benar; 1 = salah/tanpa satuan.
3. domain_richness — pemakaian istilah domain (load factor, beban puncak/lembah,
   koefisien korelasi, load shifting, dll.) yang TEPAT — bukan sekadar banyak.
   5 = tepat & informatif; 1 = tidak ada / salah pakai.
4. coherence — alur logis antar kalimat, tidak kontradiktif, tidak repetitif.
   5 = mengalir & konsisten; 1 = kacau/berulang/bertentangan.
5. fluency_id — kualitas Bahasa Indonesia formal-teknis (tata bahasa, tidak ada
   token asing/artefak). 5 = mulus; 1 = banyak kesalahan/artefak.

Lalu tentukan PREFERENSI BERPASANGAN (buta) untuk item itu:
- pref_semantic — narasi mana yang lebih baik secara keseluruhan (kesetiaan +
  ketepatan + kekayaan). Isi "A", "B", atau "tie".
- pref_coherence — narasi mana yang lebih koheren & mudah dipahami. "A"/"B"/"tie".

KELUARAN: tulis file `reports/experiments/judge/judge_scores.csv` dengan header
persis berikut dan SATU baris per item (jangan menambah/mengurangi kolom):

item_id,A_faithfulness,A_unit_correctness,A_domain_richness,A_coherence,A_fluency_id,B_faithfulness,B_unit_correctness,B_domain_richness,B_coherence,B_fluency_id,pref_semantic,pref_coherence,note

- Skor adalah bilangan bulat 1–5. pref_* adalah "A"/"B"/"tie".
- `note`: satu frasa singkat berisi bukti kutipan yang mendasari penilaian
  (mis. alasan skor faithfulness rendah). Bungkus dengan tanda kutip ganda bila
  mengandung koma; jangan pakai newline di dalam sel.
- Proses SEMUA item di file. Setelah selesai, laporkan jumlah item yang dinilai
  dan tampilkan 3 baris pertama CSV sebagai verifikasi.

Kerjakan sekarang: baca narration_pairs.jsonl, nilai semua item sesuai rubrik,
lalu tulis judge_scores.csv.
============================================================================

## Setelah judge selesai

Kembali ke sesi ini (atau jalankan sendiri) dan jalankan analisis — ini yang
meng-unblind A/B dan menghitung statistiknya:

```powershell
./.venv/Scripts/python.exe scripts/analyze_judge_results.py
```

Output: `reports/experiments/judge/judge_summary.csv` + `.md` (skor absolut per
model dengan 95% CI bootstrap, dan preferensi berpasangan dengan uji sign eksak).

## Catatan validitas (untuk laporan)

- Ini **draft judge LLM sesi tunggal**. Semua item dinilai dalam satu konteks,
  jadi independensi antar-item tidak sempurna (mitigasi: instruksi menilai tiap
  item terpisah). Untuk klaim kuat, validasi dengan subset 20–30 item dinilai
  dua penilai manusia (rubrik sama) dan laporkan korelasi judge-vs-manusia —
  lihat `docs/llm_judge_report_eval_plan.md`.
- Judge = keluarga Claude, teacher dataset = DeepSeek → beda keluarga, sehingga
  bias self-preference terhadap gaya teacher tidak berlaku.
