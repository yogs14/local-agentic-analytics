# Prompt Judge — Gelombang Guru Sekeluarga (DUA sesi terpisah)

Dokumen ini menyiapkan penilaian untuk dua perbandingan berpasangan pada
gelombang guru sekeluarga:

| Perbandingan | Direktori | Narator kiri | Narator kanan |
|---|---|---|---|
| Cabang Gemma | `reports/experiments/judge_gemma_fam/` | `gemma2-energy-insight:v3` (guru DeepSeek) | `gemma2-energy-insight-fam:v1` (guru Gemma) |
| Cabang Qwen | `reports/experiments/judge_qwen_fam/` | `qwen25-energy-insight:v1` (guru DeepSeek) | `qwen25-energy-insight-fam:v1` (guru Qwen) |

> **Jalankan sebagai DUA sesi Claude Code yang berbeda dan bersih.** Jangan
> menilai kedua perbandingan dalam satu sesi: konteks dari perbandingan pertama
> akan mencemari perbandingan kedua. Sesi ini (yang menyiapkan data) juga tidak
> boleh menjadi juri karena sudah melihat kedua sisi.

## Prasyarat

Pasangan narasi harus sudah diekspor lebih dulu (butuh Ollama lokal dengan
keempat tag terdaftar):

```powershell
./.venv/Scripts/python.exe scripts/export_report_narrations_for_judging.py `
  --n-per-chart 6 --seed 20260722 `
  --left-label gemma_cross --left-tag gemma2-energy-insight:v3 `
  --right-label gemma_fam  --right-tag gemma2-energy-insight-fam:v1 `
  --output-dir reports/experiments/judge_gemma_fam

./.venv/Scripts/python.exe scripts/export_report_narrations_for_judging.py `
  --n-per-chart 6 --seed 20260722 `
  --left-label qwen_cross --left-tag qwen25-energy-insight:v1 `
  --right-label qwen_fam  --right-tag qwen25-energy-insight-fam:v1 `
  --output-dir reports/experiments/judge_qwen_fam
```

Seed sengaja disamakan (20260722) agar kedua cabang dinilai pada blok statistik
yang sama, sehingga hasil cabang Gemma dan Qwen dapat disandingkan. Seed ini
berbeda dari seed dataset (42) dan skrip tetap menjalankan pemeriksaan kebocoran
terhadap `train.jsonl` / `val.jsonl`.

## Catatan yang wajib dicatat saat menjalankan juri

Sesuai `docs/llm_judge_report_eval_plan.md`, catat untuk tiap sesi: versi model
juri, tanggal penilaian, dan jumlah item. Ketiganya masuk ke Bab 3 dan lampiran.

---

## SESI 1 — Cabang Gemma

Salin seluruh blok di dalam pagar `====` di bawah sebagai pesan pertama pada
sesi Claude Code yang baru.

============================================================================
Anda adalah penilai (judge) yang netral dan ketat untuk kualitas narasi insight
laporan energi berbahasa Indonesia. Tugas Anda menilai pasangan narasi secara
BUTA — Anda tidak tahu model mana yang menghasilkan "A" atau "B", dan Anda
DILARANG menebak atau mencoba mencari tahu.

ATURAN INTEGRITAS (WAJIB):
- Baca HANYA file `reports/experiments/judge_gemma_fam/narration_pairs.jsonl`.
- JANGAN membuka file lain, JANGAN menjelajah repositori, dan secara khusus
  JANGAN membuka `blinding_key.json` atau file apa pun yang memetakan A/B ke
  nama model. Membukanya membatalkan validitas penilaian.
- Kedua narasi berasal dari model yang SANGAT MIRIP (hanya berbeda sumber data
  pelatihan), jadi perbedaannya bisa halus. Jangan memaksakan pemenang: gunakan
  "tie" bila memang setara.
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
   angka. Perhatikan khusus: satuan yang KELIRU (mis. menulis kWh untuk kolom
   ber-satuan Wh) harus menurunkan skor, bukan sekadar satuan yang hilang.
   5 = semua benar; 1 = salah/tanpa satuan.
3. domain_richness — pemakaian istilah domain (load factor, beban puncak/lembah,
   koefisien korelasi, load shifting, dll.) yang TEPAT — bukan sekadar banyak.
   5 = tepat & informatif; 1 = tidak ada / salah pakai.
4. coherence — alur logis antar kalimat, tidak kontradiktif, tidak repetitif.
   Narasi yang bertele-tele atau mengulang gagasan menurunkan skor.
   5 = mengalir & konsisten; 1 = kacau/berulang/bertentangan.
5. fluency_id — kualitas Bahasa Indonesia formal-teknis (tata bahasa, tidak ada
   token asing/artefak). 5 = mulus; 1 = banyak kesalahan/artefak.

Lalu tentukan PREFERENSI BERPASANGAN (buta) untuk item itu:
- pref_semantic — narasi mana yang lebih baik secara keseluruhan (kesetiaan +
  ketepatan + kekayaan). Isi "A", "B", atau "tie".
- pref_coherence — narasi mana yang lebih koheren & mudah dipahami. "A"/"B"/"tie".

KELUARAN: tulis file `reports/experiments/judge_gemma_fam/judge_scores.csv`
dengan header persis berikut dan SATU baris per item (jangan menambah/mengurangi
kolom):

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

## SESI 2 — Cabang Qwen

Sama persis dengan SESI 1, kecuali kedua path diganti menjadi
`reports/experiments/judge_qwen_fam/`. Salin blok SESI 1 ke sesi Claude Code
yang **baru lagi**, lalu ganti:

- baca: `reports/experiments/judge_qwen_fam/narration_pairs.jsonl`
- tulis: `reports/experiments/judge_qwen_fam/judge_scores.csv`

## Setelah kedua sesi juri selesai

Jalankan analisis per direktori — inilah yang meng-unblind A/B dan menghitung
statistiknya:

```powershell
./.venv/Scripts/python.exe scripts/analyze_judge_results.py --judge-dir reports/experiments/judge_gemma_fam
./.venv/Scripts/python.exe scripts/analyze_judge_results.py --judge-dir reports/experiments/judge_qwen_fam
```

Keluaran: `judge_summary.csv` + `.md` pada masing-masing direktori, berisi skor
absolut per model dengan 95% CI bootstrap dan preferensi berpasangan dengan uji
sign eksak.

## Catatan validitas (untuk laporan)

- Juri berasal dari keluarga Claude, sedangkan guru pembangkit dataset pada
  gelombang ini adalah Gemma dan Qwen. Ketiganya berbeda keluarga, sehingga
  bias self-preference terhadap gaya guru tidak berlaku. Ini juga menjaga
  konsistensi dengan gelombang pertama (guru DeepSeek, juri Claude).
- Tetap merupakan draft judge LLM sesi tunggal per perbandingan. Untuk klaim
  kuat, validasi dengan subset 20–30 item yang dinilai dua penilai manusia
  memakai rubrik yang sama, lalu laporkan korelasi judge-vs-manusia
  (lihat `docs/llm_judge_report_eval_plan.md`).
- Karena kedua narator dalam satu perbandingan hanya berbeda pada asal guru,
  proporsi "tie" yang tinggi adalah hasil yang sah dan bermakna: ia menunjukkan
  guru sekeluarga tidak memberi perbedaan kualitas yang terdeteksi.
