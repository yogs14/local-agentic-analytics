# Rencana Evaluasi Narasi Report dengan LLM-as-a-Judge

Menilai kualitas **semantik** dan **koherensi** narasi report LaTeX/PDF antara
narrator hasil fine-tune, melengkapi (bukan menggantikan) checker deterministik
yang sudah ada.

## 1. Pertanyaan Riset & Kondisi Eksperimen

Dua sumbu perbandingan, desain 2×2 (mengikuti pola gelombang fine-tune di
`docs/finetune_recipe.md`):

| Kondisi | Model | Fine-tune |
|---|---|---|
| A | gemma2-energy-insight:v3 | ya (existing) |
| B | qwen25-energy-insight:v1 | ya (baru, `finetune-qwen-energy.ipynb`) |
| C | gemma2:2b base | tidak |
| D | qwen2.5:1.5b base | tidak |

- **Antar-model fine-tuned:** A vs B — pertanyaan utama.
- **Kontribusi fine-tuning per model:** A vs C, B vs D.

Semua kondisi memakai stat block input yang SAMA (statistik chart
deterministik dari pipeline report), instruction sama, `temperature 0.4`
(paritas dengan konfigurasi `insight_model`). Satu-satunya variabel per
perbandingan adalah narratornya.

## 2. Unit Evaluasi & Ukuran Sampel

1. **Level narasi-chart** (unit utama): satu stat block → satu narasi per
   kondisi. Target **≥ 30 stat block unik** (variasikan chart_id dan periode
   data) → 30 tuple berpasangan × 4 kondisi = 120 narasi.
2. **Level report utuh** (sekunder): ≥ 10 report penuh per kondisi A/B untuk
   penilaian koherensi antar-bagian (alur, tidak kontradiktif, transisi).

Berpasangan per stat block — memungkinkan uji berpasangan (bagian 6).

## 3. Protokol Judging (dua lapis)

### 3a. Skor absolut per dimensi (skala 1–5, JSON terstruktur)

Judge menerima: stat block (ground truth angka) + narasi (satu per panggilan),
TANPA identitas model. Dimensi:

| Dimensi | Definisi operasional |
|---|---|
| faithfulness | semua angka/klaim berasal dari stat block; tanpa benchmark/prediksi karangan |
| unit_correctness | satuan baku benar & konsisten (kW/kWh/Wh/V/A) |
| domain_richness | penggunaan istilah domain (load factor, beban puncak, dsb.) yang TEPAT, bukan sekadar banyak |
| coherence | alur logis antar kalimat, tidak kontradiktif, tidak repetitif |
| fluency_id | kualitas Bahasa Indonesia formal-teknis |

Output judge dipaksa JSON (`{"faithfulness": 4, ...,"evidence": "..."}`);
field `evidence` wajib mengutip frasa narasi — menekan skor asal.

### 3b. Pairwise preference (protokol yang lebih reliabel)

Judge menerima stat block + DUA narasi (label netral "Narasi 1"/"Narasi 2")
dan memilih `1 | 2 | seri` untuk (i) semantik keseluruhan dan (ii) koherensi.
Kontrol bias posisi: **setiap pasangan dinilai dua kali dengan urutan
ditukar**; hasil dipakai hanya jika konsisten, selainnya dihitung `seri`
(dilaporkan sebagai `position_inconsistent_rate`).

## 4. Pemilihan & Validasi Judge

- **Bukan model lokal 2B** — sudah dicatat tidak reliabel di rencana induk.
- **Bukan DeepSeek** — deepseek adalah *teacher* yang membuat dataset
  fine-tune, sehingga berisiko *self-preference bias* terhadap gaya narasinya
  sendiri. Judge harus dari keluarga model berbeda: **Claude API** (dependensi
  `anthropic` sudah ada di requirements) atau GPT sebagai alternatif.
- Judge dijalankan `temperature 0`, versi model + hash prompt dicatat di
  output (reproducibility).
- **Validasi wajib sebelum hasil judge dipercaya:** dua penilai manusia menilai
  subset acak **20–30 narasi** dengan rubrik yang sama (prosedur independen +
  adjudikasi ala `docs/rag_generation_rubric.md`). Laporkan korelasi Spearman
  judge-vs-manusia per dimensi dan agreement ≤1 poin. Ambang: **ρ ≥ 0,6** pada
  faithfulness & coherence; di bawah itu, hasil judge diturunkan statusnya
  menjadi indikatif dan kesimpulan bersandar pada penilai manusia.

## 5. Lapisan Deterministik (tetap berjalan, gratis, objektif)

`unit_rule_compliance` dan `numeric_fact_coverage` (checker existing pipeline
report) dihitung untuk SEMUA narasi — angka halusinasi terdeteksi secara
deterministik, tidak bergantung judge. Judge menambah dimensi yang tak
terjangkau checker (koherensi, ketepatan jargon, fluency).

## 6. Analisis Statistik (reuse `evaluation/statistics.py`)

- **Pairwise:** win rate B-vs-A + **uji sign eksak** pada pasangan diskordan
  (`mcnemar_exact` — matematis sama), per dimensi preference.
- **Skor absolut:** mean ± std per dimensi per kondisi + **bootstrap 95% CI**
  (`bootstrap_rate_ci` digeneralisasi ke mean skor); selisih berpasangan per
  stat block (Wilcoxon signed-rank via scipy bila distribusi memungkinkan).
- Laporkan `n` di semua tabel; jangan klaim beda bila CI tumpang tindih.

## 7. Implementasi (urutan pengerjaan)

1. `scripts/export_report_narrations_for_judging.py` — generate narasi 4
   kondisi dari stat block yang sama (ganti narrator via `OLLAMA_INSIGHT_MODEL`
   / section `insight_model`), simpan
   `reports/experiments/narration_pairs.jsonl` **ter-blind** (id acak,
   mapping identitas di file terpisah yang tidak dilihat judge/penilai).
2. `scripts/run_llm_judge.py` — panggil judge API (skor absolut + pairwise
   dua-urutan), retry + validasi JSON, catat versi judge & hash prompt →
   `reports/experiments/llm_judge_scores.csv`.
3. `scripts/analyze_judge_results.py` — buka blinding, uji statistik, tabel
   ringkasan → `reports/experiments/llm_judge_summary.csv/.md`; hitung juga
   korelasi judge-vs-manusia dari sheet rubrik manual.
4. Estimasi biaya judge: ±30 blok × 4 kondisi (skor) + ±90 pasangan × 2 urutan
   (pairwise) ≈ 300-an panggilan pendek — kecil.

## 8. Ancaman Validitas yang Dikelola

- *Self-preference/style bias* → judge ≠ teacher, blind, pairwise dua arah.
- *Position bias* → dua urutan, hanya keputusan konsisten yang dihitung.
- *Verbosity bias* → rubrik menekankan ketepatan, bukan panjang; checker
  deterministik menjadi jangkar objektif.
- *Data leakage* → stat block evaluasi TIDAK boleh berasal dari contoh
  train/val dataset fine-tune (cek exact-match sebelum export).
- *Non-determinisme narrator* (temperature 0.4) → generate 1× per kondisi
  dengan seed Ollama dicatat, atau laporkan sebagai keterbatasan.
