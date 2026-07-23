# Resep Fine-tune Kaggle → Ollama (Terstandar)

> Dokumen template. Fine-tuning dilakukan **manual di Kaggle** (bukan oleh
> harness). Isi setiap kolom `<...>` saat gelombang fine-tune dijalankan, lalu
> salin nilai finalnya ke `configs/models/<model>_ft.yaml` supaya benchmark
> base-vs-finetuned tetap apple-to-apple.

Tujuan standar ini: dua model hasil fine-tune dari kandidat berbeda hanya boleh
berbeda pada **bobot dasarnya**. Dataset, konfigurasi LoRA, kuantisasi, dan
system prompt harus identik — kalau tidak, perbandingan antar-kandidat tidak
valid.

## 0. Identitas Run

| Field | Nilai |
|---|---|
| Model dasar (HF repo) | `<mis. Qwen/Qwen2.5-1.5B-Instruct>` |
| Target Ollama tag | `<mis. qwen2.5-sql-ft:v1>` |
| Notebook Kaggle (URL publik) | `<url>` |
| Tanggal run | `<YYYY-MM-DD>` |
| Commit repo saat dataset dibuat | `<git hash>` |

## 1. Dataset (identik untuk semua kandidat)

- Sumber: snapshot `data/finetune/` pada commit di atas.
- Hash snapshot: jalankan dari root repo dan catat hasilnya di sini.

  ```powershell
  ./.venv/Scripts/python.exe -c "from local_agentic_analytics.evaluation.model_benchmark import sha256_file; import pathlib; [print(p.name, sha256_file(p)) for p in sorted(pathlib.Path('data/finetune').glob('*.jsonl'))]"
  ```

| File | SHA256 |
|---|---|
| `<train.jsonl>` | `<sha256>` |
| `<val.jsonl>` | `<sha256>` |

- Format contoh (prompt/response), split train/val, dan jumlah baris:
  - train: `<n>` baris, val: `<n>` baris, split seed: `<seed>`.
- **Larangan:** tidak boleh ada pertanyaan gold eval (energy v2/v3, finance)
  yang bocor ke dataset fine-tune. Catat prosedur pengecekan kebocoran:
  `<mis. exact-match + fuzzy match pertanyaan>`.

## 2. Konfigurasi LoRA (identik untuk semua kandidat)

| Hyperparameter | Nilai |
|---|---|
| Framework | `<mis. unsloth / peft+trl, versi>` |
| LoRA rank (r) | `<mis. 16>` |
| LoRA alpha | `<mis. 32>` |
| LoRA dropout | `<mis. 0.05>` |
| Target modules | `<mis. q,k,v,o,gate,up,down proj>` |
| Epochs | `<mis. 2>` |
| Learning rate | `<mis. 2e-4>` |
| LR scheduler | `<mis. cosine>` |
| Batch size (efektif) | `<per_device x grad_accum>` |
| Max seq length | `2048` (samakan dengan context_window benchmark) |
| Seed | `<mis. 42>` |
| Precision saat training | `<mis. bf16/fp16 + 4-bit base>` |

Aturan: nilai tabel ini **dikunci sekali** lalu dipakai semua kandidat pada
gelombang yang sama. Jika perlu diubah, ulangi semua kandidat gelombang itu.

## 3. Export: Merge → GGUF → Q4_K_M

1. Merge adapter LoRA ke bobot dasar (fp16).
2. Konversi ke GGUF: `<perintah llama.cpp convert_hf_to_gguf.py + versi>`.
3. Quantize ke **Q4_K_M** (samakan dengan kandidat base yang dibandingkan):
   `<perintah llama-quantize>`.
4. Catat hash artefak:

| Artefak | SHA256 |
|---|---|
| `<model>-ft-Q4_K_M.gguf` | `<sha256>` |

## 4. Modelfile Ollama (system prompt identik baseline)

System prompt dan parameter di Modelfile harus identik dengan perilaku
baseline: prompt penuh dibangun oleh workflow (prompt template "default"),
sehingga Modelfile **tidak boleh menambahkan SYSTEM prompt lain**.

```text
FROM ./<model>-ft-Q4_K_M.gguf
# Tanpa SYSTEM tambahan: seluruh prompt datang dari workflow (apple-to-apple).
# Tanpa PARAMETER temperature/num_ctx: keduanya dikirim per-request oleh
# OllamaTool dari configs/model.yaml (temperature 0.0, num_ctx 2048).
```

Lalu:

```powershell
ollama create <target-tag> -f Modelfile
```

- SHA256 Modelfile (untuk `modelfile_sha256` di registry): `<sha256>`
  (hitung: `./.venv/Scripts/python.exe -c "from local_agentic_analytics.evaluation.model_benchmark import sha256_file; print(sha256_file('Modelfile'))"`)

## 5. Registrasi ke Harness Benchmark

Buat `configs/models/<model>_ft.yaml`:

```yaml
model:
  key: <model>_ft
  ollama_tag: "<target-tag>"
  param_count: "<sama dengan base>"
  quantization: "Q4_K_M"
  context_window: 2048
  temperature: 0.0
  prompt_template: default
  modelfile_sha256: "<sha256 dari langkah 4>"
  source: finetuned
  kaggle_notebook_url: "<url notebook>"
```

Validasi + jalankan:

```powershell
./.venv/Scripts/python.exe scripts/run_model_benchmark.py --models <model>_ft --suite sql_gold_v2 --dry-run
./.venv/Scripts/python.exe scripts/run_model_benchmark.py --models <model>_ft --suite sql_gold_v2 --repeats 3
```

## 6. Checklist Validitas

- [ ] Dataset snapshot di-hash dan identik antar kandidat gelombang ini.
- [ ] Tabel LoRA identik antar kandidat gelombang ini.
- [ ] Kuantisasi Q4_K_M untuk semua artefak fine-tune.
- [ ] Modelfile tanpa SYSTEM/parameter tambahan; hash dicatat.
- [ ] Registry YAML terisi lengkap (`source: finetuned`, `modelfile_sha256`).
- [ ] Tidak ada kebocoran soal gold eval ke data training.
- [ ] Notebook Kaggle publik dan URL-nya tercatat di registry + dokumen ini.
