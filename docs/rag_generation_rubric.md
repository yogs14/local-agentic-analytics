# Rubrik Penilaian Manual Jawaban RAG (Faithfulness & Answer Relevance)

Penilaian generasi jawaban RAG/hybrid finance_news dilakukan **manual oleh dua
penilai** (judge LLM lokal 2B tidak reliabel). Skala 1–5, mengikuti definisi
RAGAS untuk *faithfulness* dan *answer relevance*, dioperasionalkan agar dua
penilai konsisten.

## Prosedur

1. Jalankan `python scripts/export_rag_answers_for_rating.py` — menghasilkan
   `reports/experiments/rag_rating_sheet.csv` berisi query, konteks yang
   diretrieve, dan jawaban model, plus kolom skor kosong untuk dua penilai.
2. Kedua penilai mengisi kolom skornya **secara independen** (jangan saling
   melihat) berdasarkan rubrik di bawah. Kolom `notes_*` opsional.
3. Setelah keduanya selesai, hitung agreement (persentase selisih ≤ 1 poin dan
   korelasi) dan rata-ratakan skor per item. Selisih ≥ 2 poin didiskusikan dan
   dinilai ulang bersama (catat sebagai adjudikasi).

## Faithfulness (kesetiaan pada konteks) — skala 1–5

Pertanyaan panduan: *apakah setiap klaim dalam jawaban didukung oleh konteks
(headline) yang diretrieve?*

| Skor | Kriteria |
|---|---|
| 5 | Semua klaim faktual dalam jawaban didukung eksplisit oleh konteks; tidak ada tambahan informasi luar. |
| 4 | Hampir semua klaim didukung; ada ≤ 1 detail kecil yang tidak ada di konteks tapi tidak menyesatkan. |
| 3 | Inti jawaban didukung konteks, tetapi ada beberapa detail tanpa dukungan (tanggal, angka, sebab-akibat yang tidak disebut konteks). |
| 2 | Sebagian besar klaim tidak dapat dilacak ke konteks; konteks hanya menyentuh topiknya. |
| 1 | Jawaban berhalusinasi: klaim utama bertentangan dengan konteks atau sepenuhnya tanpa dasar. |

Catatan operasional:
- Kalimat pembuka/penutup generik ("Berdasarkan berita yang tersedia...")
  tidak dihitung sebagai klaim.
- Jika jawaban menyatakan "tidak ada informasi" dan konteks memang tidak
  relevan, itu **faithful** (nilai 5).

## Answer Relevance (relevansi terhadap pertanyaan) — skala 1–5

Pertanyaan panduan: *apakah jawaban benar-benar menjawab pertanyaan yang
diajukan (bukan sekadar merangkum konteks)?*

| Skor | Kriteria |
|---|---|
| 5 | Menjawab langsung dan lengkap; tidak ada bagian pertanyaan yang terabaikan; tanpa informasi tak relevan yang dominan. |
| 4 | Menjawab langsung tapi kurang lengkap pada aspek minor, atau ada sedikit isi tak relevan. |
| 3 | Menjawab sebagian; aspek penting pertanyaan terlewat, atau setengah jawaban berisi hal di luar pertanyaan. |
| 2 | Hanya menyinggung topik; tidak menjawab maksud utama pertanyaan. |
| 1 | Tidak menjawab pertanyaan sama sekali / salah topik. |

Catatan operasional:
- Relevance dinilai TERLEPAS dari benar-salahnya isi (itu urusan
  faithfulness). Jawaban salah tapi tepat sasaran pertanyaan bisa relevance 5.
- Jawaban "tidak ada informasi" pada pertanyaan yang sebenarnya terjawab oleh
  konteks = relevance rendah (≤ 2).

## Pelaporan

Laporkan per konfigurasi (mis. RAG murni vs hybrid): mean ± std kedua metrik,
n item, inter-rater agreement (≤1-poin agreement % dan korelasi Spearman),
dan jumlah item adjudikasi. Skor mentah kedua penilai disimpan apa adanya di
CSV (jangan ditimpa saat adjudikasi — tambah kolom `adjudicated_*`).
