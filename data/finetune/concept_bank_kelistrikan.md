# Concept Bank — Domain Kelistrikan & Statistika

Untuk fine-tuning Insight Agent proyek `local-agentic-analytics`.

**Aturan emas penggunaan:** Sebuah istilah hanya boleh muncul di narasi jika datanya **benar-benar mendukung**. Datasetmu (Individual Household Electric Power Consumption) punya: daya aktif (kW), daya reaktif (kVAR), tegangan (V), intensitas arus (A), tiga sub-metering (Wh), dan timestamp. Jangan pakai jargon yang butuh data yang tidak kamu punya (mis. data per-fasa, frekuensi, topologi grid).

Tiap entri di bawah punya tag:
- ✅ **AMAN** — datamu mendukung langsung
- ⚠️ **HATI-HATI** — boleh, tapi hanya jika nilainya dihitung dari input, jangan diklaim sebagai benchmark
- 🚫 **HINDARI** — datamu tidak mendukung, berisiko halusinasi

---

## A. Kosakata Statistika

| Istilah | Definisi singkat | Kapan dipakai | Tag |
|---|---|---|---|
| Rata-rata (mean) | Nilai pusat aritmetik | Selalu, jika ada di input | ✅ |
| Median | Nilai tengah terurut | Jika input menyediakan | ✅ |
| Simpangan baku (SD) | Sebaran data dari mean | Jika input menyediakan SD | ✅ |
| Koefisien variasi (CV) | SD ÷ mean, ukuran sebaran relatif | Boleh dihitung dari SD & mean di input (CV = SD/mean) | ⚠️ |
| IQR (rentang interkuartil) | Q3 − Q1, sebaran 50% data tengah | Hanya jika Q1/Q3 ada di input | ⚠️ |
| Skewness (kemiringan) | Asimetri distribusi | Hanya jika input menyebut, atau sebut kualitatif ("distribusi cenderung simetris") tanpa angka karangan | ⚠️ |
| Distribusi | Pola sebaran nilai | Untuk grafik histogram/distribusi | ✅ |
| Modus (mode bin) | Nilai/bin paling sering | Jika input menyebut mode bin | ✅ |
| Outlier | Nilai ekstrem di luar pola | Jika input menyebut, atau dari min/maks yang menyimpang jauh | ⚠️ |
| Korelasi (Pearson r) | Kekuatan hubungan linear dua variabel | Hanya jika input menyediakan nilai r (mis. heatmap) | ✅ |
| Tren | Arah perubahan terhadap waktu | Untuk grafik time-series | ✅ |
| Rentang (range) | Maks − min | Dari min & maks di input | ✅ |
| Rata-rata bergerak | Penghalusan deret waktu | 🚫 kecuali kamu benar-benar menghitungnya | 🚫 |
| Uji hipotesis / p-value / ANOVA / Jarque-Bera | Uji signifikansi statistik | 🚫 datamu tidak menjalankan uji ini — ini sumber halusinasi gaya report gempa | 🚫 |

---

## B. Kosakata Kelistrikan — Beban & Konsumsi

| Istilah | Definisi singkat | Kapan dipakai | Tag |
|---|---|---|---|
| Daya aktif (real power, kW) | Daya yang melakukan kerja nyata | Variabel utama datamu (Global_active_power) | ✅ |
| Daya reaktif (kVAR) | Daya bolak-balik tak melakukan kerja | Ada di data (Global_reactive_power) | ✅ |
| Energi (kWh) | Daya × waktu | Untuk total konsumsi (SUM/60 dari data per menit) | ✅ |
| Tegangan (V) | Beda potensial listrik | Ada di data (Voltage) | ✅ |
| Intensitas arus (A) | Arus listrik | Ada di data (Global_intensity) | ✅ |
| Beban puncak (peak load) | Konsumsi maksimum pada periode | Dari nilai maks di input | ✅ |
| Base load (beban dasar) | Konsumsi minimum berkelanjutan | Dari nilai min di input | ✅ |
| Load factor | Rasio beban rata-rata ÷ beban puncak | Boleh dihitung dari mean & maks di input | ⚠️ |
| Kurva beban (load curve) | Profil konsumsi terhadap waktu | Untuk grafik pola per jam/harian | ✅ |
| Profil beban residensial | Pola konsumsi khas rumah tangga | Untuk interpretasi pola harian | ✅ |
| Disparitas puncak-lembah | Selisih konsumsi tertinggi & terendah | Dari maks & min di input | ✅ |
| Power factor (faktor daya) | Rasio daya aktif ÷ daya semu | ⚠️ datamu punya aktif & reaktif, jadi BISA dihitung, tapi hanya sebut jika benar dihitung | ⚠️ |
| Sub-metering | Pengukuran konsumsi per zona (dapur, cucian, pemanas/AC) | Untuk grafik perbandingan sub-metering | ✅ |
| Konsumsi spesifik per zona | Distribusi energi antar sub-meter | Dari data sub-metering | ✅ |

---

## C. Kosakata Kelistrikan — Manajemen & Implikasi

> ⚠️ Bagian ini paling rawan halusinasi. Boleh dipakai sebagai **interpretasi kualitatif** ("berpotensi", "mengindikasikan peluang"), TIDAK BOLEH sebagai klaim kuantitatif berbenchmark.

| Istilah | Definisi singkat | Cara aman | Tag |
|---|---|---|---|
| Demand response | Penyesuaian konsumsi terhadap pasokan | "berpotensi mendukung demand response" — kualitatif | ⚠️ |
| Load shifting | Menjadwal ulang beban ke jam permintaan rendah | "disparitas puncak-lembah membuka peluang load shifting" | ⚠️ |
| Peak shaving | Menurunkan beban puncak | Kualitatif saja | ⚠️ |
| Efisiensi energi | Optimalisasi penggunaan daya | Kualitatif | ⚠️ |
| Anomali konsumsi | Penyimpangan pola tak terduga | Hanya jika data menunjukkan penyimpangan nyata | ⚠️ |
| Stabilitas tegangan | Konsistensi level tegangan | Dari CV/SD tegangan yang rendah | ⚠️ |
| Benchmark PLN / standar IEEE / SPLN angka spesifik | Nilai acuan industri | 🚫 JANGAN sebut angka standar yang tidak ada di input — ini jebakan utama | 🚫 |
| Prediksi/forecast kuantitatif ("turun 15%") | Proyeksi masa depan berangka | 🚫 datamu deskriptif/diagnostik, bukan prediktif | 🚫 |

---

## D. Pemetaan ke 6 Grafik Energimu

Panduan jargon mana yang relevan per grafik (dari Tabel 4.5 tesismu):

1. **Tren daya aktif harian** → tren, beban puncak/base load, kurva beban, rentang, rata-rata
2. **Pola konsumsi per jam** → load curve, load factor, base load, profil residensial, disparitas puncak-lembah, load shifting (kualitatif)
3. **Distribusi daya** → distribusi, mean, SD, koefisien variasi, modus bin, outlier
4. **Distribusi tegangan** → distribusi, stabilitas tegangan, CV, SD, modus bin
5. **Heatmap korelasi** → korelasi Pearson r, hubungan antar-variabel (mis. daya aktif vs intensitas arus r=0,9989)
6. **Perbandingan sub-metering** → sub-metering, konsumsi spesifik per zona, distribusi energi antar zona

---

## E. Daftar Larangan Keras (anti-halusinasi)

Saat menyusun atau memverifikasi narasi, TOLAK jika mengandung:

1. Angka yang tidak ada di input dan tidak bisa dihitung darinya
2. Referensi standar/benchmark eksternal dengan angka (IEEE 519, SPLN, benchmark PLN "X%")
3. Hasil uji statistik yang tidak dijalankan (p-value, ANOVA, Jarque-Bera, Pearson jika r tak disediakan)
4. Prediksi kuantitatif masa depan ("likelihood 19%", "turun 15-20%")
5. Klaim perbandingan dengan "industri" atau "kompetitor" berangka
6. Jargon yang butuh data yang tidak kamu punya (per-fasa, frekuensi grid, topologi jaringan)

Interpretasi kualitatif ("mengindikasikan", "berpotensi", "konsisten dengan pola") tetap BOLEH — itu reasoning, bukan fabrikasi.
