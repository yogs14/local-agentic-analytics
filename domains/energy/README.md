# Energy Domain Profile

Folder ini berisi konfigurasi domain untuk dataset `household_power_consumption`.

- `profile.yaml` mendeskripsikan metadata dataset, kolom, satuan, semantic type, dan aturan SQL domain energi.
- `report_config.yaml` menyimpan konfigurasi awal untuk laporan energi.

Profil ini disiapkan agar workflow berikutnya dapat membaca aturan domain dari konfigurasi, bukan dari hardcode Python.
