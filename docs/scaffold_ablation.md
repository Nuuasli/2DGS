# Ablasi Inisialisasi Anchor Scaffold-GS untuk 2DGS

## Metodologi

Eksperimen ini bernama `scaffold_anchor`, dengan deskripsi **2DGS with
Scaffold-GS anchor initialization**. Eksperimen ini bukan Scaffold-GS murni.
Scaffold-GS hanya menghasilkan posisi anchor untuk point cloud inisialisasi;
neural feature, MLP, opacity/covariance network, dan view-adaptive rendering
Scaffold tidak diteruskan ke 2DGS.

```text
COLMAP yang sudah disiapkan
  -> training Scaffold-GS
  -> ekstraksi field x/y/z anchor dari point_cloud.ply
  -> RGB dari sparse COLMAP terdekat dalam Euclidean XYZ
  -> initial_points.ply
  -> training 2DGS dengan --init_ply
  -> render dan mesh extraction 2DGS
  -> metrics/evaluasi pipeline yang sama dengan baseline
  -> CSV eksperimen terpisah
```

Kamera, train/test split, scene normalization, renderer, loss, densification,
depth distortion, normal consistency, dan mesh extraction tetap berasal dari
2DGS. Tidak ada ICP, koreksi orientasi, atau normalisasi skala tersembunyi.

Source resmi Scaffold-GS menyimpan tensor `_anchor` sebagai field `x`, `y`, dan
`z` pada `point_cloud/iteration_<N>/point_cloud.ply`. Field offset dan anchor
feature tidak digunakan. `initial_points.ply` berisi `x y z nx ny nz red green
blue`; normal diisi nol dan RGB diambil dari titik sparse COLMAP terdekat.

## Environment

Pipeline utama, konversi anchor, 2DGS, render, dan evaluasi dijalankan dalam
environment `2dgs`. Training Scaffold dijalankan sebagai subprocess dengan
Python dari environment terpisah, misalnya `scaffold_gs_cu118`.

Pipeline tidak menjalankan `conda activate`. Lokasi executable dipilih dari
environment variable `SCAFFOLD_PYTHON`; jika tidak ada, pipeline menggunakan
`scaffold.python_executable` dari experiment config. Jika keduanya kosong,
eksperimen berhenti dengan error konfigurasi yang jelas.

```bash
export SCAFFOLD_PYTHON=/home/wisnu/miniconda3/envs/scaffold_gs_cu118/bin/python
```

## Menjalankan

Dry-run satu scene (tidak menjalankan subprocess training):

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --experiment-config configs/experiment_30000_scaffold_anchor.yaml \
  --scene ganesha \
  --dry-run
```

Full run dengan reuse hasil yang telah tervalidasi:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --experiment-config configs/experiment_30000_scaffold_anchor.yaml \
  --resume
```

Jika COLMAP belum siap:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --experiment-config configs/experiment_30000_scaffold_anchor.yaml \
  --prepare \
  --resume
```

`--resume` melewati seluruh scene yang sudah sukses pada CSV ablasi. Untuk
scene yang belum sukses, Scaffold dan `initial_points.ply` digunakan ulang
hanya bila manifest, konfigurasi, commit, hash, dan artefak checkpoint cocok.
Folder output saja tidak dianggap sebagai bukti sukses.

`--force` meminta eksperimen scene dijalankan ulang, tetapi tidak menghapus
output Scaffold maupun 2DGS. Hasil Scaffold yang masih valid tetap dapat
digunakan ulang sesuai `scaffold.reuse_successful`; cleanup selalu manual.

## Output

Baseline tetap terpisah:

```text
output/togog_bali/
research_results/2dgs_30000_visual_results.csv
```

Ablasi menggunakan:

```text
output/ablations/scaffold_anchor/
  scaffold/<dataset>/<scene>/
    point_cloud/iteration_<N>/point_cloud.ply
    scaffold_stage.json
    initial_points.ply
    initialization.json
  2dgs/<dataset>/<scene>/
research_results/2dgs_scaffold_anchor_results.csv
```

`training_seconds`, `peak_vram_mb`, dan `average_vram_mb` tetap mengukur hanya
training 2DGS. Field `scaffold_training_seconds`, `scaffold_peak_vram_mb`, dan
`scaffold_average_vram_mb` berhenti pada marker resmi `Training complete.` di
`outputs.log`. Hal ini diperlukan karena entry point resmi Scaffold-GS
melanjutkan render dan metrics setelah fungsi training selesai; hasil
post-training Scaffold tersebut tidak dipakai oleh ablasi. Nilai
`total_ablation_seconds` adalah `scaffold_training_seconds + training_seconds`;
preprocessing COLMAP, render, metrics, dan evaluasi tidak dimasukkan.

Kegagalan scene dicatat dengan stage (`scaffold_training`,
`scaffold_initialization`, `2dgs_training`, `render`, `metrics`, atau
`geometry_evaluation`) dan batch melanjutkan scene berikutnya. Kegagalan master
CSV tetap menghentikan batch karena hasil berikutnya tidak dapat dicatat secara
reliable.
