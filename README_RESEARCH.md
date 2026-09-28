# Research Pipeline 2D Gaussian Splatting

Repository ini adalah fork 2D Gaussian Splatting (2DGS). Branch `main`
dipertahankan sebagai baseline, sedangkan seluruh pengembangan penelitian berada
di branch `thesis-development`. Laptop digunakan untuk pengembangan code;
training dan evaluasi aktual dijalankan di PC-server dengan GPU NVIDIA.

Pipeline penelitian merupakan wrapper/orchestrator di luar core 2DGS. Ia
memanggil `train.py`, `render.py`, dan `metrics.py` original melalui subprocess
tanpa mengganti implementasi algoritma tersebut. Submodule 2DGS hanya perlu
diinisialisasi dan dependency hanya perlu diinstal pada PC-server.

## Struktur

```text
research_pipeline/          Orchestrator, logger, monitor GPU, evaluator
configs/                    Konfigurasi global dan daftar scene per dataset
research_results/           CSV ringkasan eksperimen
docs/research_notes.md      Catatan metodologi dan eksperimen
requirements-research.txt   Dependency tambahan pipeline
```

Setiap scene langsung menghasilkan satu row di
`research_results/experiment_results.csv`, termasuk ketika scene gagal. Batch
melanjutkan scene berikutnya setelah kegagalan.

Kegagalan proses scene (training, rendering, metrics, atau evaluasi geometri)
dicatat sebagai `failed` dan tidak menghentikan scene berikutnya. Sebaliknya,
jika master `experiment_results.csv` tidak dapat dibuat atau ditulis, pipeline
melaporkan error `CRITICAL` beserta hasil scene yang gagal disimpan dan
menghentikan batch. Melanjutkan batch tanpa pencatatan hasil yang reliable tidak
diizinkan.

## Penggunaan

Semua command dijalankan dari root repository.

Dry-run tanpa dataset, CUDA, GPU, atau submodule:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --dry-run
```

Menjalankan seluruh batch:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml
```

Melewati scene yang sudah memiliki status `success`:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --resume
```

Menjalankan ulang semua scene tanpa mempertimbangkan hasil sebelumnya:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --force
```

`--force` meminta setiap scene dijalankan ulang meskipun sudah memiliki hasil
`success`. Flag ini tidak membersihkan, menghapus, atau me-reset output/model
directory lama; pengelolaan output lama tetap dilakukan secara manual.

Menjalankan satu scene:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --scene togog_ganesha
```

`--resume` dan `--force` tidak dapat digunakan bersamaan.

## Metrik

Metrik geometri:

- Chamfer Distance
- Precision
- Recall
- F1-score

Metrik citra dari `metrics.py` original:

- SSIM
- PSNR
- LPIPS

Metrik efisiensi yang hanya mencakup proses `train.py`:

- training duration
- peak VRAM
- average VRAM

Precision, Recall, dan F1 membutuhkan `distance_threshold`. Nilai default tetap
`null`; pipeline tidak menebak atau melakukan hardcode threshold. Dengan nilai
tersebut, Chamfer Distance tetap dihitung sedangkan Precision, Recall, dan F1
dikosongkan disertai warning.

Evaluator menggunakan Chamfer Distance berupa jumlah dua directional mean
nearest-neighbor Euclidean distance: `mean(R ke G) + mean(G ke R)`. Jarak tidak
dikuadratkan. Precision dan Recall menggunakan strict threshold `distance <
distance_threshold`, sehingga titik yang jaraknya tepat sama dengan threshold
tidak dihitung sebagai match. Mesh disampling menjadi point set menggunakan
`trimesh`; nearest-neighbor dihitung dengan `scipy.spatial.cKDTree`. Jumlah
sampel dikendalikan oleh `geometry_evaluation.sample_points` (default `100000`)
dan dicatat pada kolom CSV `geometry_sample_points`.

Evaluator tidak menjalankan ICP, alignment otomatis, normalisasi skala, atau
koreksi orientasi. Reconstruction dan ground truth diasumsikan sudah memiliki
coordinate system, scale, orientation, dan alignment yang sesuai. Penentuan
alignment dan distance threshold harus ditetapkan sebagai bagian metodologi
penelitian sebelum hasil Precision/Recall/F1 dilaporkan.
