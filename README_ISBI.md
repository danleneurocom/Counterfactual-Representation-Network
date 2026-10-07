# TRACE-Seg3D — pipeline sạch cho ISBI 2027

Pipeline mới nằm trong `src/trace_seg3d/` và `scripts/isbi/`. Code cũ (`baselines/`, `src/causal_mednext/causal_model.py`) vẫn giữ nguyên để tham khảo, chỉ vá các bug tiền xử lý (xem *Thay đổi*). **Mọi số đưa vào paper ISBI nên lấy từ pipeline mới.**

## 0. Dataset

| Dataset | Dùng bản nào | Ghi chú |
|---|---|---|
| UTSW-Glioma | ảnh `brain_*_ants.nii.gz` (SRI24, 1 mm), nhãn `*manual_correction*` | Case chỉ có nhãn FeTS tự động bị đưa ra khỏi val/test (`--exclude-auto-labels`). Metadata TSV dùng làm proxy (scanner make, field strength, grade). |
| BraTS 2020 | **bản NIfTI chính thức** (`MICCAI_BraTS2020_TrainingData`, 369 case, có `name_mapping.csv`) | Cùng bộ case với bản h5 Kaggle, nhưng có spacing, orientation, và `name_mapping.csv` cho site (CBICA/TCIA0x/2013…) + grade (HGG/LGG). Nếu chỉ có h5 thì dùng `brats-h5` và **bắt buộc** khai báo thứ tự kênh, rồi kiểm tra bằng `check_data`. |

## 1. Tiền xử lý (chạy 1 lần, CPU)

```bash
export PYTHONPATH=.:src
python -m trace_seg3d.preprocess utsw --root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
    --metadata data/brats/UTSW_Glioma_Metadata-2-1.tsv --out data/processed/utsw_128 --workers 8
python -m trace_seg3d.preprocess brats-nifti --root data/brats/MICCAI_BraTS2020_TrainingData \
    --out data/processed/brats_128 --workers 8
# chỉ khi dùng h5 Kaggle:
# python -m trace_seg3d.preprocess brats-h5 --root .../content/data --h5-modality-order <thứ tự> --out data/processed/brats_128

python -m trace_seg3d.check_data data/processed/utsw_128 data/processed/brats_128 --png check.png
python -m trace_seg3d.splits --index data/processed/utsw_128/index.csv  --out splits/utsw.json --exclude-auto-labels
python -m trace_seg3d.splits --index data/processed/brats_128/index.csv --out splits/brats.json
```

**Bắt buộc mở `check.png` và đọc output của `check_data`:** trong vùng ET, T1CE phải sáng hơn hẳn T1 (dòng `T1CE > T1 + 0.5 inside ET … OK`; FLAIR sáng ngang T1CE trong ET là bình thường), và hai dataset phải cùng chiều khi nhìn cùng một view. Nếu sai thì sửa `--h5-modality-order` / `--flip-axes` rồi chạy lại. Commit `splits/*.json` lên git để cả nhóm dùng chung một split.

Các bước tiền xử lý: crop theo mask não tính trên **ảnh gốc** → pad thành khối lập phương (giữ tỉ lệ, cùng scale giữa hai dataset) → resize về 128³ (≈1.3–1.5 mm) → z-score trong não, background = 0. Spacing hiệu dụng được lưu lại để tính HD95 ra **mm**.

## 2. Chạy toàn bộ thí nghiệm (GPU)

```bash
DATA_UTSW=data/processed/utsw_128 DATA_BRATS=data/processed/brats_128 \
SPLIT_UTSW=splits/utsw.json SPLIT_BRATS=splits/brats.json \
EPOCHS=150 SEEDS="0 1 2" bash scripts/isbi/run_all.sh
```

Lệnh này chạy hết train → calib → eval → audit → bảng. Nếu bị dừng giữa chừng, chạy lại đúng lệnh đó: bước nào xong rồi sẽ được bỏ qua. Có thể chạy từng phần, ví dụ `STEPS="train calib"` hoặc `SOURCES=utsw`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MAIN_METHODS` | `baseline styleaug trace` | chạy với mọi seed trong `SEEDS` |
| `ABLATIONS` | `trace_nocct trace_nostab trace_noproxy` | chạy với `ABL_SEEDS` (mặc định 1 seed) |
| `EXTRA_TRAIN_ARGS` | – | ví dụ `--patch-size 96 --batch-size 2` (T4 16 GB: mode trace ở 112³×2 bị out of memory) |
| `AMP` | 1 | mixed precision |

Nên chạy thử trước 1 run ngắn (`EPOCHS=2 SOURCES=utsw MAIN_METHODS=trace SEEDS=0 ABLATIONS=""`) để đo thời gian/epoch và VRAM, rồi mới chọn `EPOCHS`. Mode `trace` decode 2 lần mỗi bước (factual + transported) nên chậm hơn baseline khoảng 1,5 lần.

Ma trận đầy đủ gồm 2 nguồn × (3 phương pháp × 3 seed + 3 ablation) = 24 run.

### Chạy trên Kaggle (2× T4, khuyên dùng)

`scripts/isbi/TRACE_ISBI_kaggle.ipynb` (hướng dẫn cài đặt ở cell đầu). Mỗi lần chỉ cần *Save Version*: notebook lấy code từ GitHub, tải trạng thái từ 2 Kaggle Dataset private (`trace-isbi-data`, `trace-isbi-runs`, qua `kaggle_sync.py`), chạy `kaggle_runner.py` (2 job song song, thứ tự A → B → C, eval sơ bộ sau seed 0 và eval lại khi đủ seed, dừng trước 12 giờ, train dở thì tiếp từ `resume.pt`), rồi lưu trạng thái và đẩy bảng/log lên branch `isbi-results`.

## 3. Kết quả → paper

| File | Dùng cho |
|---|---|
| `results/isbi/final/table.md / .tex` | Bảng chính: Dice/HD95 (mm) WT/TC/ET, ET Dice trên case có ET, mean ± std qua seed, p-value Wilcoxon so với baseline. ID = test của nguồn, OOD = test của dataset còn lại. |
| `results/isbi/raw_no_prior/` | Ablation structural prior (ngưỡng 0.5, không prior) — không cần train lại |
| `results/isbi/cct_consensus/` | Dự đoán bằng CCT consensus thay cho factual |
| `results/isbi/audit_<src>_to_<tgt>_<method>.md` | **Bảng audit** (claim chính): Spearman, AUROC phát hiện case hỏng, AURC cho `cct_u_mean`, `cct_disagree`, `entropy_mean`, `tta_std_mean`, `ens_std_mean` |
| `runs/isbi/<src>/<run>/eval_*/maps_*.npz` | ảnh, xác suất, instability map → `python -m trace_seg3d.figures <maps...> --out fig.png` |

Quy ước metric phải ghi trong paper: Dice = 1 khi cả dự đoán lẫn GT đều rỗng; HD95 = 373.13 mm khi chỉ một bên rỗng; region map dùng OR xác suất (Eq. 1) ở cả loss lẫn metric; ngưỡng và kích thước tối thiểu component ET được chọn trên **source val**; context bank lấy từ **source train** (lưu sẵn trong checkpoint).

## 4. Phương pháp trong code (để viết lại Method cho khớp)

* **M1 – factorization**: context `z_c` là mean/std theo kênh của feature encoder ở mọi tầng; disease `z_d` lấy từ phần content đã chuẩn hóa ở bottleneck. Proxy head: context ← scanner/site, disease ← log thể tích vùng + grade. Hai head GRL đối kháng cùng orthogonality loss chống rò rỉ giữa hai factor.
* **M2 – CCT**: giữ content cố định, thay moments bằng moments của một case source (AdaIN) ở mọi skip connection và bottleneck. Lúc train: seg loss trên nhánh transported + stability hinge (Eq. 19) tính trên vùng lesion. Lúc test: K = 4 context đa dạng (farthest-point) từ bank → consensus + instability map.
* **M3 – structural prior**: bỏ các component ET nhỏ hơn `min_et_voxels` (chọn trên val), sau đó ép ET ⊆ TC ⊆ WT.

## Thay đổi so với code cũ

* `baselines/segformer3d/data/utsw.py`, `baselines/segformer3d/evaluate_causal_brats_h5.py`: crop trên ảnh gốc rồi mới normalize (trước đây crop không có tác dụng); background giữ bằng 0.
* `baselines/segformer3d/train_causal_utsw.py::_weighted_total`: báo lỗi khi gặp loss term chưa có trọng số (trước đây tự gán 1.0) và bỏ qua term có trọng số 0 (tránh 0·NaN).
* Pipeline mới chặn các trường hợp: tune ngưỡng trên test/target, context bank từ target, báo số ID trên tập không phải test, dùng metadata (grade/tumor type) lúc inference.
* Không dùng cho paper: các nhánh `phenotype_gated_et_demotion`, `nonenhancing_core_completion`, `oracle_style_selector` trong `baselines/mednext/evaluate_causal_utsw.py` (dùng metadata hoặc GT).

## Test

```bash
PYTHONPATH=.:src python -m pytest -q tests/test_trace_seg3d.py
# chạy toàn bộ pipeline trên data giả lập (~1 giờ trên CPU 2 nhân); kết quả mẫu ở docs/isbi_smoke_results/:
bash scripts/isbi/smoke_synthetic.sh
```
