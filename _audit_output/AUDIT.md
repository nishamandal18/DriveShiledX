# DriveShieldX — Read-Only Forensic Evidence Audit

**Audit date:** 2026-08-21
**Scope:** read-only. No project files were created, edited, deleted, moved, or renamed. No training was run, no packages installed, no server launched. The only write is this folder, `_audit_output/`.
**Project root:** `C:\Users\helim\Downloads\DriveShieldX\DriveShieldX\DriveShieldX` (contains `dashboard/`, `detection/`, `database/`, etc.)
**Deployed model weights:** `models/` and `backend/models/` — plus an identical copy set at `C:\Users\helim\Downloads\DriveShieldX\models\` and loose copies in `C:\Users\helim\Downloads\`.
**Environment used for inspection:** the project venv `\.venv` (torch 2.13.0+**cpu**, ultralytics 8.4.121).

> **Headline:** No Ultralytics training output (`results.csv`, `args.yaml`, `runs/`, PR/confusion figures) exists anywhere in the project. No **YOLO detection** training/validation dataset (helmet, seat-belt, two-wheeler, plate) is present on disk — every model's `data:` field points at a `/kaggle/working/...` path that is **not** in this repository or on `C:\`. The only genuine, measured accuracy numbers that survive are the **validation metrics embedded inside the `.pt` checkpoints**, and these are **aggregate (all-class) box metrics measured on Kaggle at training time**, not the per-class numbers the paper quotes. The runtime database's "number plates" are **synthetically generated demo strings**, not OCR output. **UA-DETRAC tracking ground truth DOES exist on disk** at `C:\datasets` (100 sequences of frame-level `<target id>` boxes + the official CLEAR-MOT toolkit) — but DriveShieldX has never been run on or scored against it, none of the sequences is the "MVI_007" the app names, and there is **no evaluation/metrics code in the codebase** (no MOTA/IDF1/HOTA/CER/precision-recall computation) linking the two.

> **⚠️ Correction vs first pass:** an initial project-scoped search reported "no tracking ground truth of any kind." That was wrong — the UA-DETRAC GT lives **outside** the project tree at `C:\datasets` (pointed out by the user). Section D and the tracking rows below are the corrected version.

---

## 1. RESULTS ALREADY GENUINELY MEASURED

### 1a. Detection metrics embedded in the trained checkpoints
Source: the `train_args` / `train_metrics` block stored inside each `.pt` by Ultralytics. These are **validation metrics at the saved epoch, averaged over all classes** (the Ultralytics `(B)` = bounding-box metrics). They are genuine measurements, but were produced on Kaggle and the val split is not on disk (see caveats below).

| Metric | Value | Exact source file | How it was produced |
|---|---|---|---|
| helmet — precision(B), all classes | 0.85222 | `models/helmet_yolov8.pt` (`train_metrics`) | Ultralytics val at train time, `data=/kaggle/working/merged_helmet/data.yaml`, base `yolov8s.pt`, date 2026-08-17 |
| helmet — recall(B), all classes | 0.81213 | same | same |
| helmet — **mAP50(B), all classes** | **0.8515** | same | same |
| helmet — mAP50-95(B), all classes | 0.37361 | same | same |
| seat-belt — precision(B) | 0.98056 | `models/seatbelt_yolov8.pt` | val at train time, `data=/kaggle/working/merged_seatbelt/data.yaml`, base `yolov8s.pt`, 2026-08-18 |
| seat-belt — recall(B) | 0.97117 | same | same |
| seat-belt — mAP50(B) | 0.98374 | same | same |
| seat-belt — mAP50-95(B) | 0.64877 | same | same |
| two-wheeler (helmet/triple) — precision(B) | 0.83706 | `models/twowheeler_best.pt` | val at train time, `data=/kaggle/working/tw_merged/data.yaml`, base `yolov8n.pt`, 2026-06-02. Classes: `with_helmet, without_helmet, triple_riding` |
| two-wheeler — recall(B) | 0.82242 | same | same |
| two-wheeler — mAP50(B) | 0.86756 | same | same |
| two-wheeler — mAP50-95(B) | 0.5612 | same | same |
| plate (deployed) — precision(B) | 0.983 | `models/plate_best_v2.pt` | val at train time, `data=/kaggle/working/merged/data.yaml`, base `yolov8n.pt`, 2026-06-02. Class: `License_Plate` |
| plate (deployed) — recall(B) | 0.93624 | same | same |
| plate (deployed) — **mAP50(B)** | **0.96609** | same | same |
| plate (deployed) — **mAP50-95(B)** | **0.67202** | same | same |
| plate (alt, **NOT deployed**) — precision(B) | 0.95008 | `C:\Users\helim\Downloads\best.pt` | `data=/kaggle/working/License-Plate-Recognition-4/data.yaml`, 100 epochs, 2026-05-29 |
| plate (alt) — recall(B) | 0.89267 | same | same |
| plate (alt) — mAP50(B) | 0.9399 | same | same |
| plate (alt) — mAP50-95(B) | 0.46342 | same | same |

**Caveats on 1a (critical):**
- These are **all-class averages**, not per-class. Per-class precision/recall (e.g. `no_helmet` recall) are **not** stored in the checkpoint and cannot be recovered without the dataset or a preserved `results.csv`.
- The val split that produced them (`/kaggle/working/merged_helmet/`, `/merged_seatbelt/`, `/tw_merged/`, `/merged/`) is **absent from disk**, so these numbers cannot be re-verified locally.
- `yolov8n.pt` and `models/yolo11n.pt` are stock COCO-pretrained backbones (data `coco.yaml`), not project results.

### 1b. Operational counts measured by THIS repository (runtime SQLite DB)
Source: `database/overspeed.db` (opened read-only). This reflects the app's own runs on the uploaded videos + seeded demo data.

| Metric | Value | Source | How produced |
|---|---|---|---|
| RULE_VIOLATION rows by type | three_seater = 36, no_helmet = 4, no_seatbelt = 2 (total 42) | `RULE_VIOLATION` | live pipeline runs; all status = `detected` |
| VIOLATION (speed) rows by severity | low = 29, medium = 20, high = 11, extreme = 18 (total 78) | `VIOLATION` | speed pipeline |
| SPEED_RECORD count / min / max / mean | 21726 / 0.0 / **781.68** / 12.06 km/h | `SPEED_RECORD` | centroid/ByteTrack + pixel-to-metre estimate, `source_fps` constant 25.0 |
| Speed outliers in a 60 km/h zone | >120: **133**, >150: 74, >200: 29, >300: 15, >400: 5 (>60: 402) | `SPEED_RECORD` | physically impossible; pixel-jump artefacts |
| Only persisted processing rate | **last_fps = 0.088 FPS** (11 326 ms/frame) | `STREAM_STATUS` | last live run, **CPU** torch. (`last_frame_no = 30` — the "30" is a frame index, not FPS.) |
| Row counts (other) | VEHICLE 1108, VIDEO_SESSION 61, VIOLATION_NOTICE 55, ALERT_OUTBOX 368, SYSTEM_EVENT 277 | DB | — |

---

## 2. WHERE EACH RESULT CAME FROM (provenance chains)

- **Helmet 0.8515 mAP50 / 0.852 P / 0.812 R:** `models/helmet_yolov8.pt` → `torch.load(...weights_only=False)` → `ckpt['train_args']['data'] = /kaggle/working/merged_helmet/data.yaml`, `ckpt['train_metrics'] = {precision(B)=0.85222, recall(B)=0.81213, mAP50(B)=0.8515, mAP50-95(B)=0.37361}`. Trained from `yolov8s.pt`, ultralytics 8.4.121, 2026-08-17. Dataset not on disk.
- **Seat-belt 0.984 mAP50:** `models/seatbelt_yolov8.pt` → `train_metrics` as above; `data=/kaggle/working/merged_seatbelt/data.yaml`; base `yolov8s.pt`; 2026-08-18.
- **Two-wheeler / triple-riding 0.868 mAP50:** `models/twowheeler_best.pt` → `train_metrics`; `data=/kaggle/working/tw_merged/data.yaml`; classes include `triple_riding`; base `yolov8n.pt`; 2026-06-02.
- **Plate 0.966 mAP50 / 0.672 mAP50-95:** `models/plate_best_v2.pt` (the copy actually loaded by the app, identical md5 across `models/`, `backend/models/`, and Downloads) → `train_metrics`; `data=/kaggle/working/merged/data.yaml`; base `yolov8n.pt`; 2026-06-02.
- **Operational counts / speeds / plate yield:** `database/overspeed.db`, tables `RULE_VIOLATION`, `VIOLATION`, `SPEED_RECORD`, `VEHICLE`, `STREAM_STATUS`. Produced by `detection/advanced_pipeline.py` during live-monitor runs on the uploaded videos, plus demo seeding (`smoke_test.py`, `harness_render.py`, `utils/demo_runner.py`).
- **Processing rate 0.088 FPS:** computed live in `detection/advanced_pipeline.py` (`self.last_fps` from `time.time()` deltas, lines ~1498-1506) and written to `STREAM_STATUS.last_fps`. Only the most recent value is persisted; the log `logs/overspeedx.log` records no FPS at all.

---

## 3. VALUES THAT ARE ONLY MODEL CONFIDENCES — NOT ACCURACY

> ⚠️ **Warning:** Every value in this section is a raw per-detection score. **None** is precision, recall, F1, mAP, accuracy, MOTA, IDF1 or HOTA. Do not convert or present any of them as an accuracy figure.

- **YOLO detection confidences printed by the sanity scripts:** `test_seatbelt_person_backup.py:91` `score = float(b.conf[0])` (rounded and printed); likewise the other `test_seatbelt*.py` / `test_vehicle*.py` scripts print `box.conf` values (this is where a value such as `0.776` would come from). These are confidences for a single box on a single frame.
- **`VEHICLE.ocr_confidence` in the DB:** min 0.0, max ~0.00061, mean ~1.1e-6; exactly **1** of 1108 rows is > 0. These are placeholder confidences attached to fabricated demo plates (see §7 verdict), not OCR accuracy.
- Any confidence overlaid on the files in `snapshots/` is likewise a detection confidence.

---

## 4. EXPERIMENTS NOT YET RUN

| Experiment | Status | Note |
|---|---|---|
| Helmet P/R/F1 (per class) | **PARTIALLY DONE** | Aggregate P/R exist in checkpoint (0.852 / 0.812). Per-class P/R/F1 NOT available; no dataset, no `results.csv`. |
| Triple-riding P/R/F1 | **PARTIALLY DONE** | Only the aggregate two-wheeler (3-class) metrics exist (0.837 / 0.822 / mAP50 0.868). Per-class triple-riding P/R/F1 NOT isolated. |
| Seat-belt P/R/F1 | **PARTIALLY DONE** | Aggregate P/R exist (0.981 / 0.971). Per-class + F1 not stored. |
| MOTA | **NOT RUN (now feasible)** | GT + official CLEAR-MOT toolkit present at `C:\datasets`, but DriveShieldX has never been run on a DETRAC sequence nor scored; no tracker output in the toolkit's `trackers/` dir (only DETRAC's published baselines). Needs: run pipeline on chosen sequences → export MOT-format results → run toolkit (MATLAB). |
| IDF1 | **NOT RUN** | The UA-DETRAC toolkit computes CLEAR MOT (MOTA/MOTP/PR-MOTA), **not** IDF1. Needs py-motmetrics or TrackEval (not present). |
| HOTA | **NOT RUN** | Same — HOTA needs TrackEval (not present). |
| IDSW (id switches) | **NOT RUN (now feasible)** | Reported by the CLEAR-MOT toolkit; same run-and-score gap as MOTA. |
| DeepSORT comparison | **NOT RUN** | DeepSORT is not implemented anywhere in the code. |
| ByteTrack comparison | **NOT RUN** | ByteTrack runs live (ultralytics built-in), but no tracking metric was ever computed on GT. |
| Multi-seed mean ± s.d. | **NOT RUN** | Single checkpoint per model; no seed sweep; no stats code. |
| Ablation | **NOT RUN** | No ablation harness. |
| ANPR CER (character error rate) | **NOT RUN** | No CER/Levenshtein code; no plate ground-truth strings. |
| Speed MAE | **NOT RUN** | No speed ground truth; speeds contain impossible outliers up to 781 km/h. |

---

## 5. DATA AND ANNOTATIONS REQUIRED (for each NOT-RUN / PARTIAL item)

- **Per-class helmet / seat-belt / triple-riding P/R/F1 and mAP:** the actual validation datasets. Their YAMLs are referenced but absent: `/kaggle/working/merged_helmet/data.yaml`, `/kaggle/working/merged_seatbelt/data.yaml`, `/kaggle/working/tw_merged/data.yaml`, `/kaggle/working/merged/data.yaml`. Recover the Kaggle datasets (with `images/` + YOLO `labels/`) **or** the original `runs/.../results.csv` from each training run. Then run `yolo val` per model to get per-class rows. (The `roboflow:` provenance block the paper's dataset section needs also lives only in those absent YAMLs.)
- **MOTA / IDSW:** the GT now **exists** — UA-DETRAC XML for 100 sequences at `C:\datasets\DETRAC-{Train,Test}-Annotations-XML\` plus frames at `C:\datasets\DETRAC-Images\` and the official CLEAR-MOT toolkit at `C:\datasets\DETRAC-MOT-toolkit\`. What is missing: (a) DriveShieldX tracker output on chosen sequences, exported to the toolkit's expected `trackers/<name>/<seq>.txt` MOT format; (b) a MATLAB environment to run `DETRAC_experiment.m` / `DETRAC_MOT_EVAL.exe`. No glue code links DriveShieldX to the toolkit today.
- **IDF1 / HOTA:** the UA-DETRAC toolkit does **not** produce these. Additionally required: `py-motmetrics` (IDF1) and/or `TrackEval` (HOTA), neither installed nor referenced, fed the same GT converted to MOTChallenge `gt.txt` (`frame,id,x,y,w,h,conf,cls,vis`).
- **DeepSORT vs ByteTrack:** a DeepSORT implementation (not present) plus the run-and-score pipeline above for both trackers.
- **Multi-seed mean ± s.d. / ablation:** re-training with fixed seeds and a results-aggregation script; neither the datasets nor such scripts exist.
- **ANPR CER + end-to-end accuracy:** a labelled benchmark set of plate crops/images with true plate strings, and a CER (Levenshtein) scorer. The claimed "300-image benchmark" is **not on disk** (the only "300" in code is `limit=300` on DB queries).
- **Speed MAE:** ground-truth speeds (e.g., radar or calibrated GT) for tracked vehicles; none exists, and calibration (`pixel_to_meter_scale = 0.045`, `source_fps = 25`) is uncalibrated for the uploaded clips.

---

## 6. EXACT NEXT EXPERIMENTS (ordered by paper-value ÷ effort)

1. **Recover per-class detection metrics (highest value, lowest effort).** Re-download the four Kaggle datasets to local `datasets/…`, then, per model:
   ```bash
   yolo val model=models/helmet_yolov8.pt data=datasets/merged_helmet/data.yaml split=val
   yolo val model=models/seatbelt_yolov8.pt data=datasets/merged_seatbelt/data.yaml split=val
   yolo val model=models/twowheeler_best.pt data=datasets/tw_merged/data.yaml split=val
   yolo val model=models/plate_best_v2.pt data=datasets/merged/data.yaml split=val
   ```
   Gives per-class P/R/F1, mAP50, mAP50-95, and a real confusion matrix / PR curve. **~30–60 min** once datasets are downloaded (CPU is slow; a GPU box makes it minutes). This alone settles paper claims 1–4.
2. **Confirm the helmet val split counts (727 / 1436 / 833 / 603).** After download, run a label-file counter over `merged_helmet/{train,valid,test}/labels/`. **~5 min.**
3. **Honest ANPR benchmark.** Build a small labelled plate set (even 100–300 real crops with true strings), run `detection/npr.py` (EasyOCR) over it, compute exact-match and CER with a 10-line scorer. Report as OCR accuracy — **not** the DB's demo plates. **~half a day incl. labelling.**
4. **Tracking metrics — MOTA/IDSW (now feasible, GT already on disk).** Pick a handful of UA-DETRAC sequences at `C:\datasets\DETRAC-Images\` (each ~900 frames; e.g. MVI_20011/20012/20032), run DriveShieldX's ByteTrack pipeline on them, and export per-sequence results to the toolkit's `trackers/DriveShieldX/<seq>.txt` MOT format. Then score with `C:\datasets\DETRAC-MOT-toolkit\` (`DETRAC_experiment.m` / `DETRAC_MOT_EVAL.exe`, needs MATLAB — a `_temp_matlab_R2025b_Windows` folder already sits in Downloads). Gives MOTA/MOTP/PR-MOTA + IDSW. **~1–2 days** (export glue + MATLAB run). For **IDF1/HOTA** add `py-motmetrics`/`TrackEval` over the same GT converted to `gt.txt` (**+1 day**). For the **DeepSORT vs ByteTrack** comparison you must also integrate DeepSORT (**+1–2 days**). Note: none of these sequences is "MVI_007", so the paper must not present results as coming from that named camera.
5. **Processing-rate benchmark.** Run a fixed clip end-to-end on the target hardware (ideally GPU) and record `last_fps` over N frames; report mean ± s.d. and state the hardware. **~1 hr.** Note the only persisted value today is 0.088 FPS on CPU.

---

## VERDICT ON THE PAPER'S EXISTING CLAIMS

| # | Paper claim | On-disk evidence | Verdict |
|---|---|---|---|
| 1 | **helmet mAP@50 = 0.904** | Only measured value for this model is `helmet_yolov8.pt` `train_metrics` mAP50(B) = **0.8515** (all-class). 0.904 appears nowhere. | **NOT SUPPORTED / CONTRADICTED.** Nearest (and only) evidence is 0.8515. If 0.904 came from a different run, that run's `results.csv` is not preserved. |
| 2 | **helmet recall = 0.934 / no_helmet recall = 0.779** | Checkpoint stores only aggregate recall(B) = **0.81213**; no per-class values. (Mean of 0.934/0.779 = 0.857 ≠ 0.812, so these did not come from the stored checkpoint.) | **SILENT / NOT VERIFIABLE.** Per-class recall is unrecoverable without the dataset or a preserved `results.csv`. |
| 3 | **val split = 727 images, 1,436 instances (833 / 603)** | `merged_helmet` dataset is absent (`/kaggle/working/merged_helmet/`). No labels on disk to count. | **SILENT / NOT VERIFIABLE.** Cannot confirm or deny — dataset must be recovered. |
| 4 | **plate mAP@50 = 0.968, mAP@50:95 = 0.935** | `plate_best_v2.pt` (deployed) `train_metrics`: mAP50(B) = **0.96609**, mAP50-95(B) = **0.67202**. (Alt `best.pt`: 0.9399 / 0.46342.) | **mAP@50 ≈ SUPPORTED** (0.966 vs 0.968 — off by 0.002). **mAP@50:95 CONTRADICTED** (measured 0.672, not 0.935). |
| 5 | **end-to-end ANPR = 49.7% on a 300-image benchmark** | No 300-image benchmark exists. DB plate yield = 549/1108 = 49.5%, but **548/549 plates are synthetic** `_demo_plate_for(tracker_id)` strings (`advanced_pipeline.py:170`), `ocr_confidence ≈ 0`. | **NOT SUPPORTED / CONTRADICTED.** The "49.7%" has no benchmark behind it; the DB "plates" are fabricated demo data, so no genuine ANPR accuracy exists. |
| 6 | **processing rate ≈ 30 FPS** | Only persisted FPS = `STREAM_STATUS.last_fps` = **0.088 FPS** (CPU, 11.3 s/frame). Log records no FPS. (The "30" in the row is `last_frame_no`, a frame index.) | **NOT SUPPORTED / CONTRADICTED** by the single persisted measurement. FPS is hardware-dependent; no GPU benchmark was saved. |

---

## Appendix A — Section-by-section evidence log

**A. Training evidence:** No `results.csv`, `args.yaml`, `opt.yaml`, `hyp.yaml`, or `runs/` directory anywhere except inside `.venv` (library defaults). No `confusion_matrix.png` / `results.png` / `PR_curve.png` / `R_curve.png` / `labels.jpg` in the project. → **All training-run artefacts are missing; only the `.pt` checkpoints survive.**

**B. Checkpoint-embedded config:** recovered for all 11 `.pt` files (see §1a/§2). Deployed models are `yolov8s` (helmet, seatbelt) and `yolov8n` (twowheeler, plate). Every project model's `data:` is a `/kaggle/working/...` path. `train_metrics` present for all trained models; `best_fitness` was stripped (`None`) on the renamed weights except the plate `best.pt`.

**C. Dataset provenance / size / split:** **No dataset YAML and no `images/`+`labels/` tree exists on disk** for any model. Counting images/instances is therefore impossible locally. The helmet 727/1436/833/603 claim cannot be checked (see verdict 3).

**D. Tracking ground truth — decisive answer (corrected):** **Yes, frame-level GT exists — but it is not connected to DriveShieldX and none of it is the camera the app names.** UA-DETRAC annotations are present at `C:\datasets` (outside the project tree, from the predecessor **OverSpeedX** project — DriveShieldX is a continuation and the DETRAC data has not yet been wired in): `DETRAC-Train-Annotations-XML` (60 sequences) + `DETRAC-Test-Annotations-XML` (40 sequences), each XML holding per-frame `<target id="…">` boxes with `vehicle_type`/`speed` attributes (true frame-level track identities); `DETRAC-Images` (100 sequences, ~900 frames each); and `DETRAC-MOT-toolkit` (the official CLEAR-MOT MATLAB toolkit — `CLEAR_MOT.m`, `DETRAC_MOT_EVAL.exe` — whose `trackers/` dir ships only DETRAC's published baselines: CEM, CMOT, DCT, GOG, H2T, …). **Caveats that keep this from being a finished result:** (1) **no sequence is "MVI_007"** — DETRAC names are `MVI_20011…MVI_40xxx`; the string "UA-DETRAC Highway MVI_007" is a **hardcoded seed** in `database/db_manager.py:102-105`, cosmetic only; (2) **DriveShieldX has never been run on any DETRAC sequence** (its DB video sessions are the uploaded triple-riding/seat-belt clips + temp files), and there is **no DriveShieldX result in the toolkit's `trackers/` dir**; (3) **no project code references `C:\datasets` or the toolkit** (`utils/dataset_setup.py` is only a DETRAC *download/prepare* helper); (4) the toolkit yields **MOTA/MOTP/PR-MOTA/IDSW but not IDF1 or HOTA**. → **MOTA/IDSW are computable-in-principle from data already on disk, but have not been measured; IDF1/HOTA additionally need tooling that is absent. As of now, zero tracking metrics have been produced, so none can yet appear in the paper.**

**E. Tracker & evaluation code:** ByteTrack = **Ultralytics built-in** via `model.track(tracker="bytetrack.yaml")` in `detection/multi_tracker.py:105` (not `supervision`, not custom). Fallback `CentroidTracker` in `detection/centroid_tracker.py`. The dashboard `Tracker mode` dropdown (`dashboard/app.py:973,1234`) feeds `advanced_pipeline.py` → `multi_tracker.py`. **Absent from the entire codebase:** DeepSORT, deep_sort, motmetrics, TrackEval, MOTA, IDF1, HOTA, id_switch/IDSW, ablation, wilcoxon, ttest, scipy(.stats), levenshtein, exact_match, CER, and any precision/recall/mAP computation.

**F. Videos (live-monitor inputs, user-confirmed):**
| file | frames | fps | resolution | duration |
|---|---|---|---|---|
| triple_riding_video_1.mp4 | 805 | 25.0 | 640×480 | 32.2 s |
| 15397236_1920_1080_60fps.mp4 | 1139 | 59.94 | 1920×1080 | 19.0 s |
| seat_belt_video_1.mp4 | 5121 | 59.94 | 1280×720 | 85.4 s |
| seat_belt_video_2.mp4 | 383 | 25.0 | 1280×720 | 15.3 s |
| seatbelt.mp4 | 240 | 24.0 | 1280×720 | 10.0 s |
| seatbelt det.mp4 | 240 | 24.0 | 1280×720 | 10.0 s |

None carries a sidecar annotation file. `SPEED_RECORD.source_fps` is hard-coded 25.0 regardless of the true fps above — a source of speed error.

**G. Runtime database:** see §1b. ANPR yield VEHICLE 549/1108 (of which 548 synthetic) and RULE_VIOLATION 4/42. Speed max 781.68 km/h; 133 records >120 km/h in a 60 km/h zone. VIDEO_SESSION sources are mostly one-off temp uploads (macOS `/var/folders/…` and Windows `…\Temp\…`), plus `triple_riding_video_1.mp4`; earliest log line references a macOS path `/Users/pankajmandal/Desktop/overspeed_detection 2/…`, indicating the DB predates this machine and mixes seeded demo data with real runs.

**H. Test scripts & logs:** `test_seatbelt*.py`, `test_vehicle*.py`, `visualize_seatbelt.py` = ad-hoc detection sanity scripts that print boxes / raw `conf` (no metric saved). `test_escalation.py`, `tests/test_driveshieldx.py` = functional/business-logic tests (challan amounts 500/1000/3000, session lifecycle, PNG magic, HTTP 200 — pass/fail, no numeric result). `smoke_test.py`, `harness_render.py`, `utils/demo_runner.py` = demo-data seeders (the last fabricates random speeds). No `results/`, `output/`, `debug/`, `eval/`, or `benchmark/` directory exists; the only `.txt` under `logs/` are 368 alert-outbox email/SMS notices; `logs/test_streamlit_boot.log` is a boot log. **No test script produces a saved detection/accuracy number.**
