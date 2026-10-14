# egopred: ego trajectory prediction from front-camera video

Predict where the car will be over the next 4 seconds from 2 seconds of front-camera video, the
car's own recent motion and a high-level navigation command (left / straight / right). The main
question is **how much the camera adds on top of kinematics, and in which situations**. Every
model is therefore reported next to a kinematics-only twin and a physics baseline, broken down
by scenario.

Main dataset: [NAVSIM](https://github.com/autonomousvision/navsim) (`navtrain` / `navtest`),
built on [OpenScene](https://github.com/OpenDriveLab/OpenScene) / [nuPlan](https://www.nuscenes.org/nuplan).

<p align="center"><img src="assets/turn_right.gif" width="100%"></p>
<p align="center"><img src="assets/braking.gif" width="100%"></p>
<p align="center"><img src="assets/curve.gif" width="100%"></p>
<p align="center"><img src="assets/random_window.gif" width="100%"></p>

<sub>navtest, consecutive samples at 2 Hz from a single drive.
<b>Green</b>: ground truth. <b>Blue</b>: model A (kinematics + command). <b>Orange</b>: model C (vision + kinematics + command).
Left: front camera with the 4 s trajectories projected through the real calibration and lens distortion. Right: the same trajectories in a 3D view over a ground grid.
The first three windows were selected as the ones where C beats A by the largest margin: a right turn (mean ADE 1.47 → 0.42 m), braking behind a bus (1.30 → 0.44 m) and a 12 s curvy stretch (1.23 → 0.64 m).
The last window is a random draw (0.87 → 0.52 m). Random windows are not always this favourable: in 4 out of 6 random windows A has the lower mean ADE, because on regular driving kinematics are enough.</sub>

## Results at a glance

On navtest (12,146 samples, drives never seen in training), errors are in metres over a 4 s horizon, scored on the most likely trajectory:

| model | inputs | ADE | FDE | miss rate (FDE > 2 m) | lateral err. | longitudinal err. |
|---|---|---|---|---|---|---|
| B0: constant speed and yaw rate | current dynamics | 2.41 | 5.97 | — | — | — |
| **A** | kinematics + command | 0.783 | 2.04 | 39.5 % | 0.31 | 0.62 |
| **B** | vision + speed + command | 1.194 | 2.61 | 50.9 % | 0.34 | 1.05 |
| **C** | vision + kinematics + command | 0.719 | 1.68 | 30.0 % | 0.23 | 0.62 |
| **C**, final recipe | + visual-input augmentation, aux. loss | **0.693** | **1.66** | **29.8 %** | **0.23** | **0.59** |

- **Vision helps:** C is 11 % better than A in ADE, 19 % in FDE and 25 % in miss rate. The ranking is the same on the validation split.
- **The gain is lateral, i.e. road geometry.** The lateral error drops by 26 %, while the longitudinal error (when to brake or accelerate) barely moves.
- **The gain grows with the horizon.** At 1 s C is slightly worse than A (0.17 vs 0.14 m); at 4 s it is clearly better (1.66 vs 2.04 m).
- **The camera anticipates braking but does not help once it has started.** At the onset of strong braking C cuts the FDE by 37 %; during braking that is already under way it is 3 % worse.

## Problem setup

| | |
|---|---|
| Input | front camera (CAM_F0) at t0 − 1.5, −1.0, −0.5, 0 s; ego poses over the same 2 s; current velocity, acceleration and yaw rate; NAVSIM driving command |
| Output | 8 future waypoints (x, y, yaw) at 0.5 s steps, 4 s horizon, in the ego frame at t0 |
| Metrics | ADE / FDE of the top-1 trajectory, miss rate (FDE > 2 m), lateral / longitudinal error relative to the true heading, minADE for multimodal models, errors per horizon |
| Stratification | scenario tags derived from the ground-truth future (turn left / right, curve, lateral shift, brake, accelerate, start, cruise), heading-change bands, braking phase, speed bands |

The evaluation reports open-loop displacement metrics only, not the closed-loop NAVSIM PDM score: the question here is prediction quality, not driving.

## Data pipeline

All data handling runs on a CPU machine with a ~75 GB data disk, against source archives several times that size, so everything is shard-by-shard, resumable and tracked in a SQLite manifest (`pending → done → verified`).

1. **Metadata.** The OpenScene pickles (poses at 2 Hz, dynamic state, driving command, camera calibration) are streamed and converted to one Parquet file per log.
2. **Frames.** OpenScene camera archives hold all 8 cameras. They are streamed (`curl | tar`) and only CAM_F0 is kept, for the frames NAVSIM samples actually use (current and history). Frames are re-encoded at 960×540 (JPEG q85) into tar shards with an offset index for random access. Hundreds of GB are streamed and ~14 GB kept.
3. **Sample tables.** For every NAVSIM token the pipeline builds:
   - ego-frame history and future;
   - the kinematic inputs;
   - QC flags (timestamp gaps, pose jumps);
   - scenario tags;
   - a deterministic sample ID.

   Splits are by drive, never by frame:

   | split | samples | drives |
   |---|---|---|
   | navtrain train | 91,924 (91,020 after QC) | 141 |
   | navtrain val | 11,364 | 21 |
   | navtest (official) | 12,146 | 44 |

4. **Sanity gates.** Before any training the pipeline checks that projected ground-truth trajectories line up with the road, that video and pose are in sync (visual-odometry yaw rate vs pose yaw rate), that the baselines behave as expected, and that the model can overfit a small subset.

## Visual features

The backbone is **frozen** and run once offline: training never decodes an image.

- **Backbone:** DINOv2 ViT-B/14 with registers (`vit_base_patch14_reg4_dinov2.lvd142m`, via timm), input 518×294.
- **Per frame:**
  - the CLS token (768-d);
  - the patch grid, average-pooled to 6×10 cells and projected to 256-d with a PCA fitted on 4,000 training frames.
- **Size:** ~5 GB in float16 for ~170k frames, small enough to keep the whole training set resident in GPU memory. A full 30-epoch run takes 8–15 minutes on an RTX 3090.

## Models

All variants share one architecture and differ only in which tokens they see:

```
kinematics (history poses, v, a, yaw rate, command) ──► MLP ──► 1 token ┐
CLS of 4 frames            ──► linear + time embedding ──► 4 tokens    ├─► Transformer encoder (4 layers, d=256, 8 heads)
6×10 grid of 4 frames      ──► linear + time + cell embedding ──► 240 tokens ┘                 │
                                                                                               ▼
                                         K learned queries ──► Transformer decoder (2 layers) ──► K × (trajectory, logit)
```

| variant | kinematics | vision |
|---|---|---|
| B0 | constant velocity / constant speed and yaw rate (no learning) | — |
| A | full history + current dynamics + command | — |
| B | current speed + command only | CLS + grid |
| C | full history + current dynamics + command | CLS + grid |

**Multimodal head with anchors**
- Each of the K queries predicts an offset from a fixed anchor trajectory, the k-means centres of the training futures.
- The query whose anchor is closest to the ground truth wins: it gets the regression loss, and its logit is trained with cross-entropy.
- Because the winner depends on the anchors and not on the predictions, mode assignment is stable. Plain winner-takes-all left the mode logits at chance level.
- With K=6 the best of the six modes reaches minADE ≈ 0.70 m. For top-1 accuracy, K=1 is better, because ADE rewards the conditional mean.

**Training**
- AdamW, lr 1e-4 with warmup and cosine decay, weight decay 0.01, dropout 0.1, batch 256, 30 epochs, mixed precision.
- The checkpoint with the best top-1 ADE on val is kept.

**Final recipe (C, K=1)**
- **Train-time augmentation of the visual inputs:** Gaussian noise in feature-std units, and dropped past frames, grid cells or the whole vision input. Dropped tokens are masked out of the attention instead of being filled.
- **A training-only auxiliary head** that predicts how the features of the frames at t0 + 1 s and t0 + 2 s change.
- Together they are worth about 3–4 % ADE on navtest.

## Detailed results

FDE (m) on navtest by situation, A vs the final C:

| situation | n | B0 | A | C (final) | C vs A |
|---|---|---|---|---|---|
| all | 12,146 | 5.97 | 2.04 | 1.66 | −19 % |
| curve | 1,789 | 5.34 | 2.48 | 1.77 | −29 % |
| turn left | 1,473 | 4.85 | 2.31 | 1.68 | −27 % |
| turn right | 842 | 6.35 | 2.59 | 1.95 | −25 % |
| lateral shift | 474 | 4.70 | 2.13 | 1.74 | −18 % |
| straight (heading change < 5°) | 6,399 | 6.50 | 1.76 | 1.57 | −11 % |
| sharp turn (≥ 45°) | 1,517 | 5.83 | 2.48 | 1.82 | −27 % |
| braking under way (a < −1 m/s²) | 1,135 | 10.01 | 1.27 | 1.31 | +3 % |
| braking onset | 561 | 6.48 | 3.52 | 2.53 | −28 % |
| strong braking onset (Δv > 3 m/s) | 223 | 7.90 | 4.66 | 2.96 | −37 % |
| start / acceleration onset | 637 | 6.51 | 2.70 | 1.95 | −28 % |

L2 error of the top-1 trajectory by horizon (m), navtest:

| model | 1 s | 2 s | 3 s | 4 s |
|---|---|---|---|---|
| A | **0.14** | **0.45** | 1.08 | 2.04 |
| C | 0.19 | 0.48 | 0.98 | 1.68 |
| C (final) | 0.17 | **0.45** | **0.95** | **1.66** |

Validation split (11,364 samples): A 0.749 / 1.94, C 0.681 / 1.60, final C 0.661 / 1.59 (ADE / FDE).

**What did not help**
- **Vision without the kinematic history (B)** is the weakest learned model, mostly longitudinally (1.05 m error).
- **CLS-only vs CLS + grid** are within noise once the inputs are augmented.
- **Curating away trivially predictable samples** (dropping 80% of those where B0 is already under 1 m) made both A and C slightly worse.
- **Stronger regularisation** (dropout 0.2, weight decay 0.05) made C worse.
- **Predicting future features as an auxiliary task** added at most 1 % on top of augmentation, within seed noise, and the head itself explains only ~10 % of the feature change.

The consistent picture is that the remaining error, mostly longitudinal, is limited by what a single frozen front-camera embedding contains, not by the amount of supervision.

## Limitations

- Front camera only: no map, no other cameras, no detected agents.
- The backbone is frozen and features are pooled to 6×10 cells, so small or distant cues (traffic lights, a braking car far ahead) are largely lost.
- Open-loop metrics only.
- Most comparisons use a single seed. Repeated seeds on B and C moved ADE by less than 0.01 m, well below the A–C gap.
- NAVSIM speeds stay below ~17 m/s, so nothing here says anything about highway driving.

## Repository layout

```
egopred/
  openscene/       metadata, frame shards, sample tables (targets, kinematics, QC, scenario tags)
  stream.py        streaming download/extraction with retries
  manifest.py      SQLite manifest of pipeline items
  trajectory.py    ego-frame windows, scenario tagging, B0 baselines, displacement metrics
  geometry.py      SE(3), projection with lens distortion
  sync.py, g0.py   video/pose synchronisation check and projection overlays
  g1.py            B0 baselines by scenario
  features.py      frozen DINOv2 feature extraction + PCA
  model.py         transformer model, anchors, auxiliary future head
  train.py         training, evaluation, export of predictions
  figures.py, viz.py  figures and GIFs from exported predictions
  comma2k19.py     loader for comma2k19 segments (development dataset)
scripts/           ingest and GPU-session helpers
tests/             pytest suite
```

## Usage

Everything runs in a Docker image (CPU) except feature extraction and training, which need a GPU.

```bash
make image                                   # build the dev image
make test                                    # test suite

# data (CPU)
make run ARGS="meta --split trainval" && make run ARGS="meta --split test"
make ingest                                  # front-camera frames for navtrain + navtest
make run ARGS="samples --split trainval --name navtrain --filter navtrain"
make run ARGS="samples --split test --name navtest --filter navtest"
make run ARGS="g1 navtrain navtest"          # B0 baselines

# GPU
python -m egopred features --device cuda
python -m egopred train --name A --variant A --anchors --lr 1e-4 --k 1 --device cuda
python -m egopred train --name C --variant C --anchors --lr 1e-4 --k 1 \
    --aug-noise 0.5 --aug-frame-drop 0.5 --aug-token-drop 0.5 --aug-vision-drop 0.1 \
    --future 1,2 --device cuda
python -m egopred compare A C --split test
python -m egopred export A C --device cuda   # predictions for figures, no GPU needed afterwards
python -m egopred gif --a A --c C --pick random
```

## Data and licences

Camera frames shown in this README come from the [nuPlan](https://www.nuscenes.org/nuplan) dataset (Motional) as distributed by [OpenScene](https://github.com/OpenDriveLab/OpenScene). They are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the nuPlan Dataset License Agreement for Non-Commercial Use. The GIFs in `assets/` are derived from those frames (resized, predicted trajectories drawn on top) and are released under the same CC BY-NC-SA 4.0 terms. No dataset content is included in this repository beyond these figures.

## References

- Dauner et al., *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*, NeurIPS 2024.
- OpenScene Contributors, *OpenScene: The Largest Up-to-Date 3D Occupancy Prediction Benchmark in Autonomous Driving*, 2023.
- Caesar et al., *nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles*, 2021.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024; Darcet et al., *Vision Transformers Need Registers*, ICLR 2024.
