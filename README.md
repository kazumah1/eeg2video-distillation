# eeg2video-distillation

Channel-reduction distillation for EEG visual decoding (Neurotech@Berkeley, Project A).
A teacher trained on the full EEG cap (63 channels, THINGS-EEG) teaches a student that
only sees a consumer-headset montage (Neurosity Crown, 8 channels). We measure how much
of the teacher's zero-shot image retrieval the student keeps, with and without distillation.

## Layout

| Path | What it is |
| --- | --- |
| `scripts/download_things_eeg.py` | Pulls preprocessed THINGS-EEG + CLIP RN50 image features for chosen subjects from NeuroBridge's Hugging Face copy |
| `scripts/train_teacher_neurobridge.sh` | Trains a NeuroBridge teacher with the official code (pinned commit) and links it as `runs/<name>` |
| `train_student.py` | Trains one student: montage x condition x seed; appends a row to the results CSV |
| `src/eegdistill/montages.py` | Channel sets: `all`, `posterior17`, `nested32`, `nested16`, `crown8` (nested) |
| `src/eegdistill/losses.py` | Contrastive task loss; response (embedding cosine + KL on retrieval scores), feature, relational distillation |
| `src/eegdistill/evaluate.py` | 200-way retrieval, chance-normalized top-1, % of teacher retained, student-teacher fidelity |
| `scripts/sweep.py` | Experiment grid (channels, losses, teachers, progressive, single_trial); resumable; `summary` prints mean ± std over seeds and paired gain vs the no-KD baseline |
| `tests/smoke_test.py` | CPU end-to-end test on synthetic data, including the full sweep |

## Conditions (`--condition`)

| Condition | Student loss |
| --- | --- |
| `none` | Contrastive only (the lambda = 0 baseline) |
| `response` | + (1 - cos) to the teacher's EEG embedding + KL to the teacher's softened retrieval scores (tau = `--kd_tau`) |
| `feature` | + (1 - cos) between first-layer hidden features (after spatial mixing) |
| `relational` | + MSE between trial-by-trial similarity matrices |
| `emb` / `kl` | Each half of `response` alone |
| `all` | All of the above |

Other options: `--init_from_teacher 1` (weight inheritance: the student starts from the teacher's
weights for its shared channels, all later layers and both projectors), `--train_average 0
--test_average 0` (single trials), montage `occipital8` (reference: best 8 sites for vision).

Born-again control: save an 8-channel `none` student with `--save_as`, then pass it as `--teacher`.

Design choices to know:
- NeuroBridge with `--image_aug` trains only on features of augmented images (and tests on them
  with `--image_test_aug`); its image projector never sees clean features. Students therefore
  train on a random augmented copy per trial and are evaluated on the mean of the augmented
  copies (`--eval_img augmean`). Using clean features with a NeuroBridge teacher's projector
  roughly halves the teacher's own top-1 (63% -> 34.5% on subject 8).
- Train EEG gets NeuroBridge's one-time random smoothing (`--eeg_smooth 1`), as the teacher did.
- Teacher and student see the same trial; each selects its own channels from the full 63.
- `--img_proj teacher` (default) copies and freezes the teacher's image projector in every
  condition, so all students share one embedding space and differ only in the distillation term.
- Results are final-epoch numbers. NeuroBridge reports the best epoch by test loss, which
  selects on the test set; expect our teacher numbers from `train_student.py`-style evaluation
  to be slightly below its "best" column.

## Colab runner cells

```python
# 1. GPU + code
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone -q -b student-pipeline https://github.com/kazumah1/eeg2video-distillation /content/edist || git -C /content/edist pull -q
%cd /content/edist
!pip -q install -r requirements.txt
```
```python
# 2. Data (one subject, ~3-5 GB, into the runtime, not Drive)
!python scripts/download_things_eeg.py --subjects 8 --out data
```
```python
# 3. Results go to Drive so they survive the runtime
from google.colab import drive; drive.mount('/content/drive')
RESULTS = '/content/drive/MyDrive/eegdistillation/results.csv'
```
```python
# 4. Teachers (official NeuroBridge code)
# a) published 17-channel config: reproduction check against the paper
!bash scripts/train_teacher_neurobridge.sh nb_s08_post17 8 posterior17
# b) all 63 channels: the teacher students distill from (Crown sites are not all posterior)
!bash scripts/train_teacher_neurobridge.sh nb_s08_all 8 all
```
```python
# 5. Students
!python train_student.py --teacher runs/nb_s08_all --montage crown8 --condition none --results_csv {RESULTS}
!python train_student.py --teacher runs/nb_s08_all --montage crown8 --condition response --results_csv {RESULTS}
```

## Local test

```bash
pip install torch numpy pandas
python tests/smoke_test.py
```
