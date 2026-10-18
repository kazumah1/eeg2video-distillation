"""Download preprocessed THINGS-EEG + image features from NeuroBridge's Hugging Face copy.

Only what we need: info.json, the chosen subjects' train/test.npy, and the RN50
image features (plus the fused augmentation features used by NeuroBridge).

    python scripts/download_things_eeg.py --subjects 8 --out data
    python scripts/download_things_eeg.py --list          # show what the repo holds

Files land under <out>/things_eeg/... mirroring the repo layout.
"""
import argparse
import fnmatch

from huggingface_hub import HfApi, hf_hub_download

REPO = "Feronl/NeuroBridge"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=[8])
    p.add_argument("--out", default="data")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no_aug_features", action="store_true")
    a = p.parse_args()

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    eeg = [f for f in files if f.startswith("things_eeg/")]
    if a.list:
        for f in eeg:
            print(f)
        return

    want = ["things_eeg/preprocessed_eeg/info.json", "things_eeg/image_feature/RN50/image_*.npy"]
    for s in a.subjects:
        want.append(f"things_eeg/preprocessed_eeg/sub-{s:02d}/*")
    if not a.no_aug_features:
        want.append("things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic/*")
    picked = sorted({f for f in eeg for pat in want if fnmatch.fnmatch(f, pat)})
    if not picked:
        raise SystemExit("nothing matched; run with --list to see the repo layout")
    for pat in want:
        if not any(fnmatch.fnmatch(f, pat) for f in picked):
            print(f"warning: nothing matched {pat}")
    for f in picked:
        print("downloading", f, flush=True)
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=a.out)
    print("done")


if __name__ == "__main__":
    main()
