"""
========================================================================================
IF5152: Video Understanding - Dataset Setup Helper
File: download_samples.py
========================================================================================

PEDAGOGICAL PURPOSE:
  - In a classroom environment, downloading large video datasets (like UCF101 at 6.5 GB
    or Kinetics-400 at 150 GB) is impractical and slows down live demonstrations.
  - This utility downloads small, lightweight public demonstration clips into `data/`
    so students can run experiments locally in seconds without bandwidth issues.
========================================================================================
"""
import os
import urllib.request
from utils import resolve_data_path

# Public demonstration clips (standard benchmark test snippets)
SAMPLES = {
    # Outdoor surveillance clip featuring people, bicycles, and moving cars
    "person_bicycle.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    # Official Kinetics-400 test video asset (Action: using segway, YouTube ID: SOX5yA1l24A)
    "kinetics_segway.mp4": "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/videos/SOX5yA1l24A.mp4",
}


def main():
    root_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(root_data_dir, exist_ok=True)
    print(f"Targeting shared repository data directory: {root_data_dir}")

    for filename, url in SAMPLES.items():
        target_path = resolve_data_path(filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"  [ALREADY PRESENT] {target_path} ({os.path.getsize(target_path) / 1024:.1f} KB)")
            continue
        print(f"  [DOWNLOADING] {url} -> {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  [DOWNLOAD OK] Saved {target_path} ({os.path.getsize(target_path) / 1024:.1f} KB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {url}: {e}")

    print("\nDataset preparation complete. All clips ready for lecture demos.")


if __name__ == "__main__":
    main()
