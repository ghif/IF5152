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

# Public demonstration clips (standard benchmark test snippets)
SAMPLES = {
    # Outdoor surveillance clip featuring people, bicycles, and moving cars
    "data/person_bicycle.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
}


def main():
    os.makedirs("data", exist_ok=True)
    print("Checking and fetching demonstration video clips into 'data/'...")

    for path, url in SAMPLES.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [ALREADY PRESENT] {path} ({os.path.getsize(path) / 1024:.1f} KB)")
            continue
        print(f"  [DOWNLOADING] {url} -> {path}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [DOWNLOAD OK] Saved {path} ({os.path.getsize(path) / 1024:.1f} KB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {url}: {e}")

    print("\nDataset preparation complete. All clips ready for lecture demos.")


if __name__ == "__main__":
    main()
