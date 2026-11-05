# Shared Course Data Directory

This directory stores demonstration datasets, video clips, and annotations shared across all IF5152 laboratory modules.

### Sample Media Clips:
1. `person_bicycle.mp4`: Outdoor surveillance video (people walking, riding a bicycle, cars driving).
2. `kinetics_segway.mp4`: Official Kinetics-400 test set video asset (YouTube ID: `SOX5yA1l24A`, Ground Truth Action: `using segway`).

Large binary media files (`*.mp4`, `*.avi`, etc.) are excluded from Git commits via `.gitignore`.
To download the sample clips automatically, run:
```bash
python video-understanding-lab/download_samples.py
```
