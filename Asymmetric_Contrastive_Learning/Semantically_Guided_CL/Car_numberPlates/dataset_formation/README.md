# CCPD Plate-Swapping Dataset Formation

This package builds a manipulated CCPD dataset where each image keeps the original scene but receives a license plate extracted from another randomly assigned CCPD image. It uses only CCPD filename annotations: no RPNet, SAM, or detector.

The generated images are saved at their original resolution. Any fixed-size resizing happens later in training.

## Output Layout

The output root is separate and configurable:

```text
output_root/
  plate_mapping.json
  ccpd_base/
    real/
    manipulated/
    mask/
  ccpd_blur/
    real/
    manipulated/
    mask/
```

`real/` contains full copied originals. `manipulated/` contains plate-swapped images. `mask/` contains binary PNG masks for only the destination quadrilateral.

## Generate

Small local test:

```bash
python -m dataset_formation.generate \
  --dataset-root "C:\Users\tusha\ChineseCarParkingDataset2019" \
  --output-root "C:\Users\tusha\ChineseCarParkingDataset2019_swapped_test" \
  --limit 200 \
  --workers 4
```

HPC-scale run:

```bash
python -m dataset_formation.generate \
  --dataset-root /path/to/ChineseCarParkingDataset2019 \
  --output-root /path/to/ChineseCarParkingDataset2019_plate_swapped \
  --workers 64 \
  --seed 1337
```

By default, all configured CCPD subsets are processed. Use `--limit` only for quick tests. Existing completed samples are skipped unless `--overwrite` is passed.

## Visualize

```bash
python -m dataset_formation.visualize \
  --dataset-root /path/to/ChineseCarParkingDataset2019 \
  --generated-root /path/to/ChineseCarParkingDataset2019_plate_swapped \
  --output-dir /path/to/visual_checks \
  --num-samples 32
```

Each panel stacks original, extracted source plate, mask, and manipulated image.

## Notes

- The deterministic assignment is saved in `plate_mapping.json`.
- The mapping is a derangement, so no image maps to itself.
- Plate extraction uses the four annotated vertices and a perspective warp to a tight frontal rectangle.
- Plate insertion resizes the extracted plate, warps it into the destination quadrilateral, and blends it with OpenCV `seamlessClone`.
- Multiprocessing uses CPU worker processes and is suitable for HPC nodes.
