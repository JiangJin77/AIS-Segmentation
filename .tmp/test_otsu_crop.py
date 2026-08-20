from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk


def threshold_based_crop_and_bg_median(image):
    inside_value = 0
    outside_value = 255
    bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)

    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image, image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)

    return bounding_box, sitk.RegionOfInterest(
        image,
        bounding_box[int(len(bounding_box) / 2):],
        bounding_box[0:int(len(bounding_box) / 2)],
    ), bg_mean, bin_image


root = Path(r"E:\My_vscode_project\Dataset\data\MyAisDataset\images_after_skull")
records = []
failures = []

for path in sorted(root.glob("*.nii.gz"), key=lambda p: int(p.name.split(".")[0])):
    try:
        image = sitk.ReadImage(str(path))
        bbox, cropped, bg_median, mask = threshold_based_crop_and_bg_median(image)
        image_array = sitk.GetArrayViewFromImage(image)
        mask_array = sitk.GetArrayViewFromImage(mask)
        foreground_fraction = float(np.mean(mask_array == 255))
        background = image_array[mask_array == 0]
        records.append({
            "file": path.name,
            "image_size": list(image.GetSize()),
            "bbox": list(bbox),
            "cropped_size": list(cropped.GetSize()),
            "bg_median_filter": float(bg_median),
            "bg_median_numpy": float(np.median(background)),
            "foreground_fraction": foreground_fraction,
            "image_min": float(np.min(image_array)),
            "image_max": float(np.max(image_array)),
        })
    except Exception as exc:
        failures.append({"file": path.name, "error": repr(exc)})

def values(key, axis=None):
    if axis is None:
        return np.asarray([r[key] for r in records], dtype=float)
    return np.asarray([r[key][axis] for r in records], dtype=float)

summary = {
    "tested": len(records) + len(failures),
    "passed": len(records),
    "failed": len(failures),
    "failures": failures,
}

if records:
    summary.update({
        "bbox_size_xyz_min": [int(values("bbox", i).min()) for i in range(3, 6)],
        "bbox_size_xyz_median": [float(np.median(values("bbox", i))) for i in range(3, 6)],
        "bbox_size_xyz_max": [int(values("bbox", i).max()) for i in range(3, 6)],
        "foreground_fraction_min_median_max": [
            float(values("foreground_fraction").min()),
            float(np.median(values("foreground_fraction"))),
            float(values("foreground_fraction").max()),
        ],
        "background_median_min_median_max": [
            float(values("bg_median_filter").min()),
            float(np.median(values("bg_median_filter"))),
            float(values("bg_median_filter").max()),
        ],
        "median_disagreement_max": float(np.max(np.abs(
            values("bg_median_filter") - values("bg_median_numpy")
        ))),
        "widest_foreground_cases": sorted(
            records, key=lambda r: r["foreground_fraction"], reverse=True
        )[:5],
        "smallest_foreground_cases": sorted(
            records, key=lambda r: r["foreground_fraction"]
        )[:5],
    })

print(json.dumps(summary, ensure_ascii=False, indent=2))
