"""去颅骨CT：3D脑区裁剪、归一化、XY等比例缩放并填充到256。"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def get_brain_mask(image):
    mask = sitk.Cast(image != 0, sitk.sitkUInt8)
    mask = sitk.BinaryMorphologicalClosing(mask, [1, 1, 1])
    components = sitk.RelabelComponent(sitk.ConnectedComponent(mask))
    if sitk.GetArrayViewFromImage(components).max() < 1:
        raise ValueError("未检测到脑区")
    mask = sitk.Cast(components == 1, sitk.sitkUInt8)
    return sitk.BinaryFillhole(mask)


def get_bbox(mask, margin_mm=(10, 10, 0)):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    start, size = [x, y, z], [sx, sy, sz]

    for i in range(3):
        margin = int(np.ceil(margin_mm[i] / mask.GetSpacing()[i]))
        end = min(mask.GetSize()[i], start[i] + size[i] + margin)
        start[i] = max(0, start[i] - margin)
        size[i] = end - start[i]
    return start + size


def crop(image, bbox):
    return sitk.RegionOfInterest(image, bbox[3:], bbox[:3])


def normalize_ct(image, mask, window=(-20, 100)):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    brain = sitk.GetArrayViewFromImage(mask) > 0
    output = np.zeros_like(array)
    output[brain] = (
        np.clip(array[brain], window[0], window[1]) - window[0]
    ) / (window[1] - window[0])
    output = sitk.GetImageFromArray(output)
    output.CopyInformation(image)
    return output


def resize_and_pad(image, target=256, is_label=False):
    x, y, z = image.GetSize()
    scale = min(target / x, target / y)
    new_x, new_y = round(x * scale), round(y * scale)

    resample = sitk.ResampleImageFilter()
    resample.SetSize((new_x, new_y, z))
    resample.SetOutputSpacing((
        image.GetSpacing()[0] * x / new_x,
        image.GetSpacing()[1] * y / new_y,
        image.GetSpacing()[2],
    ))
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    resample.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    resized = resample.Execute(image)

    dx, dy = target - new_x, target - new_y
    left, top = dx // 2, dy // 2
    padding = [left, top, 0, dx - left, dy - top, 0]
    output = sitk.ConstantPad(resized, padding[:3], padding[3:], 0)
    return output, {
        "resized_size": [new_x, new_y, z],
        "padding": padding,
    }


def preprocess(image, label):
    if image.GetSize() != label.GetSize():
        raise ValueError("图像与标签尺寸不一致")

    label_values = np.unique(sitk.GetArrayViewFromImage(label))
    if not np.all(np.isin(label_values, [0, 1])):
        raise ValueError(f"标签不是0/1二值图: {label_values}")

    brain_mask = get_brain_mask(image)
    bbox = get_bbox(brain_mask)
    image = crop(image, bbox)
    brain_mask = crop(brain_mask, bbox)
    label = crop(sitk.Cast(label, sitk.sitkUInt8), bbox)
    image = normalize_ct(image, brain_mask)
    image, transform = resize_and_pad(image)
    label, _ = resize_and_pad(label, is_label=True)
    transform["bbox"] = bbox
    return image, label, transform


def run(image_dir, label_dir, output_dir):
    image_dir, label_dir, output_dir = map(Path, (image_dir, label_dir, output_dir))
    image_out = output_dir / "images"
    label_out = output_dir / "labels"
    meta_out = output_dir / "metadata"
    for folder in (image_out, label_out, meta_out):
        folder.mkdir(parents=True, exist_ok=True)

    image_names = {p.name for p in image_dir.glob("*.nii.gz")}
    label_names = {p.name for p in label_dir.glob("*.nii.gz")}
    if image_names != label_names:
        raise ValueError("影像与标签文件名不匹配")

    for i, name in enumerate(sorted(image_names), 1):
        image = sitk.ReadImage(str(image_dir / name))
        label = sitk.ReadImage(str(label_dir / name))
        result, mask, meta = preprocess(image, label)
        sitk.WriteImage(result, str(image_out / name), True)
        sitk.WriteImage(mask, str(label_out / name), True)
        (meta_out / name.replace(".nii.gz", ".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[{i}/{len(image_names)}] {name}")


if __name__ == "__main__":
    run(
        r"E:\My_vscode_project\Dataset\data\MyAisDataset\images_after_skull",
        r"E:\My_vscode_project\Dataset\data\MyAisDataset\Stroke_label",
        r"E:\My_vscode_project\Dataset\data\MyAisDataset\processed",
    )
