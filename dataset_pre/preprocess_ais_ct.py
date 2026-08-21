"""AIS 去颅骨 CT：3D 脑区裁剪、HU 窗截断、XY 等比例缩放及补边。"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


CT_WINDOW = (-20, 100)
TARGET_SIZE = 256


def validate_image_pair(image, label):
    """检查 CT 与标签是否为同一物理空间中的 3D 图像。"""
    if image.GetDimension() != 3 or label.GetDimension() != 3:
        raise ValueError(
            f"仅支持 3D 图像，当前维度：CT={image.GetDimension()}，"
            f"标签={label.GetDimension()}"
        )
    if image.GetSize() != label.GetSize():
        raise ValueError(
            f"图像与标签尺寸不一致：CT={image.GetSize()}，标签={label.GetSize()}"
        )

    geometry = {
        "spacing": (image.GetSpacing(), label.GetSpacing()),
        "origin": (image.GetOrigin(), label.GetOrigin()),
        "direction": (image.GetDirection(), label.GetDirection()),
    }
    for name, (image_value, label_value) in geometry.items():
        if not np.allclose(image_value, label_value, rtol=0, atol=1e-5):
            raise ValueError(
                f"图像与标签的 {name} 不一致："
                f"CT={image_value}，标签={label_value}"
            )


def validate_ct_values(image):
    """检查 CT 是否包含无效数值。"""
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("CT 包含 NaN 或 Inf")


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


def window_ct(image, mask, window=CT_WINDOW):
    """将脑区 CT 截断到固定 HU 窗，并保留 HU 数值语义。

    脑区外保持为 0，以延续去颅骨数据的背景约定。
    不执行 min-max 或 z-score 归一化，以兼容以 HU 表示的先验阈值。
    """
    lower, upper = window
    if lower >= upper:
        raise ValueError(f"无效 CT 窗：下界 {lower} 必须小于上界 {upper}")

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    brain = sitk.GetArrayViewFromImage(mask) > 0
    output = np.zeros_like(array, dtype=np.float32)
    output[brain] = np.clip(array[brain], lower, upper)
    output = sitk.GetImageFromArray(output)
    output.CopyInformation(image)
    return output


def resize_and_pad(image, target=TARGET_SIZE, is_label=False, pad_value=0):
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
    output = sitk.ConstantPad(resized, padding[:3], padding[3:], pad_value)
    return output, {
        "resized_size": [new_x, new_y, z],
        "padding": padding,
    }


def preprocess(image, label, window=CT_WINDOW, target=TARGET_SIZE):
    validate_image_pair(image, label)
    validate_ct_values(image)

    label_values = np.unique(sitk.GetArrayViewFromImage(label))
    if not np.all(np.isin(label_values, [0, 1])):
        raise ValueError(f"标签不是0/1二值图: {label_values}")

    brain_mask = get_brain_mask(image)
    bbox = get_bbox(brain_mask)
    image = crop(image, bbox)
    brain_mask = crop(brain_mask, bbox)
    label = crop(sitk.Cast(label, sitk.sitkUInt8), bbox)
    image = window_ct(image, brain_mask, window=window)
    image, transform = resize_and_pad(image, target=target, pad_value=0)
    label, _ = resize_and_pad(label, target=target, is_label=True, pad_value=0)
    output_label_values = np.unique(sitk.GetArrayViewFromImage(label))
    if not np.all(np.isin(output_label_values, [0, 1])):
        raise RuntimeError(f"重采样后标签值异常: {output_label_values}")

    transform["bbox"] = bbox
    transform["target_xy"] = [target, target]
    transform["intensity"] = {
        "unit": "HU",
        "window": list(window),
        "normalization": None,
        "background_fill": 0,
    }
    return image, label, transform


def run(image_dir, label_dir, output_dir):
    image_dir, label_dir, output_dir = map(Path, (image_dir, label_dir, output_dir))
    for name, directory in (("CT", image_dir), ("标签", label_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name}目录不存在: {directory}")

    image_out = output_dir / "images"
    label_out = output_dir / "masks"
    meta_out = output_dir / "metadata"
    for folder in (image_out, label_out, meta_out):
        folder.mkdir(parents=True, exist_ok=True)

    image_names = {p.name for p in image_dir.glob("*.nii.gz")}
    label_names = {p.name for p in label_dir.glob("*.nii.gz")}
    if not image_names:
        raise ValueError(f"CT目录中没有 .nii.gz 文件: {image_dir}")
    if image_names != label_names:
        missing_labels = sorted(image_names - label_names)
        missing_images = sorted(label_names - image_names)
        raise ValueError(
            "影像与标签文件名不匹配；"
            f"缺少标签={missing_labels[:10]}，缺少影像={missing_images[:10]}"
        )

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
