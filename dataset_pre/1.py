# 去颅骨后的数据预处理
# 1. 去除背景; 2.# 
# 剪切去除背景，重采样到统一大小
import os
import numpy as np
import SimpleITK as sitk

def resize_image(image, new_size=(128, 128, 128), resamplemethod="Mask"):
    """
    将图像重采样到指定大小，默认(128, 128, 128)。
    :param image: 输入图像
    :param new_size: 输出图像大小
    :param resamplemethod: 插值方法
    :return: 缩放后的图像
    """
    # 获取输入图像信息：尺寸、体素间距、原点位置、方向矩阵
    size_ori = image.GetSize()  
    ori_Spacing = image.GetSpacing()    
    ori_origin = image.GetOrigin()  
    ori_direction = image.GetDirection()    
    # (D, H, W) --> (x, y, z)
    new_x, new_y, new_z = int(new_size[2]), int(new_size[1]), int(new_size[0])
    # 计算新的体素间距
    spacing_new = (ori_Spacing[0] * (size_ori[0] / new_x),
                   ori_Spacing[1] * (size_ori[1] / new_y),
                   ori_Spacing[2] * (size_ori[2] / new_z))
    # 创建重采样滤波器并设置基本参数
    resampler = sitk.ResampleImageFilter()  
    resampler.SetReferenceImage(image)
    resampler.SetSize((new_x, new_y, new_z))
    resampler.SetOutputSpacing(spacing_new)
    resampler.SetOutputOrigin(ori_origin)
    resampler.SetOutputDirection(ori_direction)
    # 插值与像素类型根据用途选择
    if resamplemethod == "Mask":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # 使用最近邻插值
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # 使用线性插值
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    image_r = resampler.Execute(image)
    return image_r

def threshold_based_crop_and_bg_median(image):
    """
    使用Otsu 阈值估计器将图像背景和前景分离。
    在医学图像中，背景通常为空气。然后使用前景的轴对齐边界框进行剪裁，并计算背景中位数的强度。
    :param image: SimpleITK image，一个图像，其中前景和背景的强度分布是二模态的（Otsu's 方法的假设）。
    :return: 裁剪后的图像，背景中位数强度值。
    """
    
    inside_value = 0
    outside_value = 255  # 值0通常表示背景，1或255表示前景
    bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)

    # 计算背景统计信息
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image, image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)

    # 获取解剖结构的边界框
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)

    # 根据边界框裁剪图像并返回结果
    return bounding_box, sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                          bounding_box[0:int(len(bounding_box) / 2)])

def batch_crop_and_resample(dataOriPath, dataDstPath, labelOriPath, labelDstPath):
    # 读取数据和标签文件列表
    datafiles = os.listdir(dataOriPath)
    datafiles.sort(key=lambda x: int(x.replace('.nii.gz', '')) if x.endswith('.nii.gz') else float('inf'))
    labelfiles = os.listdir(labelOriPath)
    labelfiles.sort(key=lambda x: int(x.replace('.nii.gz', '')) if x.endswith('.nii.gz') else float('inf'))
    # 遍历数据和标签文件进行裁剪和重采样
    for nii, nii_1 in zip(datafiles, labelfiles):
        image = sitk.ReadImage(os.path.join(dataOriPath, nii))
        label = sitk.ReadImage(os.path.join(labelOriPath, nii_1))
        # 裁剪数据、标签
        bounding_box, croppedData = threshold_based_crop_and_bg_median(image)   
        croppedLabel = sitk.RegionOfInterest(label, bounding_box[int(len(bounding_box) / 2):],
                                  bounding_box[0:int(len(bounding_box) / 2)])
        array = sitk.GetArrayFromImage(croppedData) # 获取裁剪后数据的数组形式
        # 重采样数据、标签到指定大小
        image = resize_image(croppedData, new_size=(np.shape(array)[0], 256, 256), resamplemethod="Data")
        label = resize_image(croppedLabel, new_size=(np.shape(array)[0], 256, 256), resamplemethod="Mask")
        dstDatapath = os.path.join(dataDstPath, nii)
        dstLabelpath = os.path.join(labelDstPath, nii_1)
        sitk.WriteImage(image, dstDatapath)
        print(dstDatapath)
        sitk.WriteImage(label, dstLabelpath)
        print(dstLabelpath)

if __name__ == '__main__':
    dataOriPath = r"E:\My_vscode_project\Dataset\data\ssct"   # 原始数据路径
    dataDstPath = r"E:\My_vscode_project\Dataset\data\niigz\images"    # 裁剪后的数据保存路径
    labelOriPath = r"E:\My_vscode_project\Dataset\data\labels"  # 原始标签路径
    labelDstPath = r"E:\My_vscode_project\Dataset\data\niigz\labels"  # 裁剪后的标签保存路径

    batch_crop_and_resample(dataOriPath, dataDstPath,labelOriPath,labelDstPath)

