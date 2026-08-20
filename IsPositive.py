"""读取标签文件，查看是否为阳性
读取文件夹中的标签.nii.gz文件，查看是否为阳性标签(==4)，若为阳性则输出文件名"""
import os
import numpy as np
import SimpleITK as sitk

def check_positive_labels(folder_path):
    total = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, filename)
            label = sitk.ReadImage(file_path, sitk.sitkInt16)
            label_array = sitk.GetArrayFromImage(label)
            if np.any(label_array == 4):
                print(filename)
                total += 1
    return total


if __name__ == "__main__":
    folder_path = r"E:\My_vscode_project\Dataset\data\MyAisDataset\labels"
    total = check_positive_labels(folder_path)
    print(f"Total number of positive labels: {total}")