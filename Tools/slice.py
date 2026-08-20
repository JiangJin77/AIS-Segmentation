"""
将预处理过的.nii.gz文件转换为.pt格式的切片张量并保存
"""
import os
import SimpleITK as sitk
import torch

def get_slice_tensor(file_path, save_file, flag='data'):
    """将nii.gz文件转换为切片张量并保存
        :param file_path: 输入数据集路径
        :param save_file: 输出数据集路径
        :param flag: 数据集类型
    """
    if flag not in {'data', 'label'}:
        raise ValueError(f"无效标签: {flag}")

    os.makedirs(save_file, exist_ok=True)
    files = [f for f in os.listdir(file_path) if f.endswith(".nii.gz")]
    for file in files:
        image = sitk.ReadImage(os.path.join(file_path, file))    # 获取数据
        if flag == 'data':
            image_array = sitk.GetArrayFromImage(image)    # 转换为numpy数组 （Z, Y, X)
            image_tensor = torch.from_numpy(image_array)     # 转换为张量   （Z, Y, X) = (D, H, W)
        elif flag == 'label':
            image_array = sitk.GetArrayFromImage(image)
            image_tensor = torch.from_numpy(image_array).to(torch.uint8)# 转换为 uint8 保存

        for i in range(image_tensor.shape[0]):
            slice_tensor = image_tensor[i, :, :]
            slice_save_path = os.path.join(save_file, f"{file.replace('.nii.gz', '')}_{i}.pt")
            torch.save(slice_tensor, slice_save_path) 


if __name__ == "__main__":
    data_path = r"E:\JiangJin\dataset\train_test\data"
    label_path = r"E:\JiangJin\dataset\train_test\labelmap"
    data_save_path = r"E:\My_vscode_project\Guidance\Project_1\dataset\5folds\tensor\images"
    label_save_path = r"E:\My_vscode_project\Guidance\Project_1\dataset\5folds\tensor\masks"
    get_slice_tensor(file_path=data_path, 
                     save_file=data_save_path, 
                     flag='data')
    get_slice_tensor(file_path=label_path, 
                     save_file=label_save_path,
                     flag='label')
    print("数据集处理完成！")