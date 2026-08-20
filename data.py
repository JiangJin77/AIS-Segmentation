import logging
import torch
from pathlib import Path
from torch.utils.data import Dataset

# 日志
logger = logging.getLogger(__name__)


class TensorFlipAugment:
    """对图像、掩码张量执行同步翻转增强"""

    FLIP_MODES = {
        0: None,      # 原始
        1: [2],       # 水平翻转
        2: [1],       # 垂直翻转
        3: [1, 2],    # 水平+垂直翻转
    }

    def __init__(self, mode: int):
        self.mode = mode
        self.dims = self.FLIP_MODES.get(mode)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if self.dims is None:
            return image, mask
        return torch.flip(image, dims=self.dims), \
               torch.flip(mask, dims=self.dims)


class DataSetFFM(Dataset):
    """基于 images/masks 子目录的 PyTorch Dataset。

    支持可选的 in-memory 缓存，避免每个 epoch 重复读取 PT 文件。
    """

    def __init__(self, data_path, augment: bool = True, cache: bool = True, verbose: bool = True):
        self.data_path = Path(data_path)
        self.augment = augment
        self.cache = cache
        self.flip_modes = sorted(TensorFlipAugment.FLIP_MODES.keys()) if augment else [0]
        self.augment_factor = len(self.flip_modes)

        self.data_list = self._load_data_list()
        self.total_len = len(self.data_list) * self.augment_factor

        # 预构建增强对象，避免 __getitem__ 中重复 new
        self._flip_augments = {mode: TensorFlipAugment(mode=mode) for mode in self.flip_modes}

        # in-memory 缓存
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        if self.cache:
            self._preload_to_memory()

        if verbose:
            logger.info('从 %s 中导入 %d 个文件', self.data_path, len(self.data_list))
            if self.cache:
                logger.info('已启用内存缓存，共 %d 组数据', len(self.data_list))
                self._log_cache_size()
            logger.info('数据总数: %d', self.total_len)

    def _log_cache_size(self):
        """估算缓存占用内存大小"""
        if not self._cache:
            return
        sample = next(iter(self._cache.values()))
        total_bytes = 0
        for key in ('image', 'mask'):
            total_bytes += sample.get(key, torch.empty(0)).element_size() * \
                           max(sample.get(key, torch.empty(0)).numel(), 1)
        avg_item_bytes = total_bytes / len(self._cache) if len(self._cache) > 1 else total_bytes
        total_items = len(self._cache)
        logger.info('缓存估算: ~%.1f MB (%d items, ~%.0f KB/item)',
                    avg_item_bytes * total_items / 1024 / 1024,
                    total_items,
                    avg_item_bytes / 1024)

    def _preload_to_memory(self):
        """预加载所有 PT 数据到内存中"""
        for entry in self.data_list:
            name = entry['base_name']
            self._cache[name] = {
                'image': self._load_pt(entry['image']),
                'mask': self._load_pt(entry['mask']),
            }

    def _get_tensors(self, name):
        """根据缓存配置从内存或磁盘读取数据"""
        if self.cache:
            cached = self._cache[name]
            return cached['image'].clone(), cached['mask'].clone()
        entry = next(e for e in self.data_list if e['base_name'] == name)
        return self._load_pt(entry['image']), \
               self._load_pt(entry['mask'])

    def _load_data_list(self):
        """加载 images、masks 目录中的公共文件名。"""
        images_dir = self.data_path / 'images'
        masks_dir = self.data_path / 'masks'

        for directory in (images_dir, masks_dir):
            if not directory.exists():
                logger.warning('目录不存在: %s', directory)
                return []

        image_names = {path.stem for path in images_dir.glob('*.pt')}
        mask_names = {path.stem for path in masks_dir.glob('*.pt')}

        common_names = sorted(image_names & mask_names)
        return [
            {
                'base_name': name,
                'image': images_dir / f'{name}.pt',
                'mask': masks_dir / f'{name}.pt',
            }
            for name in common_names
        ]

    # 确保张量的数据类型和形状
    @staticmethod
    def _load_pt(path):
        return torch.load(path, weights_only=True)

    # 确保张量的数据类型正确，并且不包含梯度信息
    @staticmethod
    def _ensure_tensor(value, dtype):
        tensor = torch.as_tensor(value)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.detach()

    # 如果图像或掩码是二维或三维的，添加一个通道维度
    @staticmethod
    def _add_channel_dim(tensor: torch.Tensor):
        if tensor.ndim in {2, 3}:
            return tensor.unsqueeze(0)
        return tensor

    # 验证掩码张量的值
    @staticmethod
    def _validate_mask(mask: torch.Tensor):
        if mask.min() < 0 or mask.max() > 2:
            logger.warning('Mask contains values outside [0, 1, 2]: %s', torch.unique(mask))
            return mask.clamp(0, 2)
        return mask

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if not self.data_list:
            raise IndexError('数据集为空，无法索引。')

        data_idx = idx // self.augment_factor
        aug_mode = idx % self.augment_factor
        entry = self.data_list[data_idx]

        image, mask = self._get_tensors(entry['base_name'])

        image = self._ensure_tensor(image, torch.float32)
        mask = self._ensure_tensor(mask, torch.long)

        mask = self._validate_mask(mask)

        image = self._add_channel_dim(image)
        mask = self._add_channel_dim(mask)

        if self.augment and aug_mode != 0:
            image, mask = self._flip_augments[aug_mode](image, mask)

        file_name = f"{entry['base_name']}_aug{aug_mode}"
        return image, mask, file_name

    def __repr__(self):
        return f'{self.__class__.__name__}(data_path={self.data_path}, augment={self.augment}, total_len={self.total_len})'
