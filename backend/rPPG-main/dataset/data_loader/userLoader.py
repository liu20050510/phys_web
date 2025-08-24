import os
import cv2
import numpy as np
import torch
from dataset.data_loader.BaseLoader import BaseLoader


class SingleVideoLoader(BaseLoader):
    """用于加载单个视频的dataloader（不涉及 ground truth）"""

    def __init__(self, name, data_path, config_data):
        """
        初始化单个视频dataloader.

        Args:
            name (str): dataloader的名称.
            data_path (str): 存放视频文件的文件夹路径.
            config_data (CfgNode): 数据设置参数.
        """
        self.data_path = data_path
        self.processed_data = []  # 初始化为空列表
        print(f"Initializing SingleVideoLoader with data_path: {data_path}")
        super().__init__(name, data_path, config_data)

    def __getitem__(self, index):
        """
        返回形状为 (1, D, C, H, W) 的数据，其中 D 为每个切片的帧数。
        这里假设 self.processed_data 是一个列表，每个元素形状为 (D, H, W, C)。
        """
        # 取出对应的切片
        chunk = self.processed_data[index]  # shape: (D, H, W, C)

        # 调整维度： (D, H, W, C) -> (D, C, H, W)
        chunk = np.transpose(chunk, (0, 3, 1, 2))

        # 增加 batch 维度，变为 (1, D, C, H, W)
        chunk = np.expand_dims(chunk, axis=0)

        print("item shape:", chunk.shape)
        return torch.tensor(chunk, dtype=torch.float32)

    def __len__(self):
        return len(self.processed_data)

    def get_raw_data(self, data_path):
        """
        对于单个视频的情况，直接返回一个列表，其中包含一个字典，
        字典中的 "path" 指向存放视频文件的文件夹，"index" 可以设为一个固定标识
        """
        return [{"index": "user_video", "path": data_path}]

    def split_raw_data(self, data_dirs, begin, end):
        """
        对于单视频的情况，不需要划分数据集，直接返回原始列表即可。
        """
        return data_dirs

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """适用于单个视频的简化预处理：读取完整视频并按 CHUNK_LENGTH 切片"""
        video_path = self.get_video_path()  # 获取单个视频文件路径
        if not video_path:
            print(f"Error: No valid video found in {self.data_path}")
            return

        # 读取完整视频帧
        frames = self.read_video(video_path, config_preprocess)
        print(f"Loaded {len(frames)} frames from {video_path}")
        print("Frames shape:", frames.shape)  # 调试：检查帧形状

        if len(frames) == 0:
            print(f"Error: No frames read from {video_path}")
            return

        # 确保 frames 的形状为 (T, H, W, C)
        T, H, W, C = frames.shape
        chunk_len = config_preprocess.CHUNK_LENGTH  # 每个切片的帧数

        # 计算切片数量（不足一片的也算一片，在 __getitem__ 中会用零补齐，但这里提前补齐以便后续使用）
        n_chunks = (T + chunk_len - 1) // chunk_len

        # 按照 chunk_len 切片，并对最后不足的部分进行 zero padding
        sliced_data = []
        for i in range(n_chunks):
            start = i * chunk_len
            end_idx = start + chunk_len
            if end_idx > T:
                # 不足一片，补零
                pad_len = end_idx - T
                pad = np.zeros((pad_len, H, W, C), dtype=frames.dtype)
                chunk = np.concatenate((frames[start:], pad), axis=0)
            else:
                chunk = frames[start:end_idx]
            sliced_data.append(chunk)

        # 更新 processed_data 为切片后的列表，preprocessed_data_len 更新为切片数量
        self.processed_data = sliced_data
        self.preprocessed_data_len = len(sliced_data)
        print(f"Preprocessed data into {self.preprocessed_data_len} chunks of length {chunk_len}")

        # 构造文件列表字典，字典的每个 key 对应一个切片的信息
        file_list_dict = {}
        for i in range(n_chunks):
            file_list_dict[i] = [{'path': video_path, 'index': f'chunk_{i}'}]
        self.build_file_list(file_list_dict)  # 传递包含所有切片的字典

    def get_video_path(self):
        """
        获取文件夹中的视频文件路径.

        假设视频文件格式为 .avi、.mp4 或 .mov.

        Returns:
            str: 视频文件的完整路径.
        """
        video_extensions = ['.avi', '.mp4', '.mov']
        for file in os.listdir(self.data_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                return os.path.join(self.data_path, file)
        raise ValueError(f"在路径 {self.data_path} 中未找到视频文件!")

    @staticmethod
    def read_video(video_path, config_preprocess):
        """
        读取视频文件，返回视频帧数组，形状为 (T, H, W, 3)。

        Args:
            video_path (str): 视频文件的路径.

        Returns:
            np.array: 包含视频帧的数组.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened() and count < 150:  # 只读取 150 帧
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        cap.release()
        return np.array(frames)

    def load_data(self):
        """
        加载视频帧数据.

        Returns:
            np.array: 视频帧数组.
        """
        if not self.processed_data:
            print(
                "Warning: Processed data is empty. Ensure preprocess_dataset was called.")
        print(f"Preprocessed data length: {len(self.processed_data)}")
        return self.processed_data  # 返回已处理的数据

    def load_preprocessed_data(self, config_data=None):
        """
        直接使用 preprocess_dataset 处理的数据，而不是从文件读取。
        """
        if not self.processed_data:
            raise ValueError(
                "No preprocessed data available. Ensure preprocess_dataset() was called.")

        self.inputs = [self.processed_data]  # 作为输入
        self.labels = [None]  # 没有 ground truth
        print(f"SingleVideoLoader loaded {len(self.inputs[0])} frames for inference.")


class RealtimeLoader(BaseLoader):
    """用于处理实时帧数据的dataloader"""

    def __init__(self, name, frames_data, config_data):
        """
        初始化实时帧数据加载器

        Args:
            name (str): dataloader的名称
            frames_data (list): 帧数据列表，每个元素是numpy数组格式的图像帧
            config_data (CfgNode): 数据设置参数
        """
        self.frames_data = frames_data
        self.processed_data = []
        self.name = name
        self.config_data = config_data
        super().__init__(name, "", config_data)
        self.preprocess_frames(frames_data, config_data.PREPROCESS)

    def preprocess_frames(self, frames_data, config_preprocess):
        """预处理实时帧数据"""
        if len(frames_data) == 0:
            print("Warning: No frames data provided.")
            return

        # 转换帧数据为numpy数组
        frames = np.array(frames_data)
        print(f"Loaded {len(frames)} frames for real-time processing")
        print("Frames shape:", frames.shape)

        # 确保 frames 的形状为 (T, H, W, C)
        if frames.ndim == 3:
            # 单帧情况 (H, W, C) -> (1, H, W, C)
            frames = np.expand_dims(frames, axis=0)

        T, H, W, C = frames.shape
        chunk_len = config_preprocess.CHUNK_LENGTH  # 每个切片的帧数

        # 计算切片数量
        n_chunks = (T + chunk_len - 1) // chunk_len

        # 按照 chunk_len 切片，并对最后不足的部分进行 zero padding
        sliced_data = []
        for i in range(n_chunks):
            start = i * chunk_len
            end_idx = start + chunk_len
            if end_idx > T:
                # 不足一片，补零
                pad_len = end_idx - T
                pad = np.zeros((pad_len, H, W, C), dtype=frames.dtype)
                chunk = np.concatenate((frames[start:], pad), axis=0)
            else:
                chunk = frames[start:end_idx]
            sliced_data.append(chunk)

        # 更新 processed_data 为切片后的列表
        self.processed_data = sliced_data
        self.preprocessed_data_len = len(sliced_data)
        print(f"Preprocessed real-time data into {self.preprocessed_data_len} chunks of length {chunk_len}")

    def __getitem__(self, index):
        """
        返回形状为 (1, D, C, H, W) 的数据
        """
        # 取出对应的切片
        chunk = self.processed_data[index]  # shape: (D, H, W, C)

        # 调整维度： (D, H, W, C) -> (D, C, H, W)
        chunk = np.transpose(chunk, (0, 3, 1, 2))

        # 增加 batch 维度，变为 (1, D, C, H, W)
        chunk = np.expand_dims(chunk, axis=0)

        print("Real-time item shape:", chunk.shape)
        return torch.tensor(chunk, dtype=torch.float32)

    def __len__(self):
        return len(self.processed_data)

    def get_raw_data(self, data_path):
        return [{"index": "realtime", "path": data_path}]

    def split_raw_data(self, data_dirs, begin, end):
        return data_dirs

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        # 实时数据不需要这个方法
        pass

    def load_data(self):
        if not self.processed_data:
            print("Warning: Processed data is empty.")
        print(f"RealtimeLoader preprocessed data length: {len(self.processed_data)}")
        return self.processed_data

    def load_preprocessed_data(self, config_data=None):
        self.inputs = [self.processed_data]
        self.labels = [None]
        print(f"RealtimeLoader loaded {len(self.inputs[0])} frames for inference.")
