from torchvision import datasets, transforms
from base import BaseDataLoader

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from base import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MentalHealthDataset(Dataset):
    def __init__(self, image_dir, csv_path, task_name, file_list=None, transform=None):
        """
        Args:
            task_name: 
                - 'sds': 自杀风险分类
                - 'result-X': 单项症状分类 (如 result-1)
                # 1 抑郁
                # 2 愤怒
                # 3* 易激惹
                # 4 躁狂
                # 5 焦虑
                # 6 躯体不适症状
                # 7* 注意力不集中
                # 8 自杀想法
                # 9 精神病性症状
                # 10 睡眠问题
                # 11* 记忆
                # 12 重复的想法或行为
                # 13* 解离
                # 14* 人格功能
                # 15 物质使用。
                - 'overall': 整体心理健康状态分类 (新增)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path) 
        
        # 确保 Case_id 是字符串以便匹配
        self.df['Case_id'] = self.df['Case_id'].astype(str)
        self.id_to_row = {str(row['Case_id']): row for _, row in self.df.iterrows()}

        self.task_name = task_name
        # 获取目录下所有png文件
        self.full_file_list = [f for f in os.listdir(image_dir) if f.endswith('.png')] if file_list is None else file_list
        
        # 过滤有效样本
        self.samples = []
        self._filter_valid_samples()

    def _get_file_id(self, filename):
        # 解析文件名 Video1_1.png -> 1
        try:
            name_part = os.path.splitext(filename)[0] 
            case_id = name_part.split('_')[-1]
            return case_id
        except:
            return None

    def _calculate_sds_label(self, row):
        # SDS 计分逻辑
        weights = [1, 2, 6, 10, 10, 4]
        cols = [f'SDS-{i}' for i in range(1, 7)]
        score = 0
        valid_cols = 0
        
        for col, w in zip(cols, weights):
            val = row.get(col, -1)
            if val != -1:
                score += val * w
                valid_cols += 1
        
        # 如果所有SDS列都未选，视为无效样本
        if valid_cols == 0:
            return -1
            
        # >6 有风险(1), 否则无风险(0)
        return 1 if score > 6 else 0

    def _calculate_overall_label(self, row):
        """
        计算整体心理健康状态:
        1. 如果 result-1 到 result-15 中出现至少一个 1 -> Label 1
        2. 如果 全为 -1 -> Label -1 (排除)
        3. 否则 (即没有1，且不全是-1，意味着全为0或者0与-1混合) -> Label 0
        """
        cols = [f'result-{i}' for i in range(1, 16)]
        values = [row.get(c, -1) for c in cols] # 获取15列的值，缺省为-1

        if 1 in values:
            return 1
        elif all(v == -1 for v in values):
            return -1 # 排除
        else:
            return 0 # 视为健康

    def _filter_valid_samples(self):
        valid_samples = []
        for fname in self.full_file_list:
            cid = self._get_file_id(fname)
            
            # 这里的 cid 需要与 CSV 中的 Case_id 对应
            # 如果 CSV 中 ID 是 'op0802001' 而文件名解析出的是 '1'
            # 你需要在这里加一个映射逻辑，或者保证 CSV 里的 ID 也是 '1'
            if cid not in self.id_to_row:
                continue
                
            row = self.id_to_row[cid]
            label = -1
            
            if self.task_name == 'sds':
                label = self._calculate_sds_label(row)
            
            elif self.task_name == 'overall':
                # 新增的整体分类任务
                label = self._calculate_overall_label(row)
                
            elif self.task_name.startswith('result-'):
                # 单项任务 result-1 到 result-15
                col_name = self.task_name
                if col_name in row:
                    label = row[col_name]
            else:
                # 容错处理
                pass

            # 只有当标签为 0 或 1 时才加入训练（排除 -1）
            if label in [0, 1]:
                valid_samples.append((fname, label))
        
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image, label


class MentalHealthDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, task_name, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.task_name = task_name
        
        # 定义预处理
        trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
        if training:
            # 1. 解析所有文件的 Case ID 用于分组切分
            file_id_map = []
            for f in all_files:
                try:
                    cid = os.path.splitext(f)[0].split('_')[-1]
                    file_id_map.append({'file': f, 'group': cid})
                except:
                    continue
            
            df_files = pd.DataFrame(file_id_map)
            
            # 2. 按 Case_id 进行切分，防止同一人的数据泄露
            if validation_split > 0:
                splitter = GroupShuffleSplit(test_size=validation_split, n_splits=1, random_state=42)
                train_idx, val_idx = next(splitter.split(df_files, groups=df_files['group']))
                
                train_files = df_files.iloc[train_idx]['file'].tolist()
                val_files = df_files.iloc[val_idx]['file'].tolist()
            else:
                train_files = df_files['file'].tolist()
                val_files = []
            # 打印训练集和验证集的样本数量
            print(f"Training samples: {len(train_files)}...")
            if val_files:
                print(f"Validation samples: {len(val_files)}...")
            else:
                print("No validation samples.")
            
            self.dataset = MentalHealthDataset(data_dir, csv_path, task_name, file_list=train_files, transform=trsfm)
            self.val_dataset = MentalHealthDataset(data_dir, csv_path, task_name, file_list=val_files, transform=trsfm) if val_files else None
        else:
            self.dataset = MentalHealthDataset(data_dir, csv_path, task_name, file_list=all_files, transform=trsfm)
            self.val_dataset = None

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def split_validation(self):
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)