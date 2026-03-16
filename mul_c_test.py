import csv
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc
import torch.nn.functional as F
from scipy import stats
import math
import time
from typing import Optional, Tuple
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 标签定义
BINARY_CLASSES = ['正常', '异常']
MULTI_CLASS_LABELS = {
    0: '肢体导联低电压',
    1: '下壁异常Q',
    2: 'ST改变',
    3: 'T波改变',
    4: '房性早搏',
    5: '室性早搏',
    6: '颤动',
    7: '一度房室阻滞',
    8: '左、右束支阻滞',
    9: '左束支分支阻滞',
    10: '电轴左偏',
    11: '电轴右偏'
}


# 可视化函数类
class ECGVisualizer:
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_training_history(self, trainer, model_type="二分类"):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_type}模型训练历史', fontsize=16, fontweight='bold')

        # 损失曲线
        axes[0, 0].plot(trainer.train_losses, label='训练损失', color='blue', linewidth=2)
        axes[0, 0].plot(trainer.val_losses, label='验证损失', color='red', linewidth=2)
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[0, 1].plot(trainer.train_accuracies, label='训练准确率', color='blue', linewidth=2)
        axes[0, 1].plot(trainer.val_accuracies, label='验证准确率', color='red', linewidth=2)
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1分数曲线
        if hasattr(trainer, 'val_f1_scores') and trainer.val_f1_scores:
            axes[1, 0].plot(trainer.val_f1_scores, label='验证F1分数', color='green', linewidth=2)
            axes[1, 0].set_title('验证F1分数')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1分数')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 学习率曲线（如果可用）
        if hasattr(trainer, 'scheduler'):
            lrs = []
            # 这里简化处理，实际应该记录每个epoch的学习率
            for i in range(len(trainer.train_losses)):
                lrs.append(trainer.scheduler.get_last_lr()[0] if i == 0 else lrs[-1])
            axes[1, 1].plot(lrs, label='学习率', color='purple', linewidth=2)
            axes[1, 1].set_title('学习率变化')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('学习率')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{model_type}_训练历史.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels, title="混淆矩阵", model_type=""):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_type}{title}', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{model_type}_混淆矩阵.png', dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_metrics_comparison(self, metrics_dict, title="模型性能比较"):
        """绘制多个模型的性能指标比较"""
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [metrics_dict[model][metric] for model in models]

            bars = ax.bar(models, values, color=self.colors[:len(models)], alpha=0.7)
            ax.set_title(f'{metric}比较')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)

            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('模型性能比较.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_distribution(self, labels, title="类别分布", model_type=""):
        """绘制类别分布图"""
        if isinstance(labels[0], str):
            # 如果是字符串标签（如最终预测）
            label_counts = Counter(labels)
            classes = list(label_counts.keys())
            counts = list(label_counts.values())
        else:
            # 如果是数值标签
            label_counts = Counter(labels)
            classes = [MULTI_CLASS_LABELS.get(i, f'类别{i}') for i in label_counts.keys()]
            counts = list(label_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, counts, color=self.colors[:len(classes)], alpha=0.7)
        plt.title(f'{model_type}{title}', fontsize=16, fontweight='bold')
        plt.xlabel('类别')
        plt.ylabel('样本数量')
        plt.xticks(rotation=45, ha='right')

        # 在柱子上显示数量
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{model_type}_类别分布.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_f1(self, y_true, y_pred, labels, title="精确率-召回率-F1分数", model_type=""):
        """绘制每个类别的精确率、召回率和F1分数"""
        # 计算每个类别的指标
        precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        # 准备数据
        if len(labels) == 2:  # 二分类
            class_names = BINARY_CLASSES
        else:  # 多分类
            class_names = [MULTI_CLASS_LABELS.get(i, f'类别{i}') for i in labels]

        x = np.arange(len(class_names))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='精确率', alpha=0.7, color='skyblue')
        plt.bar(x, recall, width, label='召回率', alpha=0.7, color='lightgreen')
        plt.bar(x + width, f1, width, label='F1分数', alpha=0.7, color='salmon')

        plt.title(f'{model_type}{title}', fontsize=16, fontweight='bold')
        plt.xlabel('类别')
        plt.ylabel('分数')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            plt.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{model_type}_精确率召回率F1.png', dpi=300, bbox_inches='tight')
        plt.show()

        return precision, recall, f1

    def plot_roc_curves(self, y_true_list, y_pred_proba_list, model_names, title="ROC曲线比较"):
        """绘制ROC曲线（需要概率预测）"""
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(10, 8))

        for i, (y_true, y_pred_proba, model_name) in enumerate(zip(y_true_list, y_pred_proba_list, model_names)):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                     color=self.colors[i], linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ROC曲线比较.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_performance_summary(self, metrics_dict, save_path="性能总结报告.html"):
        """创建性能总结报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ECG分类模型性能报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; color: #2c3e50; }
                .metric-card { 
                    background: #f8f9fa; 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin: 10px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #2980b9; 
                }
                .model-comparison { display: flex; justify-content: space-around; flex-wrap: wrap; }
                .model-section { flex: 1; min-width: 300px; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #34495e; color: white; }
                tr:hover { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 ECG分类模型性能报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>
        """.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

        for model_name, metrics in metrics_dict.items():
            html_content += f"""
            <div class="model-section">
                <h2>🎯 {model_name} 模型性能</h2>
                <div class="model-comparison">
            """

            for metric_name, value in metrics.items():
                html_content += f"""
                    <div class="metric-card">
                        <h3>{metric_name}</h3>
                        <div class="metric-value">{value:.4f}</div>
                    </div>
                """

            html_content += """
                </div>
            </div>
            """

        html_content += """
            <div class="header">
                <h3>📈 可视化图表</h3>
                <p>请查看生成的PNG图片文件以获取详细的可视化结果</p>
            </div>
        </body>
        </html>
        """

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"性能总结报告已保存至: {save_path}")


# 1. 修复的标签加载函数
def load_labels(csv_path, mode='binary'):
    """
    加载标签数据
    mode: 'binary' 或 'multi'
    """
    if mode == 'binary':
        binary_labels = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                row_vals = [float(x) for x in row]
                # 第13列表示正常，如果第13列为1则是正常，否则为异常
                if row_vals[-1] == 1:  # 最后一列是正常标签
                    binary_labels.append(0)  # 正常
                else:
                    binary_labels.append(1)  # 异常
        return binary_labels, header

    else:  # multi-class (12分类，不包含正常)
        multi_labels = []
        binary_labels = []  # 同时生成二分类标签

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row_idx, row in enumerate(reader):
                row_vals = [float(x) for x in row]

                # 检查数据长度
                if len(row_vals) < 13:
                    print(f"警告: 第{row_idx + 1}行数据长度不足: {len(row_vals)}")
                    continue

                # 第13列是正常标签
                is_normal = (row_vals[-1] == 1)

                if is_normal:
                    # 正常样本
                    binary_labels.append(0)  # 正常
                    multi_labels.append(-1)  # 多分类中标记为-1（不用于多分类训练）
                else:
                    # 异常样本 - 从前12列中找出异常类型
                    abnormal_types = []
                    for i in range(12):  # 前12列对应12种异常
                        if row_vals[i] == 1:
                            abnormal_types.append(i)

                    if abnormal_types:
                        # 如果有多个异常，选择第一个出现的异常作为主要标签
                        primary_abnormal = abnormal_types[0]
                        binary_labels.append(1)  # 异常
                        multi_labels.append(primary_abnormal)  # 主要异常标签
                    else:
                        # 没有检测到具体异常类型，但仍标记为异常
                        print(f"警告: 第{row_idx + 1}行标记为异常但没有具体异常类型")
                        binary_labels.append(1)  # 异常
                        multi_labels.append(-1)  # 不用于多分类训练

        # 统计多分类标签分布
        valid_multi_labels = [label for label in multi_labels if label != -1]
        if valid_multi_labels:
            label_dist = Counter(valid_multi_labels)
            print(f"多分类标签分布: {label_dist}")
            print(f"有效多分类样本数量: {len(valid_multi_labels)}")

        return binary_labels, multi_labels, header


# 2. 优化的数据集类
class OptimizedECGDataset(Dataset):
    def __init__(self, hdf5_path, label_indices, labels, use_magnitude=True,
                 use_phase=True, max_sequence_length=256,
                 num_classes=2, apply_channel_norm=True):
        self.hdf5_path = hdf5_path
        self.label_indices = label_indices
        self.labels = labels
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.apply_channel_norm = apply_channel_norm

        # 预计算特征维度
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
            sample_key = keys[0]
            sample_data = f[sample_key][0]

            self.sequence_length = min(sample_data.shape[0], max_sequence_length)
            self.num_channels = sample_data.shape[1]

            # 计算特征维度
            if use_magnitude and use_phase:
                self.feature_dim = 2
            else:
                self.feature_dim = 1

            print(f"序列长度: {self.sequence_length}, 通道数: {self.num_channels}, "
                  f"特征维度: {self.feature_dim}, 类别数: {self.num_classes}")

    def __len__(self):
        return len(self.label_indices)

    def process_sample(self, complex_sample):
        """优化的样本处理"""
        try:
            # 检查数据有效性
            if np.any(np.isnan(complex_sample)) or np.any(np.isinf(complex_sample)):
                complex_sample = np.nan_to_num(complex_sample)

            # 智能下采样
            if complex_sample.shape[0] > self.max_sequence_length:
                # 使用stride-based下采样
                stride = max(1, complex_sample.shape[0] // self.max_sequence_length)
                complex_sample = complex_sample[::stride]
                complex_sample = complex_sample[:self.max_sequence_length]
            elif complex_sample.shape[0] < self.max_sequence_length:
                # 填充到目标长度
                pad_length = self.max_sequence_length - complex_sample.shape[0]
                complex_sample = np.pad(complex_sample, ((0, pad_length), (0, 0)),
                                        mode='constant')

            # 提取特征
            if self.use_magnitude and self.use_phase:
                magnitude = np.abs(complex_sample).astype(np.float32)
                phase = np.angle(complex_sample).astype(np.float32)
                # 在通道维度合并
                features = np.stack([magnitude, phase], axis=-1)
            elif self.use_magnitude:
                features = np.abs(complex_sample).astype(np.float32)
                features = np.expand_dims(features, -1)
            elif self.use_phase:
                features = np.angle(complex_sample).astype(np.float32)
                features = np.expand_dims(features, -1)
            else:
                features = complex_sample.real.astype(np.float32)
                features = np.expand_dims(features, -1)

            return features
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return np.zeros((self.max_sequence_length, self.num_channels, self.feature_dim),
                            dtype=np.float32)

    def channel_wise_normalize(self, features):
        """通道级归一化"""
        if len(features.shape) == 3:
            normalized_features = np.zeros_like(features)
            for channel in range(features.shape[1]):
                for feat in range(features.shape[2]):
                    channel_data = features[:, channel, feat]

                    # 移除异常值
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)

                    # 使用Z-score归一化
                    normalized_channel = (channel_data - mean_val) / (std_val + 1e-8)

                    # 限制极端值
                    normalized_channel = np.clip(normalized_channel, -5, 5)
                    normalized_features[:, channel, feat] = normalized_channel

            return normalized_features
        return features

    def __getitem__(self, idx):
        try:
            actual_idx = self.label_indices[idx]

            with h5py.File(self.hdf5_path, 'r') as f:
                dataset_key = list(f.keys())[0]
                complex_sample = f[dataset_key][actual_idx]

                features = self.process_sample(complex_sample)

                if self.apply_channel_norm:
                    features = self.channel_wise_normalize(features)

                label = self.labels[idx]

                return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"获取数据项时出错 {idx}: {e}")
            default_features = np.zeros((self.max_sequence_length, self.num_channels, self.feature_dim),
                                        dtype=np.float32)
            return torch.FloatTensor(default_features), torch.tensor(0, dtype=torch.long)


# 3. 进一步优化的Transformer模型
class FastECGTransformer(nn.Module):
    def __init__(self, seq_len=256, num_channels=12, feature_dim=2, d_model=64,
                 nhead=8, num_layers=3, num_classes=2, dropout=0.1,
                 use_attention_pool=True):
        super(FastECGTransformer, self).__init__()

        self.d_model = d_model
        self.use_attention_pool = use_attention_pool
        self.num_classes = num_classes

        # 更高效的特征投影
        self.input_projection = nn.Sequential(
            nn.Linear(num_channels * feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_encoding = self.create_learnable_positional_encoding(seq_len, d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 注意力池化
        if use_attention_pool:
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
                nn.Softmax(dim=1)
            )

        # 更简化的分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def create_learnable_positional_encoding(self, seq_len, d_model):
        """可学习的位置编码"""
        pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(pos_encoding, std=0.02)
        return pos_encoding

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, seq_len, channels, features = x.shape

        # 重塑和投影
        x_flat = x.reshape(batch_size, seq_len, channels * features)
        x_proj = self.input_projection(x_flat)

        # 位置编码
        if seq_len <= self.pos_encoding.shape[1]:
            x = x_proj + self.pos_encoding[:, :seq_len, :]
        else:
            # 插值适应不同长度
            pos_encoding = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x_proj + pos_encoding

        # Transformer处理
        x = self.transformer(x)

        # 池化
        if self.use_attention_pool:
            attention_weights = self.attention_pool(x)
            x_pooled = torch.sum(x * attention_weights, dim=1)
        else:
            # 平均池化
            x_pooled = x.mean(dim=1)

        # 分类
        logits = self.classifier(x_pooled)

        return logits


# 4. 优化的训练器类
class FastTrainer:
    def __init__(self, model, train_loader, val_loader, num_epochs=20,
                 learning_rate=2e-3, weight_decay=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = model.num_classes

        self.model.to(self.device)
        self.setup_optimizer(learning_rate, weight_decay)

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

    def setup_optimizer(self, lr, weight_decay):
        """设置优化器和学习率调度器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 使用OneCycle学习率调度器，训练更快
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )

    def calculate_class_weights(self):
        """计算类别权重，处理缺失类别"""
        all_train_labels = []
        for _, targets in self.train_loader:
            # 确保targets有正确的形状
            if targets.dim() > 0:
                all_train_labels.extend(targets.cpu().numpy())
            else:
                # 如果是标量，直接添加
                all_train_labels.append(targets.item())

        class_counts = Counter(all_train_labels)
        total = sum(class_counts.values())

        # 处理缺失的类别
        class_weights = []
        for i in range(self.num_classes):
            if i in class_counts and class_counts[i] > 0:
                class_weights.append(total / class_counts[i])
            else:
                # 对于缺失的类别，使用平均权重
                class_weights.append(1.0)
                print(f"警告: 类别 {i} 在训练集中没有样本，使用默认权重1.0")

        class_weights = torch.tensor(class_weights, device=self.device)
        print(f"类别权重: {class_weights}")
        print(f"类别分布: {dict(class_counts)}")
        return class_weights

    def focal_loss(self, outputs, targets, alpha=None, gamma=1.5):
        """Focal Loss用于处理类别不平衡"""
        if alpha is None:
            alpha = torch.ones(self.num_classes, device=self.device)

        # 确保targets有正确的形状
        if targets.dim() == 0:
            # 如果是标量，添加批次维度
            targets = targets.unsqueeze(0)

        # 检查是否有有效的目标
        if targets.numel() == 0:
            print("警告: 目标张量为空，跳过此批次")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 确保目标和输出有相同的批次大小
        if outputs.size(0) != targets.size(0):
            print(f"警告: 输出批次大小 {outputs.size(0)} 与目标批次大小 {targets.size(0)} 不匹配")
            # 调整目标大小以匹配输出
            if outputs.size(0) > targets.size(0):
                targets = targets.repeat(outputs.size(0))
            else:
                targets = targets[:outputs.size(0)]

        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total_samples = 0
        start_time = time.time()

        # 只在第一个batch计算类别权重
        if epoch == 0:
            self.class_weights = self.calculate_class_weights()

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # 确保targets有正确的形状
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            data, targets = data.to(self.device), targets.to(self.device)

            # 跳过空批次
            if targets.numel() == 0:
                print(f"警告: 批次 {batch_idx} 目标为空，跳过")
                continue

            self.optimizer.zero_grad()
            outputs = self.model(data)

            # 使用Focal Loss
            loss = self.focal_loss(outputs, targets, alpha=self.class_weights)

            # 如果损失为0，跳过此批次
            if loss.item() == 0:
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # 每20个batch报告一次进度
            if batch_idx % 20 == 0:
                batch_acc = correct / total_samples if total_samples > 0 else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch + 1}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}, LR: {current_lr:.2e}')

        epoch_time = time.time() - start_time
        train_acc = correct / total_samples if total_samples > 0 else 0
        avg_train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0

        print(f'Epoch {epoch + 1} 训练完成, 时间: {epoch_time:.2f}s, '
              f'平均损失: {avg_train_loss:.4f}, 准确率: {train_acc:.4f}')

        return avg_train_loss, train_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                # 确保targets有正确的形状
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)

                data, targets = data.to(self.device), targets.to(self.device)

                # 跳过空批次
                if targets.numel() == 0:
                    print("警告: 验证批次目标为空，跳过")
                    continue

                outputs = self.model(data)
                loss = self.focal_loss(outputs, targets, alpha=self.class_weights)

                # 如果损失为0，跳过此批次
                if loss.item() == 0:
                    continue

                val_loss += loss.item()
                valid_batches += 1

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        if valid_batches == 0:
            print("警告: 验证集没有有效批次")
            return 0.0, 0.0, 0.0, [], []

        val_acc = accuracy_score(val_targets, val_preds) if val_targets else 0.0

        # 计算F1分数
        if self.num_classes == 2:
            # 二分类使用binary F1
            val_f1 = f1_score(val_targets, val_preds, average='binary') if val_targets else 0.0
        else:
            # 多分类使用weighted F1，考虑类别不平衡
            val_f1 = f1_score(val_targets, val_preds, average='weighted') if val_targets else 0.0

        avg_val_loss = val_loss / valid_batches

        return avg_val_loss, val_acc, val_f1, val_preds, val_targets

    def train(self):
        """完整的训练过程"""
        print(f"使用设备: {self.device}")
        print(f"分类任务，共 {self.num_classes} 个类别")

        patience = 8
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # 验证阶段
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)

            print(f'\nEpoch [{epoch + 1}/{self.num_epochs}] 总结:')
            print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
            print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 验证F1分数: {val_f1:.4f}')

            # 早停检查 - 基于准确率和F1分数
            if val_acc > self.best_val_acc or val_f1 > self.best_val_f1:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                epochs_no_improve = 0
                model_type = "binary" if self.num_classes == 2 else "multi"

                # 保存完整的检查点
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'num_classes': self.num_classes
                }, f'best_{model_type}_class_model.pth')
                print(f'✨ 保存最佳模型，验证准确率: {val_acc:.4f}, 验证F1分数: {val_f1:.4f}')
            else:
                epochs_no_improve += 1
                print(f'验证指标未提升，连续 {epochs_no_improve} 轮')

            if epochs_no_improve >= patience:
                print(f"⏹️ 早停在第 {epoch + 1} 轮")
                break

            # 清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return (self.train_losses, self.val_losses,
                self.train_accuracies, self.val_accuracies, self.val_f1_scores)


# 5. 优化的级联分类系统
class FastCascadeECGClassifier:
    def __init__(self, binary_model_path, multi_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualizer = ECGVisualizer()

        # 加载二分类模型
        self.binary_model = FastECGTransformer(
            seq_len=256, num_channels=12, feature_dim=2,
            d_model=64, nhead=8, num_layers=3, num_classes=2
        )

        # 修复PyTorch 2.6兼容性问题
        try:
            # 尝试使用weights_only=False加载
            binary_checkpoint = torch.load(binary_model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"使用weights_only=False加载失败: {e}")
            # 如果失败，尝试其他方法
            try:
                # 使用pickle加载
                import pickle
                with open(binary_model_path, 'rb') as f:
                    binary_checkpoint = pickle.load(f)
            except Exception as e2:
                print(f"使用pickle加载失败: {e2}")
                # 最后尝试使用torch.load但不设置weights_only
                binary_checkpoint = torch.load(binary_model_path, map_location=self.device)

        # 从检查点中提取模型状态字典
        if isinstance(binary_checkpoint, dict) and 'model_state_dict' in binary_checkpoint:
            binary_state_dict = binary_checkpoint['model_state_dict']
        else:
            binary_state_dict = binary_checkpoint

        self.binary_model.load_state_dict(binary_state_dict)
        self.binary_model.to(self.device)
        self.binary_model.eval()

        # 加载多分类模型
        self.multi_model = FastECGTransformer(
            seq_len=256, num_channels=12, feature_dim=2,
            d_model=64, nhead=8, num_layers=3, num_classes=12
        )

        # 修复PyTorch 2.6兼容性问题
        try:
            # 尝试使用weights_only=False加载
            multi_checkpoint = torch.load(multi_model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"使用weights_only=False加载失败: {e}")
            # 如果失败，尝试其他方法
            try:
                # 使用pickle加载
                import pickle
                with open(multi_model_path, 'rb') as f:
                    multi_checkpoint = pickle.load(f)
            except Exception as e2:
                print(f"使用pickle加载失败: {e2}")
                # 最后尝试使用torch.load但不设置weights_only
                multi_checkpoint = torch.load(multi_model_path, map_location=self.device)

        # 从检查点中提取模型状态字典
        if isinstance(multi_checkpoint, dict) and 'model_state_dict' in multi_checkpoint:
            multi_state_dict = multi_checkpoint['model_state_dict']
        else:
            multi_state_dict = multi_checkpoint

        self.multi_model.load_state_dict(multi_state_dict)
        self.multi_model.to(self.device)
        self.multi_model.eval()

        print("快速级联分类器加载完成")

    def predict(self, data_loader):
        """快速预测"""
        all_binary_preds = []
        all_final_preds = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                # 确保targets有正确的形状
                if targets.dim() > 0:
                    targets = targets.cpu().numpy()
                else:
                    targets = [targets.item()]

                # 第一步：二分类
                binary_outputs = self.binary_model(data)
                binary_probs = F.softmax(binary_outputs, dim=1)
                binary_confidences, binary_preds = torch.max(binary_probs, dim=1)
                binary_preds = binary_preds.cpu().numpy()

                # 第二步：多分类
                multi_outputs = self.multi_model(data)
                multi_probs = F.softmax(multi_outputs, dim=1)
                multi_confidences, multi_preds = torch.max(multi_probs, dim=1)
                multi_preds = multi_preds.cpu().numpy()

                # 合并结果
                final_preds = []
                for i, (binary_pred, multi_pred) in enumerate(zip(binary_preds, multi_preds)):
                    if binary_pred == 0:  # 正常
                        final_pred = "正常"
                    else:  # 异常
                        final_pred = MULTI_CLASS_LABELS.get(multi_pred, f"未知异常{multi_pred}")

                    final_preds.append(final_pred)

                all_binary_preds.extend(binary_preds)
                all_final_preds.extend(final_preds)
                all_targets.extend(targets)

        return all_binary_preds, all_final_preds, all_targets


# 6. 评估函数
def evaluate_fast_classifier(classifier, test_loader, true_labels, visualizer):
    """评估分类器"""
    print("开始快速分类评估...")

    binary_preds, final_preds, _ = classifier.predict(test_loader)

    # 将预测结果转换为数值标签
    pred_numeric = []
    for pred in final_preds:
        if pred == "正常":
            pred_numeric.append(12)  # 正常标记为12
        else:
            # 找到对应的异常标签
            found = False
            for label_id, label_name in MULTI_CLASS_LABELS.items():
                if pred == label_name:
                    pred_numeric.append(label_id)
                    found = True
                    break
            if not found:
                pred_numeric.append(-1)  # 未知标签

    # 计算二分类准确率
    binary_true = [1 if label != 12 else 0 for label in true_labels]
    binary_acc = accuracy_score(binary_true, binary_preds)
    binary_f1 = f1_score(binary_true, binary_preds, average='binary')
    binary_precision = precision_score(binary_true, binary_preds, average='binary')
    binary_recall = recall_score(binary_true, binary_preds, average='binary')

    print(f"二分类准确率: {binary_acc:.4f}")
    print(f"二分类F1分数: {binary_f1:.4f}")
    print(f"二分类精确率: {binary_precision:.4f}")
    print(f"二分类召回率: {binary_recall:.4f}")

    # 可视化二分类结果
    visualizer.plot_confusion_matrix(binary_true, binary_preds, [0, 1],
                                     title="二分类混淆矩阵", model_type="级联系统")
    visualizer.plot_precision_recall_f1(binary_true, binary_preds, [0, 1],
                                        title="二分类性能指标", model_type="级联系统")

    # 计算多分类准确率（只考虑异常样本）
    abnormal_indices = [i for i, t in enumerate(true_labels) if t != 12]
    if abnormal_indices:
        abnormal_true = [true_labels[i] for i in abnormal_indices]
        abnormal_pred = [pred_numeric[i] for i in abnormal_indices]
        # 只计算有效预测
        valid_abnormal_indices = [i for i in range(len(abnormal_pred)) if abnormal_pred[i] != -1]
        if valid_abnormal_indices:
            valid_abnormal_true = [abnormal_true[i] for i in valid_abnormal_indices]
            valid_abnormal_pred = [abnormal_pred[i] for i in valid_abnormal_indices]
            multi_acc = accuracy_score(valid_abnormal_true, valid_abnormal_pred)
            multi_f1 = f1_score(valid_abnormal_true, valid_abnormal_pred, average='weighted')
            multi_precision = precision_score(valid_abnormal_true, valid_abnormal_pred, average='weighted',
                                              zero_division=0)
            multi_recall = recall_score(valid_abnormal_true, valid_abnormal_pred, average='weighted', zero_division=0)

            print(f"异常样本多分类准确率: {multi_acc:.4f}")
            print(f"异常样本多分类F1分数: {multi_f1:.4f}")
            print(f"异常样本多分类精确率: {multi_precision:.4f}")
            print(f"异常样本多分类召回率: {multi_recall:.4f}")

            # 可视化多分类结果
            multi_labels = list(range(12))
            visualizer.plot_confusion_matrix(valid_abnormal_true, valid_abnormal_pred, multi_labels,
                                             title="多分类混淆矩阵", model_type="级联系统")
            visualizer.plot_precision_recall_f1(valid_abnormal_true, valid_abnormal_pred, multi_labels,
                                                title="多分类性能指标", model_type="级联系统")

    # 总体准确率
    overall_acc = accuracy_score(true_labels, pred_numeric)
    overall_f1 = f1_score(true_labels, pred_numeric, average='weighted')
    overall_precision = precision_score(true_labels, pred_numeric, average='weighted', zero_division=0)
    overall_recall = recall_score(true_labels, pred_numeric, average='weighted', zero_division=0)

    print(f"总体准确率: {overall_acc:.4f}")
    print(f"总体F1分数: {overall_f1:.4f}")
    print(f"总体精确率: {overall_precision:.4f}")
    print(f"总体召回率: {overall_recall:.4f}")

    # 可视化最终预测分布
    visualizer.plot_class_distribution(final_preds, title="最终预测分布", model_type="级联系统")

    # 详细分类报告
    target_names = list(MULTI_CLASS_LABELS.values()) + ['正常']
    print("\n详细分类报告:")
    print(classification_report(true_labels, pred_numeric,
                                labels=list(range(13)),
                                target_names=target_names, zero_division=0))

    return (overall_acc, overall_f1, overall_precision, overall_recall,
            final_preds, true_labels, binary_acc, binary_f1, binary_precision, binary_recall)


# 7. 训练二分类模型
def train_binary_classifier():
    """训练二分类模型"""
    print("=== 训练二分类模型 ===")

    # 加载二分类标签
    binary_labels, header = load_labels('D:/our_data/label-15439.csv', mode='binary')

    # 数据分割
    indices = list(range(len(binary_labels)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=binary_labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[binary_labels[i] for i in temp_idx], random_state=42
    )

    train_labels = [binary_labels[i] for i in train_idx]
    val_labels = [binary_labels[i] for i in val_idx]

    # 创建数据集和数据加载器
    train_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        train_idx, train_labels, num_classes=2
    )
    val_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        val_idx, val_labels, num_classes=2
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 创建和训练模型
    model = FastECGTransformer(num_classes=2)
    trainer = FastTrainer(model, train_loader, val_loader, num_epochs=20)
    trainer.train()

    # 可视化训练历史
    visualizer = ECGVisualizer()
    visualizer.plot_training_history(trainer, model_type="二分类")

    print("二分类模型训练完成")
    return trainer


# 8. 训练多分类模型
def train_multi_classifier():
    """训练多分类模型（12分类，只包含异常）"""
    print("=== 训练多分类模型 (12分类) ===")

    # 加载标签数据
    binary_labels, multi_labels, header = load_labels('D:/our_data/label-15439.csv', mode='multi')

    # 只使用异常样本训练多分类模型（multi_labels中不为-1的样本）
    abnormal_indices = [i for i, label in enumerate(multi_labels) if label != -1]
    abnormal_multi_labels = [multi_labels[i] for i in abnormal_indices]

    print(f"用于多分类训练的异常样本数量: {len(abnormal_multi_labels)}")
    print("异常类别分布:", Counter(abnormal_multi_labels))

    # 可视化类别分布
    visualizer = ECGVisualizer()
    visualizer.plot_class_distribution(abnormal_multi_labels, title="训练集类别分布", model_type="多分类")

    # 数据分割
    train_idx, val_idx = train_test_split(
        abnormal_indices, test_size=0.2, stratify=abnormal_multi_labels, random_state=42
    )

    train_labels = [multi_labels[i] for i in train_idx]
    val_labels = [multi_labels[i] for i in val_idx]

    # 创建数据集和数据加载器
    train_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        train_idx, train_labels, num_classes=12
    )
    val_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        val_idx, val_labels, num_classes=12
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # 减小批量大小
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)  # 减小批量大小

    # 创建和训练模型
    model = FastECGTransformer(num_classes=12)
    trainer = FastTrainer(model, train_loader, val_loader, num_epochs=20)
    trainer.train()

    # 可视化训练历史
    visualizer.plot_training_history(trainer, model_type="多分类")

    print("多分类模型训练完成")
    return trainer


# 9. 单独评估二分类和多分类模型
def evaluate_individual_models():
    """单独评估二分类和多分类模型"""
    print("=== 单独评估二分类和多分类模型 ===")

    visualizer = ECGVisualizer()

    # 加载标签数据
    binary_labels, multi_labels, header = load_labels('D:/our_data/label-15439.csv', mode='multi')

    # 准备二分类测试数据
    indices = list(range(len(binary_labels)))
    _, test_idx = train_test_split(
        indices, test_size=0.3, stratify=binary_labels, random_state=42
    )

    test_binary_labels = [binary_labels[i] for i in test_idx]
    test_binary_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        test_idx, test_binary_labels, num_classes=2
    )
    test_binary_loader = DataLoader(test_binary_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 准备多分类测试数据（只包含异常样本）
    abnormal_indices = [i for i, label in enumerate(multi_labels) if label != -1]
    _, test_abnormal_idx = train_test_split(
        abnormal_indices, test_size=0.3, stratify=[multi_labels[i] for i in abnormal_indices], random_state=42
    )
    test_multi_labels = [multi_labels[i] for i in test_abnormal_idx]
    test_multi_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        test_abnormal_idx, test_multi_labels, num_classes=12
    )
    test_multi_loader = DataLoader(test_multi_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 评估二分类模型
    print("\n--- 二分类模型评估 ---")
    binary_model = FastECGTransformer(num_classes=2)

    # 修复PyTorch 2.6兼容性问题
    try:
        # 尝试使用weights_only=False加载
        binary_checkpoint = torch.load('best_binary_class_model.pth', map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"使用weights_only=False加载失败: {e}")
        # 如果失败，尝试其他方法
        try:
            # 使用pickle加载
            import pickle
            with open('best_binary_class_model.pth', 'rb') as f:
                binary_checkpoint = pickle.load(f)
        except Exception as e2:
            print(f"使用pickle加载失败: {e2}")
            # 最后尝试使用torch.load但不设置weights_only
            binary_checkpoint = torch.load('best_binary_class_model.pth', map_location='cpu')

    # 从检查点中提取模型状态字典
    if isinstance(binary_checkpoint, dict) and 'model_state_dict' in binary_checkpoint:
        binary_state_dict = binary_checkpoint['model_state_dict']
    else:
        binary_state_dict = binary_checkpoint

    binary_model.load_state_dict(binary_state_dict)
    binary_model.eval()

    binary_preds = []
    binary_targets = []
    with torch.no_grad():
        for data, targets in test_binary_loader:
            outputs = binary_model(data)
            _, predicted = torch.max(outputs, 1)
            binary_preds.extend(predicted.cpu().numpy())
            binary_targets.extend(targets.cpu().numpy())

    binary_acc = accuracy_score(binary_targets, binary_preds)
    binary_f1 = f1_score(binary_targets, binary_preds, average='binary')
    binary_precision = precision_score(binary_targets, binary_preds, average='binary')
    binary_recall = recall_score(binary_targets, binary_preds, average='binary')

    print(f"二分类模型准确率: {binary_acc:.4f}")
    print(f"二分类模型F1分数: {binary_f1:.4f}")
    print(f"二分类模型精确率: {binary_precision:.4f}")
    print(f"二分类模型召回率: {binary_recall:.4f}")

    # 可视化二分类结果
    visualizer.plot_confusion_matrix(binary_targets, binary_preds, [0, 1],
                                     title="混淆矩阵", model_type="二分类")
    visualizer.plot_precision_recall_f1(binary_targets, binary_preds, [0, 1],
                                        title="性能指标", model_type="二分类")

    print("二分类详细报告:")
    print(classification_report(binary_targets, binary_preds, target_names=BINARY_CLASSES))

    # 评估多分类模型
    print("\n--- 多分类模型评估 ---")
    multi_model = FastECGTransformer(num_classes=12)

    # 修复PyTorch 2.6兼容性问题
    try:
        # 尝试使用weights_only=False加载
        multi_checkpoint = torch.load('best_multi_class_model.pth', map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"使用weights_only=False加载失败: {e}")
        # 如果失败，尝试其他方法
        try:
            # 使用pickle加载
            import pickle
            with open('best_multi_class_model.pth', 'rb') as f:
                multi_checkpoint = pickle.load(f)
        except Exception as e2:
            print(f"使用pickle加载失败: {e2}")
            # 最后尝试使用torch.load但不设置weights_only
            multi_checkpoint = torch.load('best_multi_class_model.pth', map_location='cpu')

    # 从检查点中提取模型状态字典
    if isinstance(multi_checkpoint, dict) and 'model_state_dict' in multi_checkpoint:
        multi_state_dict = multi_checkpoint['model_state_dict']
    else:
        multi_state_dict = multi_checkpoint

    multi_model.load_state_dict(multi_state_dict)
    multi_model.eval()

    multi_preds = []
    multi_targets = []
    with torch.no_grad():
        for data, targets in test_multi_loader:
            outputs = multi_model(data)
            _, predicted = torch.max(outputs, 1)
            multi_preds.extend(predicted.cpu().numpy())
            multi_targets.extend(targets.cpu().numpy())

    multi_acc = accuracy_score(multi_targets, multi_preds)
    multi_f1 = f1_score(multi_targets, multi_preds, average='weighted')
    multi_precision = precision_score(multi_targets, multi_preds, average='weighted', zero_division=0)
    multi_recall = recall_score(multi_targets, multi_preds, average='weighted', zero_division=0)

    print(f"多分类模型准确率: {multi_acc:.4f}")
    print(f"多分类模型F1分数: {multi_f1:.4f}")
    print(f"多分类模型精确率: {multi_precision:.4f}")
    print(f"多分类模型召回率: {multi_recall:.4f}")

    # 可视化多分类结果
    multi_labels_list = list(range(12))
    visualizer.plot_confusion_matrix(multi_targets, multi_preds, multi_labels_list,
                                     title="混淆矩阵", model_type="多分类")
    visualizer.plot_precision_recall_f1(multi_targets, multi_preds, multi_labels_list,
                                        title="性能指标", model_type="多分类")

    print("多分类详细报告:")
    print(classification_report(multi_targets, multi_preds,
                                target_names=list(MULTI_CLASS_LABELS.values()),
                                zero_division=0))

    # 创建性能比较
    metrics_dict = {
        '二分类模型': {
            '准确率': binary_acc,
            'F1分数': binary_f1,
            '精确率': binary_precision,
            '召回率': binary_recall
        },
        '多分类模型': {
            '准确率': multi_acc,
            'F1分数': multi_f1,
            '精确率': multi_precision,
            '召回率': multi_recall
        }
    }

    visualizer.plot_metrics_comparison(metrics_dict, title="二分类 vs 多分类模型性能比较")

    return (binary_acc, binary_f1, binary_precision, binary_recall,
            multi_acc, multi_f1, multi_precision, multi_recall)


# 10. 主函数
def main():
    """主函数：训练级联分类器并评估"""
    print("=== 快速ECG信号级联分类系统 ===")

    visualizer = ECGVisualizer()

    # 检查是否已有训练好的模型
    import os
    has_binary_model = os.path.exists('best_binary_class_model.pth')
    has_multi_model = os.path.exists('best_multi_class_model.pth')

    # 训练模型（如果不存在）
    if not has_binary_model:
        print("训练二分类模型...")
        binary_trainer = train_binary_classifier()

    if not has_multi_model:
        print("训练多分类模型...")
        multi_trainer = train_multi_classifier()

    # 单独评估二分类和多分类模型
    (binary_acc, binary_f1, binary_precision, binary_recall,
     multi_acc, multi_f1, multi_precision, multi_recall) = evaluate_individual_models()

    # 加载完整的测试数据
    print("准备测试数据...")
    binary_labels, multi_labels, header = load_labels('D:/our_data/label-15439.csv', mode='multi')

    # 使用完整数据进行测试（包括正常和异常）
    indices = list(range(len(binary_labels)))
    _, test_idx = train_test_split(
        indices, test_size=0.3, stratify=binary_labels, random_state=42
    )

    # 获取测试集的真实多分类标签（用于评估）
    test_true_labels = [multi_labels[i] for i in test_idx]
    # 对于测试集，我们需要将正常样本的标签设为12，异常样本保持原标签
    test_true_labels_numeric = []
    for i, label in enumerate(test_true_labels):
        if label == -1:  # 正常样本
            test_true_labels_numeric.append(12)
        else:  # 异常样本
            test_true_labels_numeric.append(label)

    # 创建测试数据集
    test_binary_labels = [binary_labels[i] for i in test_idx]
    test_dataset = OptimizedECGDataset(
        'D:/our_data/ECG-Tracing-1.5w-fft.hdf5',
        test_idx, test_binary_labels, num_classes=2
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 创建级联分类器并评估
    print("创建快速级联分类器...")
    classifier = FastCascadeECGClassifier('best_binary_class_model.pth', 'best_multi_class_model.pth')

    print("开始评估...")
    (overall_acc, overall_f1, overall_precision, overall_recall,
     final_preds, targets, binary_acc_cascade, binary_f1_cascade,
     binary_precision_cascade, binary_recall_cascade) = evaluate_fast_classifier(
        classifier, test_loader, test_true_labels_numeric, visualizer)

    print(f"\n最终结果:")
    print(f"级联分类系统总体准确率: {overall_acc:.4f}")
    print(f"级联分类系统总体F1分数: {overall_f1:.4f}")
    print(f"级联分类系统总体精确率: {overall_precision:.4f}")
    print(f"级联分类系统总体召回率: {overall_recall:.4f}")

    # 创建完整的性能总结
    metrics_summary = {
        '二分类模型': {
            '准确率': binary_acc,
            'F1分数': binary_f1,
            '精确率': binary_precision,
            '召回率': binary_recall
        },
        '多分类模型': {
            '准确率': multi_acc,
            'F1分数': multi_f1,
            '精确率': multi_precision,
            '召回率': multi_recall
        },
        '级联系统': {
            '准确率': overall_acc,
            'F1分数': overall_f1,
            '精确率': overall_precision,
            '召回率': overall_recall
        }
    }

    visualizer.plot_metrics_comparison(metrics_summary, title="完整模型性能比较")
    visualizer.create_performance_summary(metrics_summary, "ECG分类性能报告.html")

    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()