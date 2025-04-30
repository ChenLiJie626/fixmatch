import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你把 mish 和 PSBatchNorm2d 放在 models.wideresnet 中，也可以这样导入：
# from models.wideresnet import mish, PSBatchNorm2d

logger = logging.getLogger(__name__)

def mish(x):
    """Mish 激活函数"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """带偏置的 BatchNorm，复用你给的代码"""
    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        # 在标准 BN 输出的基础上加上一个常量 alpha
        return super().forward(x) + self.alpha


class MNISTNet(nn.Module):
    """一个适用于 MNIST（28×28 单通道）的轻量级卷积网络"""
    def __init__(self, num_classes=10, dropout=0.5):
        super(MNISTNet, self).__init__()
        # --------------------------------------------------------------------
        # 1. 卷积层 & BN & 激活
        # --------------------------------------------------------------------
        # 输入通道 =1（灰度图），输出通道 =32，3×3 卷积，padding=1 保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 对应的 PSBatchNorm
        self.bn1 = PSBatchNorm2d(32)
        
        # 第二个卷积：32→64，3×3，padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = PSBatchNorm2d(64)
        
        # 第三个卷积：64→128，3×3，padding=1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = PSBatchNorm2d(128)
        
        # --------------------------------------------------------------------
        # 2. 全连接层 & BN & 激活 & Dropout
        # --------------------------------------------------------------------
        # 注意此时特征图尺寸已经被两次 2×2 池化降到 7×7，通道数 128
        # 因此展平后输入维度 =128*7*7
        self.fc1 = nn.Linear(128 * 7 * 7, 256, bias=False)
        # 对 fc1 的输出做普通一维 BN
        self.bn4 = nn.BatchNorm1d(256)
        # 最终分类层：256→num_classes
        self.fc2 = nn.Linear(256, num_classes)
        
        # 随机失活比例
        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------------------------
        # 3. 权重初始化
        # --------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, PSBatchNorm2d)):
                # BN 层权重初始化为 1，偏置为 0
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用 Xavier 初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        前向传播：
        输入 x 形状 [B,1,28,28]，输出 logits 形状 [B,num_classes]
        """
        # 第一块：Conv1 -> BN -> Mish
        x = mish(self.bn1(self.conv1(x)))
        # 第二块：Conv2 -> BN -> Mish -> 2×2 最大池化
        x = mish(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)    # [B,64,14,14]
        # 第三块：Conv3 -> BN -> Mish -> 2×2 最大池化
        x = mish(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)    # [B,128,7,7]

        # 展平成 [B,128*7*7]
        x = x.view(x.size(0), -1)
        # 全连接1 -> BN -> Mish -> Dropout
        x = mish(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        # 最后一层线性分类
        x = self.fc2(x)
        return x


def build_mnist_model(num_classes=10, dropout=0.5):
    """
    构造并返回模型的工厂函数，便于和 wideresnet/resnext 的接口保持一致
    """
    logger.info(f"Model: MNISTNet(num_classes={num_classes}, dropout={dropout})")
    return MNISTNet(num_classes=num_classes, dropout=dropout)
